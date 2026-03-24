"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem

KEY DIFFERENCE FROM fourier_2d.py and fourier_2d_time.py:
  - This model treats 2D space + time together as a single 3D volume.
  - Input:  u(x, y, t=1..10)  — the first 10 timesteps of the flow field
  - Output: u(x, y, t=11..50) — ALL next 40 timesteps predicted at once
  - There is NO autoregressive loop (unlike fourier_2d_time.py).
    The model sees 10 frames and directly outputs 40 frames in one forward pass.

WHY NO AUTOREGRESSIVE LOOP?
  Both files solve the Navier-Stokes equation but use fundamentally different strategies:

  fourier_2d_time.py — Autoregressive (step by step):
    - The model only operates in 2D space (x, y).
    - Time is handled by repeatedly applying the model:
        predict step t → feed prediction back as input → predict step t+1 → ...
    - It is like a recurrence: the model is a "one-step-ahead" function applied 40 times.
    - Problem: errors compound — a small mistake at t=5 gets fed back and amplifies at
      t=6, t=7, etc. That is why InstanceNorm2d was needed to stabilize the rollout.

  fourier_3d.py — Direct space-time (all at once):
    - The model operates in 3D space-time (x, y, t) simultaneously.
    - The entire 40-timestep future is treated as a 3D volume and predicted in
      one single forward pass.
    - There is no feedback loop, so errors cannot compound.
    - Tradeoff: the model must learn full temporal dynamics internally rather than
      just one step at a time, so width=20 (smaller than fourier_2d's width=64)
      is used to keep GPU memory manageable.

  Summary:
    +-----------------------+----------------------+----------------------+
    |                       | fourier_2d_time.py   | fourier_3d.py        |
    +-----------------------+----------------------+----------------------+
    | Spatial dims modeled  | 2D (x, y)            | 3D (x, y, t)         |
    | Time handling         | Loop: 1 step, repeat | Single forward pass  |
    | Error compounding     | Yes (needs InstNorm) | No                   |
    | Memory cost           | Lower per step       | Higher (3D volume)   |
    +-----------------------+----------------------+----------------------+
"""

from timeit import default_timer

import torch.nn.functional as F

from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

################################################################
# 3d fourier layers
################################################################


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        NEW vs SpectralConv2d: We now have THREE frequency dimensions (x, y, t)
        and FOUR weight tensors instead of two.

        WHY THE NUMBER OF WEIGHTS DIFFERS ACROSS 1D / 2D / 3D:
          The root cause is which dimensions get full FFT vs half FFT (rfft).

          Rule: rfft/rfft2/rfftn always applies the half-FFT to the LAST
          dimension only, and full FFT to all other dimensions.

            Half FFT (last dim)  → only POSITIVE frequencies → 1 region to keep
            Full FFT (other dims) → POSITIVE + NEGATIVE frequencies → 2 regions to keep

          Applying this rule:
            +----------------+----------+---------------------------+---------------+
            | File           | FFT used | Dims (full=2, half=1)     | Total weights |
            +----------------+----------+---------------------------+---------------+
            | fourier_1d.py  | rfft     | x (last → half → 1)      |  1            |
            | fourier_2d.py  | rfft2    | x (full→2), y (last→1)   |  2 x 1 = 2   |
            | fourier_3d.py  | rfftn    | x (full→2), y (full→2),  |  2 x 2 x 1=4 |
            |                |          | t (last → half → 1)       |               |
            +----------------+----------+---------------------------+---------------+

          Why does the last dimension get the half treatment?
            For real-valued input, the FFT has conjugate symmetry: F[-k] = conj(F[k]).
            Negative frequencies carry no new information — they are the complex
            conjugate of the positive ones. PyTorch exploits this for the last
            dimension only to halve memory usage.

          So the number of weights doubles each time a non-last spatial dimension
          is added (one that gets full FFT):
            1D → 2D: x becomes a full-FFT dim  → x2  → 2 weights
            2D → 3D: y becomes a full-FFT dim  → x2  → 4 weights

        Why four weights in 3D specifically?
          After rfftn, the Fourier tensor has shape:
            [batch, channel, Nx, Ny, Nt//2+1]
          - The LAST dimension (time/z) uses rfft so it only has POSITIVE
            frequencies: indices 0, 1, ..., Nt//2.
          - The FIRST two dimensions (x, y) use full fft so they have both
            POSITIVE frequencies (indices 0..k_max-1) and
            NEGATIVE frequencies (indices -k_max..-1, stored at the END).

          This gives 2x2 = 4 combinations of x and y frequency corners:
            weights1: (+x, +y)  top-left in x-y plane
            weights2: (-x, +y)  bottom-left in x-y plane
            weights3: (+x, -y)  top-right in x-y plane
            weights4: (-x, -y)  bottom-right in x-y plane

          Diagram of the 3D Fourier space (one slice at fixed t):
                  y: 0..k_max-1     y: -k_max..-1
                 +---------------+---------------+
          x:     |               |               |
          0..    |   weights1    |   weights3    |
          k_max-1|  (+x freq)    |  (+x freq)    |
                 |  (+y freq)    |  (-y freq)    |
                 +---------------+---------------+
          x:     |               |               |
          -k_max |   weights2    |   weights4    |
          ..-1   |  (-x freq)    |  (-x freq)    |
                 |  (+y freq)    |  (-y freq)    |
                 +---------------+---------------+
          All 4 corners are taken along t: 0..modes3-1 (positive time freqs only)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # number of x-frequency modes to keep (k_max for x)
        self.modes2 = modes2  # number of y-frequency modes to keep (k_max for y)
        self.modes3 = modes3  # number of t-frequency modes to keep (k_max for t)

        self.scale = 1 / (in_channels * out_channels)

        # Four learnable complex weight tensors, one per corner of the x-y Fourier plane.
        # Shape: [in_channels, out_channels, modes1, modes2, modes3]
        # dtype=torch.cfloat: complex float, because Fourier coefficients are complex numbers.
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # Einstein summation for batched channel mixing per Fourier mode.
        # input:   (batch, in_channel,  x, y, t)
        # weights: (in_channel, out_channel, x, y, t)
        # output:  (batch, out_channel, x, y, t)
        #
        # For each spatial position (x,y,t) and each batch sample b,
        # it computes: output[b, o, x, y, t] = sum_i input[b, i, x, y, t] * weights[i, o, x, y, t]
        # This is the learnable linear transform R in the paper's Equation (5),
        # now applied in 3D Fourier space.
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # --- Step 1: Apply 3D real FFT ---
        # rfftn computes the n-dimensional real FFT along the last 3 dimensions.
        # Input  x shape: [batch, channel, Nx, Ny, Nt]
        # Output x_ft shape: [batch, channel, Nx, Ny, Nt//2+1]  (complex)
        #
        # Why rfftn?
        #   - For real-valued input, the FFT has conjugate symmetry:
        #     F[-k] = conj(F[k]), so we only need to store half the spectrum.
        #   - rfftn exploits this for the LAST dimension only, halving memory.
        #   - The first two dims (x, y) are stored in full (both pos & neg freqs).
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # --- Step 2: Multiply the four corners of the Fourier space ---
        # We keep only low-frequency modes (up to k_max) to learn the global
        # structure of the solution. High-frequency modes are zeroed out.
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Corner 1: positive x-freq (0..modes1-1), positive y-freq (0..modes2-1), low t-freq
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )

        # Corner 2: negative x-freq (-modes1..-1), positive y-freq (0..modes2-1), low t-freq
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )

        # Corner 3: positive x-freq (0..modes1-1), negative y-freq (-modes2..-1), low t-freq
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )

        # Corner 4: negative x-freq (-modes1..-1), negative y-freq (-modes2..-1), low t-freq
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # --- Step 3: Inverse 3D real FFT back to physical space ---
        # s=(x.size(-3), x.size(-2), x.size(-1)) explicitly tells irfftn the
        # original signal size so it correctly reconstructs the last dimension
        # from the half-spectrum.
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        # NEW vs fourier_2d.py: uses Conv3d (kernel_size=1) instead of Conv2d.
        # A 3D conv with kernel_size=1 is equivalent to a pointwise linear transform
        # applied independently at every (x, y, t) voxel — same idea as before,
        # just extended to 3D.
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)  # GELU non-linearity between the two linear layers
        x = self.mlp2(x)
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        NEW vs fourier_2d.py: The input now has 13 channels:
          - 10 channels: the Navier-Stokes velocity field at t=1,2,...,10
                         (stacked initial snapshots, repeated for each output time)
          - 3 channels:  the (x, y, t) grid coordinates

        input shape:  (batchsize, x=64, y=64, t=40, c=13)
          Note: the 10 initial snapshots are REPLICATED across all 40 output timesteps
          so the network always knows "where we started" at every output time.

        output shape: (batchsize, x=64, y=64, t=40, c=1)
          The model directly predicts all 40 future timesteps in one shot —
          no autoregressive rollout needed.
        """

        self.modes1 = modes1  # Fourier modes for x dimension
        self.modes2 = modes2  # Fourier modes for y dimension
        self.modes3 = modes3  # Fourier modes for t dimension
        self.width = width  # hidden channel dimension d_v
        self.padding = 6  # pad along the TIME axis (non-periodic in time)
        # The flow field is periodic in x,y but not in time.

        # Lifting layer: maps 13 input channels → width hidden channels
        # input channel is 13: u(1..10, x, y) stacked + x-coord + y-coord + t-coord
        self.p = nn.Linear(13, self.width)

        # 4 SpectralConv3d layers (the K operator in u' = (W + K)(u))
        self.conv0 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv1 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv2 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv3 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )

        # 4 MLP layers (pointwise non-linear transform after each spectral layer)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)

        # 4 local linear layers (the W operator in u' = (W + K)(u))
        # Conv3d with kernel_size=1 = pointwise linear transform (no spectral mixing)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        # Projection layer: maps width hidden channels → 1 output channel (velocity scalar)
        self.q = MLP(self.width, 1, self.width * 4)

    def forward(self, x):
        # x shape on entry: [batch, x=64, y=64, t=40, 10]
        # (10 = number of initial snapshots repeated across 40 output time steps)

        # --- Step 1: Append grid coordinates as extra input channels ---
        # get_grid returns [batch, x, y, t, 3]: the (x, y, t) coordinates for every voxel
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # → [batch, x, y, t, 13]

        # --- Step 2: Lift to hidden channel dimension ---
        x = self.p(x)  # → [batch, x, y, t, width]
        # Permute to PyTorch channel-first format for Conv3d/SpectralConv3d
        x = x.permute(0, 4, 1, 2, 3)  # → [batch, width, x, y, t]

        # --- Step 3: Pad along the time dimension ---
        # F.pad(x, [0, self.padding]) pads the LAST dimension (time) at the END.
        # This adds self.padding=6 zeros at the end of the time axis.
        # Purpose: the Navier-Stokes solution is NOT periodic in time, so without
        # padding, the Fourier transform would assume periodicity and introduce
        # wrap-around artifacts (Gibbs phenomenon) at the time boundaries.
        x = F.pad(x, [0, self.padding])  # → [batch, width, x, y, t+6]

        # --- Step 4: Four FNO layers (spectral conv + MLP + local linear + GELU) ---
        # Each layer applies: x = GELU(MLP(SpectralConv(x)) + W(x))
        # The last layer omits GELU (no activation before projection).

        x1 = self.conv0(x)  # global spectral mixing (K operator)
        x1 = self.mlp0(x1)  # pointwise non-linear transform
        x2 = self.w0(x)  # local linear bypass (W operator)
        x = x1 + x2  # skip connection: u' = K(u) + W(u)
        # NONLINEAR ACTIVATION (GELU):
        # Both SpectralConv (K) and the bypass Conv3d (W) are LINEAR operations.
        # Without this activation, stacking 4 such layers would still be equivalent
        # to a single linear operator — the network would collapse:
        #   W3*(W2*(W1*x)) = W_combined * x  (just one matrix multiply)
        # No matter how many layers, a stack of linear transforms cannot learn the
        # nonlinear dynamics of fluid flow (Navier-Stokes is a nonlinear PDE).
        #
        # GELU breaks this collapse by applying a smooth nonlinear function:
        #   - Large positive values: pass through almost unchanged
        #   - Large negative values: suppressed toward zero
        #   - Near zero: smooth transition (no sharp corner like ReLU)
        #
        # After GELU, the output can no longer be reduced to one linear operation,
        # giving the network the expressive power to model nonlinear physics.
        #
        # NOTE: GELU is applied after layers 1-3 but NOT after layer 4 (the last
        # FNO layer). The final layer feeds directly into the projection head (self.q),
        # so no activation is needed — we want the raw linear output there.
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)  # nonlinear activation — same role as above

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)  # nonlinear activation — same role as above

        x1 = self.conv3(x)  # last FNO layer — no GELU after this one
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # --- Step 5: Remove the time padding ---
        # x[..., :-self.padding] strips the 6 zero-padded time steps that were
        # added in Step 3, restoring the original time dimension size (T=40).
        x = x[..., : -self.padding]  # → [batch, width, x, y, t=40]

        # --- Step 6: Project hidden channels → 1 output channel ---
        x = self.q(x)  # → [batch, 1, x, y, t=40]

        # Permute back to (batch, x, y, t, channel) format
        x = x.permute(0, 2, 3, 4, 1)  # → [batch, x=64, y=64, t=40, 1]
        return x

    def get_grid(self, shape, device):
        """
        Build a 3D coordinate grid (x, y, t) to append to the input.
        NEW vs fourier_2d.py: three grids (gridx, gridy, gridz) instead of two.

        shape: (batchsize, size_x, size_y, size_z)  where size_z = T = 40
        Returns: [batchsize, size_x, size_y, size_z, 3]  (3 = x + y + t coords)

        WHY DOES EVERY VOXEL (Volumetric Pixel) NEED A COORDINATE VALUE?
          The model needs to know WHERE in (x, y, t) space each voxel sits.
          This means storing a coordinate value at every single voxel position,
          not just once per axis. For example:

            voxel (i=0, j=0, k=0) → x=0.0,  y=0.0,  t=0.0
            voxel (i=0, j=1, k=0) → x=0.0,  y=0.33, t=0.0
            voxel (i=2, j=3, k=1) → x=0.67, y=1.0,  t=0.33

          Each voxel [b, i, j, k] has five indices:
            b → which sample in the batch      (0..batchsize-1)
            i → which x-position on the grid   (0..size_x-1)
            j → which y-position on the grid   (0..size_y-1)
            k → which timestep                 (0..size_z-1)
            last dim → channel slot that holds the coordinate value

        WHY DOES gridx NEED A y-DIMENSION?
          gridx stores the x-coordinate for every voxel. Even though its value
          only depends on i (not j or k), it must physically exist at every
          (i, j, k) position so its shape matches gridy and gridz for
          torch.cat() to work. The value is simply repeated (copied) across
          all j and k positions.

          Example: for i=2, gridx stores 0.67 at ALL (j, k) combinations:
            gridx[b, 2, 0, 0, 0] = 0.67
            gridx[b, 2, 1, 0, 0] = 0.67   ← same value, different j
            gridx[b, 2, 3, 1, 0] = 0.67   ← same value, different j and k

        THE reshape + repeat PATTERN:
          The trick is to put the coordinate values in the right dimension via
          reshape (size 1 everywhere else), then copy across all other dims via repeat:

            gridx → reshape: [1, Nx, 1,  1,  1]  ← Nx values in x-dim (dim 1)
                    repeat:  [B, 1,  Ny, Nz, 1]  ← copy across batch, y, t
            gridy → reshape: [1, 1,  Ny, 1,  1]  ← Ny values in y-dim (dim 2)
                    repeat:  [B, Nx, 1,  Nz, 1]  ← copy across batch, x, t
            gridz → reshape: [1, 1,  1,  Nz, 1]  ← Nz values in t-dim (dim 3)
                    repeat:  [B, Nx, Ny, 1,  1]  ← copy across batch, x, y

          All three end up with shape [batchsize, Nx, Ny, Nz, 1].

        THE LAST DIMENSION (size 1 → size 3 after cat):
          Each individual grid has last dim = 1, acting as a channel slot that
          holds one coordinate value per voxel. After concatenating all three:
            gridx [B, Nx, Ny, Nz, 1]  ← x-coord in slot 0
            gridy [B, Nx, Ny, Nz, 1]  ← y-coord in slot 0   → [B, Nx, Ny, Nz, 3]
            gridz [B, Nx, Ny, Nz, 1]  ← t-coord in slot 0

          At every voxel [b, i, j, k, :], the 3 values are:
            [x-coord, y-coord, t-coord] = [i/(Nx-1), j/(Ny-1), k/(Nz-1)]
        """
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]

        # x-coordinate: linspace values placed in dim 1 (x), repeated across all other dims
        # gridx[b, i, j, k, 0] = linspace[i]  for all b, j, k
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1]
        )
        # shape: [batchsize, size_x, size_y, size_z, 1]

        # y-coordinate: linspace values placed in dim 2 (y), repeated across all other dims
        # gridy[b, i, j, k, 0] = linspace[j]  for all b, i, k
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1]
        )
        # shape: [batchsize, size_x, size_y, size_z, 1]

        # t-coordinate: linspace values placed in dim 3 (t), repeated across all other dims
        # gridz[b, i, j, k, 0] = linspace[k]  for all b, i, j
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1]
        )
        # shape: [batchsize, size_x, size_y, size_z, 1]

        # Concatenate along the last dimension: 1+1+1 = 3 coordinate channels
        # result[b, i, j, k, :] = [x-coord, y-coord, t-coord]
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
        # Output shape: [batchsize, size_x, size_y, size_z, 3]


################################################################
# configs
################################################################

# Two separate .mat files for train and test (unlike fourier_2d which uses one file).
TRAIN_PATH = "data/ns_data_V100_N1000_T50_1.mat"
TEST_PATH = "data/ns_data_V100_N1000_T50_2.mat"

ntrain = 1000  # number of training samples
ntest = 200  # number of test samples

modes = 8  # Fourier modes to keep in each of x, y, t (same for all 3 dims here)
width = 20  # hidden channel dimension (smaller than fourier_2d's 64 — 3D is expensive)

batch_size = 10  # small batch size — 3D volumes are memory-intensive
learning_rate = 0.001
epochs = 500
iterations = epochs * (ntrain // batch_size)  # total gradient steps: 500 * 100 = 50000

# Paths for saving model and results (auto-named from hyperparameters)
path = (
    "ns_fourier_3d_N"
    + str(ntrain)
    + "_ep"
    + str(epochs)
    + "_m"
    + str(modes)
    + "_w"
    + str(width)
)
path_model = "model/" + path
path_train_err = "results/" + path + "train.txt"
path_test_err = "results/" + path + "test.txt"
path_image = "image/" + path

runtime = np.zeros(
    2,
)
t1 = default_timer()

sub = 1  # no spatial subsampling (use full 64x64 grid)
S = 64 // sub  # spatial resolution: 64 x 64
T_in = 10  # number of INPUT timesteps (the "initial condition" window)
T = 40  # number of OUTPUT timesteps to predict
# Comment: T=40 for V1e-3 (viscosity 1e-3); T=20 for V1e-4; T=10 for V1e-5
# Higher viscosity → smoother flow → easier to predict longer horizon

################################################################
# load data
################################################################

# Load training data from the first .mat file.
# 'u' field shape: [N, x=64, y=64, T_total=50]
reader = MatReader(TRAIN_PATH)
# train_a: first T_in=10 timesteps as the initial condition snapshot
#   shape: [1000, 64, 64, 10]
train_a = reader.read_field("u")[:ntrain, ::sub, ::sub, :T_in]
# train_u: next T=40 timesteps as the prediction target
#   shape: [1000, 64, 64, 40]
train_u = reader.read_field("u")[:ntrain, ::sub, ::sub, T_in : T + T_in]

# Load test data from the SECOND .mat file (completely separate dataset).
reader = MatReader(TEST_PATH)
test_a = reader.read_field("u")[
    -ntest:, ::sub, ::sub, :T_in
]  # shape: [200, 64, 64, 10]
test_u = reader.read_field("u")[
    -ntest:, ::sub, ::sub, T_in : T + T_in
]  # shape: [200, 64, 64, 40]

print(train_u.shape)  # expected: torch.Size([1000, 64, 64, 40])
print(test_u.shape)  # expected: torch.Size([200, 64, 64, 40])
assert S == train_u.shape[-2]  # verify spatial size matches S=64
assert T == train_u.shape[-1]  # verify temporal size matches T=40

# --- Normalize inputs and outputs (prevent data leakage: fit on train only) ---
a_normalizer = UnitGaussianNormalizer(
    train_a
)  # compute mean/std from training inputs only
train_a = a_normalizer.encode(train_a)  # normalize training inputs
test_a = a_normalizer.encode(test_a)  # apply same stats to test inputs (no leakage)

y_normalizer = UnitGaussianNormalizer(
    train_u
)  # compute mean/std from training targets only
train_u = y_normalizer.encode(train_u)  # normalize training targets
# Note: test_u is NOT normalized here — it is decoded at evaluation time instead.

# --- Reshape initial condition to match the 3D input format ---
# The model input shape is [batch, x, y, t_out=40, 10].
# We replicate (repeat) the 10 initial snapshots along the time output axis,
# so at every output timestep the network always has access to the full initial state.
#
# Before: train_a shape = [1000, 64, 64, 10]
# After:  train_a shape = [1000, 64, 64, 40, 10]
#
# Step-by-step:
#   reshape(ntrain, S, S, 1, T_in)  → [1000, 64, 64, 1, 10]  (insert a dim for T_out)
#   repeat([1,1,1,T,1])             → [1000, 64, 64, 40, 10]  (replicate T=40 times)
train_a = train_a.reshape(ntrain, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, T, 1])

# Wrap data in DataLoaders for batching
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False
)

t2 = default_timer()

print("preprocessing finished, time used:", t2 - t1)
device = torch.device(
    "cuda"
)  # this file assumes CUDA; replace with 'mps' or 'cpu' for Mac

################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width).cuda()  # instantiate 3D FNO on GPU
print(count_params(model))  # print total number of trainable parameters

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)

# Move the output normalizer to GPU so we can decode predictions on the GPU
y_normalizer.cuda()

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()  # x: [batch, 64, 64, 40, 10], y: [batch, 64, 64, 40]

        optimizer.zero_grad()
        # Forward pass: model outputs [batch, 64, 64, 40, 1]
        # .view() reshapes to [batch, S=64, S=64, T=40] for loss computation
        out = model(x).view(batch_size, S, S, T)

        # MSE loss (for monitoring only — computed in normalized space)
        mse = F.mse_loss(out, y, reduction="mean")
        # mse.backward()  ← commented out: MSE is monitored but NOT used for training

        # Decode both prediction and target from normalized space back to physical units
        # before computing the relative L2 loss (training objective).
        # This ensures the loss reflects physical error magnitudes.
        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)

        # Relative L2 loss (training objective): compare in physical space
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()  # backpropagate through the L2 loss

        optimizer.step()  # update model weights
        scheduler.step()  # decay learning rate (cosine annealing)
        train_mse += mse.item()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).view(batch_size, S, S, T)
            out = y_normalizer.decode(out)  # decode prediction to physical space
            # y (test_u) was never normalized, so it is already in physical space
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)  # average MSE per batch
    train_l2 /= ntrain  # average relative L2 per training sample
    test_l2 /= ntest  # average relative L2 per test sample

    t2 = default_timer()
    print(ep, t2 - t1, train_mse, train_l2, test_l2)
# torch.save(model, path_model)  # optionally save the trained model

################################################################
# per-sample evaluation after training
################################################################

# Allocate tensor to store predictions for all 200 test samples
pred = torch.zeros(test_u.shape)  # shape: [200, 64, 64, 40]
index = 0

# Reload test set with batch_size=1 to evaluate one sample at a time
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False
)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x)
        out = y_normalizer.decode(out)  # decode prediction to physical space
        pred[index] = out  # store prediction for this sample (shape: [64, 64, 40])

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1

# Save all predictions to a .mat file for post-processing / visualization
scipy.io.savemat("pred/" + path + ".mat", mdict={"pred": pred.cpu().numpy()})
