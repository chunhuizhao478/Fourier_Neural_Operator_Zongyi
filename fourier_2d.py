"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

from timeit import default_timer

import torch.nn.functional as F

from utilities3 import *

torch.manual_seed(0)  # fix PyTorch random seed for reproducibility
np.random.seed(0)  # fix NumPy random seed for reproducibility


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # number of Fourier modes to keep in x-direction, at most floor(N/2) + 1
        self.modes2 = modes2  # number of Fourier modes to keep in y-direction

        # scaling factor to keep initial weights small, preventing exploding activations
        self.scale = 1 / (in_channels * out_channels)

        # learnable complex weight tensor for the top-left corner of Fourier space
        # shape: [in_channels, out_channels, modes1, modes2]
        # dtype=cfloat: complex numbers needed since FFT output is complex
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

        # learnable complex weight tensor for the bottom-left corner of Fourier space
        # (2D FFT has symmetry: we need to handle both top and bottom frequency corners)
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # Einstein summation: mixes input channels with weight matrix per Fourier mode
        # input:   (batch, in_channel, x, y)
        # weights: (in_channel, out_channel, x, y)
        # output:  (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    # This is essentially equation (5) of the paper
    def forward(self, x):
        batchsize = x.shape[0]  # number of samples in the batch

        # Step 1: 2D real FFT → transform spatial signal to Fourier space
        #
        # The input is a 2D field a(x, y) defined on an 85x85 grid over [0,1]^2.
        # The 2D Fourier transform decomposes this field into 2D waves, each
        # characterized by a pair of frequencies (kx, ky):
        #
        #   kx = x-frequency: number of full wave cycles across the x-axis
        #   ky = y-frequency: number of full wave cycles across the y-axis
        #
        # Each Fourier mode (kx, ky) represents a 2D wave pattern:
        #
        #   (kx=1, ky=0)        (kx=0, ky=1)        (kx=2, ky=1)
        #   varies along x      varies along y      varies along both
        #   only                only
        #
        #   ████░░░░████░░░░    ████████████████    ████░░░░████░░░░
        #   ████░░░░████░░░░    ████████████████    ░░░░████░░░░████
        #   ████░░░░████░░░░    ░░░░░░░░░░░░░░░░    ████░░░░████░░░░
        #   ████░░░░████░░░░    ░░░░░░░░░░░░░░░░    ░░░░████░░░░████
        #
        # Why keep only low frequencies (modes=12)?
        #   The Darcy Flow solution is smooth — it has no sharp, rapidly oscillating
        #   features. Low frequencies (small kx, ky) capture large-scale variation.
        #   High frequencies capture fine-grained noise that is not physically meaningful.
        #
        #   low freq (kx=1, ky=1):    high freq (kx=40, ky=40):
        #   smooth, large features     rapid oscillations, noise
        #
        #   ~~~~~~~~~~~~~~~~~~~~       |||||||||||||||||||||||
        #   ~~~~~~~~~~~~~~~~~~~~       |||||||||||||||||||||||
        #   ~~~~~~~~~~~~~~~~~~~~       |||||||||||||||||||||||
        #
        #   Only the 12x12=144 lowest frequency modes are kept out of the full 85x85=7225
        #   Fourier grid. The remaining 7225-144=7081 high-frequency modes are set to zero.
        #
        # rfft2 exploits real-valued symmetry: output shape is [batch, channels, x, y//2+1]
        x_ft = torch.fft.rfft2(x)

        # Step 2: allocate output tensor in Fourier space, initialized to zero
        # shape: [batch, out_channels, x, y//2+1] — same shape as x_ft
        # zeros mean: all Fourier modes start at 0 (high frequencies are discarded)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Step 3: multiply the TOP-LEFT corner of Fourier space
        #
        # After rfft2, the output is a 2D grid of complex Fourier coefficients:
        #
        #          y-frequencies:  0   1   2  ...  11 | 12  ...  y//2
        #          x=0  (low)    [ ■   ■   ■  ...  ■  |  0  ...   0  ]
        #          x=1           [ ■   ■   ■  ...  ■  |  0  ...   0  ]
        #          ...
        #          x=11          [ ■   ■   ■  ...  ■  |  0  ...   0  ]  <-- TOP-LEFT (weights1)
        #          x=12          [ 0   0   0  ...  0  |  0  ...   0  ]  <-- zeroed (discarded)
        #          ...
        #          x=-12         [ 0   0   0  ...  0  |  0  ...   0  ]  <-- zeroed (discarded)
        #          x=-11         [ ■   ■   ■  ...  ■  |  0  ...   0  ]
        #          ...
        #          x=-1 (high)   [ ■   ■   ■  ...  ■  |  0  ...   0  ]  <-- BOTTOM-LEFT (weights2)
        #
        # TOP-LEFT    = low positive x-frequencies (x=0..11) x low y-frequencies (y=0..11)
        # BOTTOM-LEFT = low negative x-frequencies (x=-11..-1) x low y-frequencies (y=0..11)
        # Everything else is left as zero → high frequencies are discarded
        #
        # Why two corners?
        #   rfft2 handles y-direction symmetry automatically (returns only y=0..y//2).
        #   But the x-direction still has both positive AND negative frequencies.
        #   Positive x-frequencies: indices [:modes1]  → top of the x-axis
        #   Negative x-frequencies: indices [-modes1:] → bottom of the x-axis
        #   Both carry independent information, so both must be kept with separate weights.
        #
        # In 1D (fourier_1d.py), only one weight tensor was needed because rfft
        # handles the single frequency axis symmetry entirely.
        #
        # x_ft[:, :, :modes1, :modes2] selects top-left: x-indices 0..11, y-indices 0..11
        #
        # Equation (5) of the paper for the TOP-LEFT corner:
        #
        #   out_ft[b, l, kx, ky] = sum_{j=1}^{d_v} R1[kx, ky, l, j] * x_ft[b, j, kx, ky]
        #
        #   for kx = 0, 1, ..., k_max-1        (positive x-frequencies, top of x-axis)
        #       ky = 0, 1, ..., k_max-1        (low y-frequencies)
        #       l  = 0, 1, ..., d_v-1          (output channel index)
        #       j  = 0, 1, ..., d_v-1          (input channel index, summed over)
        #
        #   where R1 = self.weights1 is the learnable complex weight tensor for this corner
        #
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )

        # Step 4: multiply the BOTTOM-LEFT corner of Fourier space
        # x_ft[:, :, -modes1:, :modes2] selects x-indices -11..-1 (negative x-frequencies)
        # and y-indices 0..11 (same low y-frequencies as top-left)
        #
        # Equation (5) of the paper for the BOTTOM-LEFT corner:
        #
        #   out_ft[b, l, -kx, ky] = sum_{j=1}^{d_v} R2[-kx, ky, l, j] * x_ft[b, j, -kx, ky]
        #
        #   for kx = 1, 2, ..., k_max          (negative x-frequencies, bottom of x-axis)
        #       ky = 0, 1, ..., k_max-1        (low y-frequencies, same as top-left)
        #       l  = 0, 1, ..., d_v-1          (output channel index)
        #       j  = 0, 1, ..., d_v-1          (input channel index, summed over)
        #
        #   where R2 = self.weights2 is a SEPARATE learnable weight tensor for this corner
        #   R1 != R2 because positive and negative x-frequencies carry different information
        #
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Step 5: inverse 2D FFT → transform back to physical space
        # s=(x.size(-2), x.size(-1)) ensures output spatial size matches input
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        # first linear layer: mix channels from in_channels → mid_channels
        # Conv2d with kernel_size=1 is equivalent to a pointwise Linear layer on 2D data
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        # second linear layer: mix channels from mid_channels → out_channels
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)  # linear transform W_1: shape unchanged except channels
        x = F.gelu(x)  # non-linear activation: adds expressiveness
        x = self.mlp2(x)  # linear transform W_2: project to output channels
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1  # number of Fourier modes in x-direction (k_max = 12)
        self.modes2 = modes2  # number of Fourier modes in y-direction (k_max = 12)
        self.width = width  # number of hidden channels (d_v = 32)
        self.padding = (
            9  # padding size added to handle non-periodic boundary conditions
        )

        # lifting layer P: maps 3 input channels (a(x,y), x, y) → width=32 hidden channels
        self.p = nn.Linear(3, self.width)

        # 4 spectral convolution layers: each applies F^{-1}(R_phi · F(v_t)) in 2D
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # 4 MLP layers: one per Fourier layer, adds non-linearity to the spectral path
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)

        # 4 local linear bypass layers W: pointwise Conv2d with kernel_size=1
        # these are the W*v_t term in the update equation v_{t+1} = sigma(W*v_t + K(v_t))
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # projection layer Q: maps width=32 hidden channels → 1 output channel
        # mid_channels = width*4 = 128 gives more capacity for the final projection
        self.q = MLP(self.width, 1, self.width * 4)

    def forward(self, x):
        # x input shape: [batch, s, s, 1] — the diffusion coefficient a(x,y) on a 2D grid

        # Step 1: append (x, y) grid coordinates as extra input channels
        # this tells the model where each point is in the domain [0,1]^2
        grid = self.get_grid(x.shape, x.device)  # shape: [batch, s, s, 2]
        x = torch.cat(
            (x, grid), dim=-1
        )  # shape: [batch, s, s, 3] — (a, x_coord, y_coord)

        # Step 2: lift from 3 input channels to width=32 hidden channels (lifting P)
        x = self.p(x)  # shape: [batch, s, s, 32]

        # Step 3: permute to channels-first format required by Conv2d layers
        x = x.permute(0, 3, 1, 2)  # shape: [batch, 32, s, s]

        # Step 4: pad the spatial domain to handle non-periodic boundary conditions
        #
        # The Fourier transform inherently assumes the signal is PERIODIC — it treats
        # the left edge and right edge as if they are connected, and similarly for
        # top and bottom edges. This is fine for problems like Navier-Stokes on a
        # periodic torus, but Darcy Flow has NON-PERIODIC boundary conditions:
        # the solution value at the left edge has no relation to the right edge.
        #
        # Without padding, the FFT sees an artificial discontinuity at the boundary
        # where the right edge "wraps around" to the left edge:
        #
        #   actual signal:          what FFT sees (without padding):
        #   |~~~~~\                 |~~~~~\ /~~~~~|   ← sharp jump at boundary
        #    0    85                 0    85      170  ← FFT wraps around
        #
        # This jump introduces spurious high-frequency artifacts (Gibbs phenomenon)
        # that pollute the Fourier coefficients and degrade accuracy.
        #
        # Padding fixes this by appending 9 zeros to the right (x) and top (y) edges,
        # giving the signal room to taper toward zero before wrapping:
        #
        #   padded signal:          what FFT sees (with padding):
        #   |~~~~~\____             |~~~~~\____ ____\~~~~~|   ← smooth transition
        #    0    85  94             0    85  94          188
        #          ^^^
        #          9 zeros added
        #
        # The 9 zero-padded columns/rows act as a buffer that smooths the artificial
        # boundary discontinuity, making the signal approximately periodic and reducing
        # Fourier artifacts.
        #
        # F.pad(x, [0, padding, 0, padding]) pads:
        #   last  dimension (y-axis): 0 on left,       padding=9 on right
        #   second-last dim (x-axis): 0 on left/bottom, padding=9 on top/right
        #
        # After padding: shape [batch, 32, s, s] → [batch, 32, s+9, s+9]
        # The 9 extra rows/cols are removed after the Fourier layers in Step 6.
        x = F.pad(x, [0, self.padding, 0, self.padding])  # shape: [batch, 32, s+9, s+9]

        # Step 5: four Fourier layers — each applies spectral conv + MLP + bypass + activation
        x1 = self.conv0(
            x
        )  # spectral path: F^{-1}(R_phi · F(x)), global/low-freq features
        x1 = self.mlp0(x1)  # MLP: non-linear channel mixing on spectral output
        x2 = self.w0(x)  # bypass path: local pointwise linear transform W*x
        x = x1 + x2  # combine spectral and local paths
        x = F.gelu(x)  # outer non-linear activation sigma

        x1 = self.conv1(x)  # Fourier layer 2: spectral path
        x1 = self.mlp1(x1)  # MLP on spectral output
        x2 = self.w1(x)  # bypass path
        x = x1 + x2  # combine paths
        x = F.gelu(x)  # activation

        x1 = self.conv2(x)  # Fourier layer 3: spectral path
        x1 = self.mlp2(x1)  # MLP on spectral output
        x2 = self.w2(x)  # bypass path
        x = x1 + x2  # combine paths
        x = F.gelu(x)  # activation

        x1 = self.conv3(
            x
        )  # Fourier layer 4: spectral path (no activation after last layer)
        x1 = self.mlp3(x1)  # MLP on spectral output
        x2 = self.w3(x)  # bypass path
        x = x1 + x2  # combine paths — no activation here (last Fourier layer)

        # Step 6: remove the padding added in Step 4
        #
        # The padding was only needed DURING the Fourier layers to prevent boundary
        # artifacts. Now that all 4 Fourier layers have finished, it has served its
        # purpose and must be removed for three reasons:
        #
        #   1. Shape mismatch — the ground truth y has shape [batch, 85, 85].
        #      If padding is not removed, the prediction shape [batch, 94, 94]
        #      won't match y and the loss computation will fail.
        #
        #   2. Meaningless region — the 9 extra rows/cols were filled with zeros
        #      and do not correspond to any point in the physical domain [0,1]^2.
        #      Keeping them would mean predicting values outside the domain.
        #
        #   3. Padding is artificial — it was never real data, only a construct
        #      to help the FFT handle non-periodic boundaries smoothly.
        #
        # Lifecycle of padding:
        #
        #   input:           [batch, 32, 85, 85]   original 85x85 grid
        #       F.pad ↓      add 9 zeros on right and top edges
        #   padded:          [batch, 32, 94, 94]   85+9=94, larger grid for FFT
        #       Fourier layers 0-3 ↓   boundary artifacts absorbed by extra rows/cols
        #   after layers:    [batch, 32, 94, 94]   still 94x94
        #       x[...,:-9,:-9] ↓  strip off the 9 padded rows and cols
        #   output:          [batch, 32, 85, 85]   restored to original 85x85
        #
        # x[..., :-padding, :-padding] removes the last 9 elements along both
        # the x-axis (second-last dim) and y-axis (last dim)
        x = x[..., : -self.padding, : -self.padding]  # shape: [batch, 32, s, s]

        # Step 7: project from 32 hidden channels → 1 output channel (projection Q)
        x = self.q(x)  # shape: [batch, 1, s, s]

        # Step 8: permute back to channels-last format to match target shape
        x = x.permute(0, 2, 3, 1)  # shape: [batch, s, s, 1]
        return x

    def get_grid(self, shape, device):
        # creates a coordinate grid for the 2D domain [0,1]^2
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]

        # x-coordinates: evenly spaced from 0 to 1 along the x-axis
        # shape after reshape and repeat: [batch, size_x, size_y, 1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])

        # y-coordinates: evenly spaced from 0 to 1 along the y-axis
        # shape after reshape and repeat: [batch, size_x, size_y, 1]
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

        # concatenate x and y grids along the last dimension
        # output shape: [batch, size_x, size_y, 2]
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################
TRAIN_PATH = "data/piececonst_r421_N1024_smooth1.mat"  # training data file: 1024 Darcy flow samples
TEST_PATH = "data/piececonst_r421_N1024_smooth2.mat"  # test data file: separate 1024 Darcy flow samples

ntrain = 1000  # number of training samples to use
ntest = 100  # number of test samples to use

batch_size = 20  # samples per gradient update
learning_rate = 0.001  # initial learning rate for Adam optimizer
epochs = 500  # total number of passes through the training data
iterations = epochs * (ntrain // batch_size)  # total gradient steps: 500 * 50 = 25000

modes = 12  # number of Fourier modes to keep in each direction (k_max)
width = 32  # number of hidden channels (d_v) — smaller than 1D (32 vs 64) since 2D is more expensive

r = 5  # subsampling rate: take every 5th point in each spatial direction
h = int(
    ((421 - 1) / r) + 1
)  # grid size after subsampling: (421-1)/5 + 1 = 85 points per side
s = h  # s = 85: spatial resolution used throughout

################################################################
# load data and data normalization
################################################################

# load training data from .mat file
reader = MatReader(TRAIN_PATH)

# read diffusion coefficient field 'coeff' and subsample by r=5 in both x and y
# [:ntrain] takes first 1000 samples, [::r,::r] subsamples, [:,:s,:s] clips to 85x85
x_train = reader.read_field("coeff")[:ntrain, ::r, ::r][
    :, :s, :s
]  # shape: [1000, 85, 85]

# read solution field 'sol' with same subsampling
y_train = reader.read_field("sol")[:ntrain, ::r, ::r][
    :, :s, :s
]  # shape: [1000, 85, 85]

# reload reader with test data file and read test samples
reader.load_file(TEST_PATH)
x_test = reader.read_field("coeff")[:ntest, ::r, ::r][:, :s, :s]  # shape: [100, 85, 85]
y_test = reader.read_field("sol")[:ntest, ::r, ::r][:, :s, :s]  # shape: [100, 85, 85]

# ── INPUT NORMALIZATION ───────────────────────────────────────────────────────
# The diffusion coefficient a(x,y) can vary at very different scales across
# samples. Without normalization, large input values cause large activations
# inside the network, making training unstable and slow.
#
# UnitGaussianNormalizer transforms inputs to zero mean and unit variance:
#
#   a_hat(x,y) = (a(x,y) - mu) / (sigma + eps)
#
# where mu and std are computed from training data only.
#
# IMPORTANT: the normalizer is fitted on x_train and the SAME mu/sigma are
# applied to x_test. Refitting on x_test would be data leakage — the test
# set must be treated as completely unseen, including using training statistics:
#
#   CORRECT:                        WRONG:
#   fit normalizer on x_train       fit separate normalizer on x_test
#   apply to x_train   ✓            apply to x_test   ✗ (data leakage)
#   apply to x_test    ✓
#
x_normalizer = UnitGaussianNormalizer(x_train)  # fit: compute mu, sigma from x_train
x_train = x_normalizer.encode(x_train)          # transform x_train → zero mean, unit variance
x_test = x_normalizer.encode(x_test)            # apply same mu, sigma to x_test (no refit)

# ── OUTPUT NORMALIZATION ──────────────────────────────────────────────────────
# The solution u(x,y) can also have large values. If the loss is computed in
# physical units with large u, the denominator ||u||_2 in the relative L2 loss:
#
#   L = ||u_pred - u||_2 / ||u||_2
#
# dominates and makes the loss insensitive to actual prediction errors,
# causing unstable training.
#
# By normalizing y, the model learns in a well-scaled space. However, the loss
# must be computed in PHYSICAL units (not normalized units) so that it is
# meaningful. This is why y_normalizer.decode() is called during training:
#
#   y_train (physical) → encode → normalized y → DataLoader
#                                                     ↓
#                                               model trains on normalized y
#                                                     ↓
#                              model output (normalized) → decode → physical
#                                                                       ↓
#                                                            loss vs decoded y
#
# NOTE: y_test is NEVER encoded — it stays in physical units throughout.
# During evaluation, only the model output is decoded before comparing to y_test:
#
#   out = y_normalizer.decode(out)   # prediction → physical units
#   loss = myloss(out, y_test)       # y_test already in physical units, no decode needed
#
y_normalizer = UnitGaussianNormalizer(y_train)  # fit: compute mu, sigma from y_train
y_train = y_normalizer.encode(y_train)          # transform y_train → zero mean, unit variance
                                                # y_test is intentionally NOT encoded

# add a channel dimension to inputs: model expects (batch, x, y, channels)
x_train = x_train.reshape(ntrain, s, s, 1)  # shape: [1000, 85, 85, 1]
x_test = x_test.reshape(ntest, s, s, 1)  # shape: [100,  85, 85, 1]

# wrap data in DataLoader for batching
# shuffle=True: randomize sample order each epoch to prevent overfitting to ordering
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)
# shuffle=False: fixed order for reproducible test evaluation
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
)

################################################################
# training and evaluation
################################################################

# instantiate 2D FNO with modes=12 in both directions, width=32, move to GPU
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))  # print total number of learnable parameters

# Adam optimizer with L2 regularization (weight_decay) to prevent overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# cosine annealing: smoothly decay learning rate from 0.001 → 0 over 25000 steps
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

# relative L2 loss: ||pred - truth||_2 / ||truth||_2, summed over batch
myloss = LpLoss(size_average=False)

# move y_normalizer's mean and std tensors to GPU so decode() works on GPU tensors
y_normalizer.cuda()

for ep in range(epochs):  # loop over all epochs
    model.train()  # set to training mode
    t1 = default_timer()  # record epoch start time
    train_l2 = 0  # accumulate training loss over all batches

    for x, y in train_loader:  # iterate over mini-batches of 20 samples
        x, y = x.cuda(), y.cuda()  # move batch to GPU

        optimizer.zero_grad()  # clear gradients from previous step

        # forward pass: predict solution on 2D grid, reshape to [batch, s, s]
        out = model(x).reshape(batch_size, s, s)

        # decode prediction from normalized space back to physical units
        out = y_normalizer.decode(out)

        # decode target from normalized space back to physical units
        # must decode y too so loss is computed in the same physical space as out
        y = y_normalizer.decode(y)

        # compute relative L2 loss between prediction and ground truth
        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()  # backpropagate gradients

        optimizer.step()  # update model weights
        scheduler.step()  # decay learning rate by one step
        train_l2 += loss.item()  # accumulate batch loss

    model.eval()  # set to evaluation mode
    test_l2 = 0.0
    with torch.no_grad():  # disable gradient tracking for evaluation
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            # forward pass on test batch
            out = model(x).reshape(batch_size, s, s)

            # decode prediction back to physical units
            # note: y is NOT decoded here since we only need to compare out vs y in
            # the same space — myloss handles this correctly with decoded out
            out = y_normalizer.decode(out)

            # accumulate test loss (y stays encoded, out is decoded — both in same space
            # because myloss computes relative error)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain  # average training L2 per sample
    test_l2 /= ntest  # average test L2 per sample

    t2 = default_timer()
    # print: epoch | time(s) | train L2 | test L2
    print(ep, t2 - t1, train_l2, test_l2)
