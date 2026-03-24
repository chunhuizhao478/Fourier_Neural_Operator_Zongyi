"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""

import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

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
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
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
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic
                         # NOTE: padding is defined here but commented out in forward()
                         # because the Navier-Stokes spatial domain (x, y) is periodic
                         # (flow wraps around on a torus), so no padding is needed.

        self.p = nn.Linear(12, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        # InstanceNorm2d: normalizes each sample and each channel independently
        # across the spatial (x, y) dimensions only.
        #
        # Why needed here but NOT in fourier_2d.py?
        #   This model is applied RECURRENTLY — the output of one step becomes the
        #   input of the next step, repeated 40 times. Without normalization, small
        #   errors in the prediction can compound and grow exponentially over time
        #   (error accumulation / exploding activations).
        #
        # Why InstanceNorm and not BatchNorm?
        #   BatchNorm normalizes across the batch dimension — its statistics depend
        #   on other samples in the batch, which causes inconsistency when the model
        #   is applied step-by-step (each step processes a different temporal slice).
        #   InstanceNorm normalizes each sample independently, so it is stable
        #   regardless of batch composition or which timestep is being processed.
        #
        # It is applied BEFORE and AFTER each spectral conv (see forward()):
        #   self.norm(self.conv0(self.norm(x)))
        #   ├── inner self.norm(x):    normalize input  before FFT
        #   └── outer self.norm(...):  normalize output after  inverse FFT
        #   This double normalization keeps activations in a controlled range
        #   at every layer, preventing blow-up during recurrent rollout.
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        #
        # WHY PADDING IS NOT APPLIED HERE (neither x, y, nor t):
        #
        # Padding is only needed for dimensions that go through the FFT AND are
        # not periodic. The FFT assumes the signal wraps around at the boundary.
        # If the signal is not actually periodic, you get boundary artifacts
        # (Gibbs phenomenon). Padding adds zeros to prevent that wrap-around.
        #
        # In this file, rfft2 only transforms x and y:
        #   - x and y: FFT'd, but the Navier-Stokes domain IS periodic → no pad needed
        #   - t:        NOT FFT'd (time is handled by the autoregressive loop,
        #               stacked as input channels) → FFT never sees t,
        #               so it cannot make a wrong periodicity assumption → no pad needed
        #
        # Compare to fourier_3d.py which DOES pad:
        #   rfftn transforms x, y, AND t. Since t is not periodic, 6 zeros are
        #   padded along the time axis to avoid wrap-around artifacts.
        #
        # Simple rule: only pad a dimension if it goes through the FFT AND is not periodic.
        #   +-----+----------------------+------------------+-----------+
        #   | Dim | fourier_2d_time.py   | fourier_3d.py    | Pad here? |
        #   +-----+----------------------+------------------+-----------+
        #   |  x  | FFT'd, periodic      | FFT'd, periodic  |    No     |
        #   |  y  | FFT'd, periodic      | FFT'd, periodic  |    No     |
        #   |  t  | NOT FFT'd            | FFT'd, not       |  No / Yes |
        #   |     | (autoregressive loop)| periodic         |           |
        #   +-----+----------------------+------------------+-----------+

        # Fourier layer 1 — with InstanceNorm wrapping the spectral conv:
        #   self.norm(x)        → normalize input before FFT (stabilize spectral path entry)
        #   self.conv0(...)     → spectral conv: F^{-1}(R_phi · F(x))
        #   self.norm(...)      → normalize output after inverse FFT (stabilize spectral path exit)
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)     # MLP: non-linear channel mixing on spectral output
        x2 = self.w0(x)        # bypass path: local linear W*x (no norm on bypass)
        x = x1 + x2            # combine spectral and local paths
        x = F.gelu(x)          # outer non-linear activation

        x1 = self.norm(self.conv1(self.norm(x)))  # Fourier layer 2 with InstanceNorm
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))  # Fourier layer 3 with InstanceNorm
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))  # Fourier layer 4 with InstanceNorm
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################

TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'

ntrain = 1000
ntest = 200

modes = 12
width = 20

batch_size = 20
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)

path = 'ns_fourier_2d_time_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

sub = 1
S = 64
T_in = 10
T = 40 # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0   # cumulative loss summed over every individual timestep prediction
    train_l2_full = 0   # cumulative loss over the full predicted trajectory per sample
    for xx, yy in train_loader:
        loss = 0                    # reset loss accumulator for this batch
        xx = xx.to(device)          # xx: input window, shape [batch, 64, 64, 10]
                                    #     contains the 10 most recent timesteps
        yy = yy.to(device)          # yy: full target trajectory, shape [batch, 64, 64, 40]
                                    #     contains ground truth for all 40 future timesteps

        # ── AUTOREGRESSIVE / SLIDING WINDOW LOOP ─────────────────────────────
        # Instead of predicting all 40 timesteps at once, the model predicts
        # ONE timestep at a time and uses its own prediction as input for the next.
        #
        # Sliding window illustration (T_in=10, step=1):
        #
        #   t=0:  input=[t-10..t-1]  → predict t=0  → window slides: drop t-10, append t=0
        #   t=1:  input=[t-9 ..t=0]  → predict t=1  → window slides: drop t-9,  append t=1
        #   ...
        #   t=39: input=[t=29..t=38] → predict t=39 → done
        #
        for t in range(0, T, step):
            y = yy[..., t:t + step]     # ground truth for this timestep: shape [batch, 64, 64, 1]

            im = model(xx)              # predict next timestep from current window
                                        # im shape: [batch, 64, 64, 1]

            # accumulate per-step loss: sum of relative L2 error at each timestep
            # this is what gets backpropagated — the model is trained to be accurate
            # at EVERY individual step, not just the final one
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            # collect predictions into full trajectory tensor for full-trajectory loss
            if t == 0:
                pred = im                           # first step: initialize pred
            else:
                pred = torch.cat((pred, im), -1)    # subsequent steps: append along time axis

            # ── KEY: slide the input window forward by one step ───────────────
            # xx[..., step:] drops the oldest timestep (leftmost along time axis)
            # im             appends the new prediction as the most recent timestep
            # result: window always contains the 10 most recent timesteps
            #
            # Before: xx = [u(t-10), u(t-9), ..., u(t-1)]   shape: [batch, 64, 64, 10]
            # After:  xx = [u(t-9),  u(t-8), ..., u(t-0)]   shape: [batch, 64, 64, 10]
            #                                        ↑ this is im (model's own prediction)
            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()    # sum of per-step losses for this batch

        # full trajectory loss: compare entire predicted sequence vs ground truth
        # this gives a holistic measure of trajectory quality beyond per-step accuracy
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()   # clear gradients
        loss.backward()         # backpropagate through ALL 40 timestep predictions
        optimizer.step()        # update weights
        scheduler.step()        # decay learning rate

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():       # no gradients needed during evaluation
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            # same autoregressive loop as training but weights are frozen
            for t in range(0, T, step):
                y = yy[..., t:t + step]     # ground truth at this timestep
                im = model(xx)              # predict next timestep
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)   # slide window forward

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    # print: epoch | time | train_l2_step/sample | train_l2_full/sample
    #             | test_l2_step/sample | test_l2_full/sample
    # train_l2_step / (T/step): average per-step loss (normalized by number of steps)
    # train_l2_full:             average full-trajectory loss per sample
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)
# torch.save(model, path_model)

