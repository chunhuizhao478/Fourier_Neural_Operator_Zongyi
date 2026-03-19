"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

from timeit import default_timer

import torch.nn.functional as F

from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width)  # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 2)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


################################################################
#  configurations
################################################################
ntrain = 1000  # number of training samples
ntest = 100  # number of test samples

sub = (
    2**3
)  # subsampling rate: take every 8th point from the raw grid !subsampling works better for smooth equations
h = 2**13 // sub  # grid size after subsampling: 8192 / 8 = 1024 points
s = h  # s is the spatial resolution used throughout (1024)

batch_size = 20  # number of samples processed together in each gradient update
learning_rate = 0.001  # initial learning rate for Adam optimizer
epochs = 500  # total number of full passes through the training data
iterations = epochs * (ntrain // batch_size)  # total number of gradient update steps:
# 500 epochs * (1000/20) = 25000 steps

modes = 16  # number of Fourier modes to keep (k_max): truncate at 16 lowest frequencies
width = 64  # number of channels (d_v) in the hidden representation

################################################################
# read data
################################################################

# Load the Burgers equation dataset from a .mat file.
# The file contains two fields:
#   'a': initial conditions u(x, 0),  shape [N, 8192]
#   'u': solutions at time 1, u(x, 1), shape [N, 8192]
dataloader = MatReader("data/burgers_data_R10.mat")

# Read initial conditions and subsample: take every 8th point → shape [N, 1024]
x_data = dataloader.read_field("a")[:, ::sub]

# Read solutions and subsample: take every 8th point → shape [N, 1024]
y_data = dataloader.read_field("u")[:, ::sub]

# Split into train and test sets
x_train = x_data[:ntrain, :]  # first 1000 samples → shape [1000, 1024]
y_train = y_data[:ntrain, :]  # first 1000 solutions → shape [1000, 1024]
x_test = x_data[-ntest:, :]  # last 100 samples → shape [100, 1024]
y_test = y_data[-ntest:, :]  # last 100 solutions → shape [100, 1024]

# Add a channel dimension to x (input has 1 channel: the initial condition value)
# shape: [1000, 1024] → [1000, 1024, 1]
# y stays flat [1000, 1024] since it is the regression target
x_train = x_train.reshape(ntrain, s, 1)
x_test = x_test.reshape(ntest, s, 1)

# Wrap data in DataLoader for batching during training
# shuffle=True: randomize sample order each epoch to reduce overfitting
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)
# shuffle=False: keep fixed order for evaluation so results are reproducible
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
)

# Instantiate the FNO model with 16 Fourier modes and 64 hidden channels,
# and move it to GPU memory
model = FNO1d(modes, width).cuda()
print(count_params(model))  # print total number of learnable parameters

################################################################
# training and evaluation
################################################################

# Adam optimizer: adapts the learning rate per parameter using gradient history.
# weight_decay=1e-4 adds L2 regularization to prevent overfitting.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Cosine annealing: smoothly decays the learning rate from learning_rate → 0
# over T_max=25000 gradient steps (the full training run)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

# LpLoss: relative L2 loss ||pred - truth||_2 / ||truth||_2, summed over the batch
myloss = LpLoss(size_average=False)

for ep in range(epochs):  # loop over epochs
    model.train()  # set model to training mode (enables dropout/batchnorm if any)
    t1 = default_timer()  # record epoch start time
    train_mse = 0  # accumulate MSE loss over all training batches
    train_l2 = 0  # accumulate relative L2 loss over all training batches

    for x, y in train_loader:  # iterate over mini-batches of size 20
        x, y = x.cuda(), y.cuda()  # move batch to GPU

        optimizer.zero_grad()  # clear gradients from previous step
        out = model(x)  # forward pass: predict u(x,1) from u(x,0)
        # out shape: [20, 1024, 1]

        # MSE loss: average squared difference per grid point (for monitoring only)
        mse = F.mse_loss(
            out.view(batch_size, -1), y.view(batch_size, -1), reduction="mean"
        )

        # Relative L2 loss: used as the actual training objective
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))

        l2.backward()  # backpropagate gradients through the network

        optimizer.step()  # update all model parameters using computed gradients
        scheduler.step()  # decay the learning rate by one cosine annealing step
        train_mse += mse.item()  # accumulate batch MSE (scalar)
        train_l2 += l2.item()  # accumulate batch L2 loss (scalar)

    model.eval()  # set model to evaluation mode (disables dropout/batchnorm)
    test_l2 = 0.0
    with torch.no_grad():  # disable gradient computation — saves memory and speeds up
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)  # forward pass on test batch (no gradient tracking)
            # accumulate relative L2 loss across all test batches
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)  # average MSE per batch
    train_l2 /= ntrain  # average relative L2 per training sample
    test_l2 /= ntest  # average relative L2 per test sample

    t2 = default_timer()
    # print: epoch | time(s) | train MSE | train L2 | test L2
    print(ep, t2 - t1, train_mse, train_l2, test_l2)

################################################################
# per-sample evaluation after training
################################################################

# torch.save(model, 'model/ns_fourier_burgers')  # optionally save the trained model

# Allocate a tensor to store predictions for all 100 test samples
pred = torch.zeros(y_test.shape)  # shape: [100, 1024]
index = 0  # counter to track which test sample we are on

# Reload test set with batch_size=1 to evaluate one sample at a time
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False
)

with torch.no_grad():  # no gradients needed for evaluation
    for x, y in test_loader:
        test_l2 = 0  # reset per-sample loss
        x, y = x.cuda(), y.cuda()

        out = model(x).view(-1)  # predict and flatten to 1D: shape [1024]
        pred[index] = out  # store prediction for this sample

        # compute relative L2 error for this single sample
        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)  # print: sample index | per-sample L2 error
        index = index + 1  # move to next test sample

# scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
