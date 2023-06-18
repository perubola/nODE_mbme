# Most of this code is from the following repository:
# Will be modifying as needed for our purposes
# https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py
# ==============================================================================
"""Train a neural stochastic differential equation (nSDE) model."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.swa_utils as swa_utils

import numpy as np
import matplotlib.pyplot as plt

import torchcde
import torchsde
import tqdm

###################
# First some standard helper objects.
###################

class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [nn.Linear(in_size, mlp_size),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(nn.Linear(mlp_size, mlp_size))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(nn.Tanh())
        self._model = nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

###################
# SDE Stuff
#
# First is generator SDE
###################
class GeneratorFunc(nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'diagonal'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        self._drift = MLP(1+hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)

###################
# Wrap it to compute the SDE
###################
class Generator(nn.Module):
    def __init__(self,
                 data_size,
                 initial_noise_size,
                 noise_size,
                 hidden_size,
                 mlp_size,
                 num_layers,
                ):
         super().__init__()
         self._initial_noise_size = initial_noise_size
         self._hidden_size = hidden_size
         
         self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
         self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
         self._readout = nn.Linear(hidden_size, data_size)

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and is where we want to evaluate SDE at
        # 
        # solve the SDE
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)

        # use reverse Heun for good gradients with adjoint method
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0,
                                     adjoint_method='adjoint_reversible_heun',)
        xs = xs.transpose(0,1)
        ys = self._readout(xs)

        # Normalize to data that discriminator expects, in this case time is a channel
        ts = ts.unsqueeze(0).unsqueeve(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2)) 
    
# Now CDE stuff
class DiscriminatorFunc(nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)

# this is honestly a classifier, but we call it a discriminator to be consistent with the literature  
class Discriminator(nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = nn.Linear(hidden_size, 1)

    def forward(self, ys_coeffs):
        # ys_coeffs has shape (batch_size, data_size, 1 + data_size)
        # the +1 lets us handle irregular time as a channel 
        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', backend='torchsde',
                             dt=1.0, adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()

# Right now this is for toy data, will replace with real data later 
def get_data(batch_size, device):
    dataset_size = 100
    t_size = 100

    ts = torch.linspace(0, 1, 100).to(device)
    ys = torch.sin(2 * np.pi * ts).to(device)
    ###################
    # As discussed, time must be included as a channel for the discriminator.
    ###################
    ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1),
                    ys.transpose(0, 1)], dim=2)
    
    data_size = ys.size(-1) - 1 # -1 for time channel
    ys_coeffs = torchcde.linear_interpolation_coeffs(ys)
    dataset = torch.utils.data.TensorDataset(ys_coeffs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return ts, data_size, dataloader

################### 
# GAN Training
#
# Have not finished looking through repo example, will update as needed 
###################


def get_loss(ts, batch_size, dataloader, generator, discriminator):
    with torch.inference_mode():  # same thing as torch.no_grad()
        total_samples = 0
        total_loss = 0
        for real_samples in dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples

def main(
        # architectural hyperparameters
        initial_noise_size=1,  # noise dimension at start of sde
        noise_size=3,          # noise dimensions of the brownian motion
        hidden_size=32,        # hidden size of the generator and discriminator
        mlp_size=32,           # size of the hidden layers of the mlps
        num_layers=2,          # number of hidden layers of the mlps

        # training hyperparameters
        generator_lr=1e-3,     # learning rate of the generator
        discriminator_lr=1e-3, # learning rate of the discriminator
        batch_size=100,        # batch size
        steps=10,              # number of steps to train for
        swa_step_start=5,      # step to start swa
        weight_decay=0.01,     # weight decay
):
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

    # Data
    ts, data_size, dataloader = get_data(batch_size=batch_size, device=device)

    # Models
    generator = Generator(data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers).to(device)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers).to(device)
    # weight averaging helps gan training apparently
    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    # need to initialize right

    # optimisers. Adadelta is used in the paper
    generator_optimiser = torch.optim.Adadelta(generator.parameters(), lr=generator_lr, weight_decay=weight_decay)

    # training loop
    trange = tqdm.tqdm(range(steps))