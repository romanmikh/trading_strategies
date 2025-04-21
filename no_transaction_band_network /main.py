#!/usr/bin/env python
# coding: utf-8

# # No-Transaction Band Network

# ## Imports & Config

# In[2]:


get_ipython().run_cell_magic('capture', '', '!curl --silent https://raw.githubusercontent.com/pfnet-research/NoTransactionBandNetwork/main/utils.py > utils.py\n!pip install matplotlib\n!pip install numpy\n!pip install seaborn\n!pip install torch\n!pip install tqdm\n!pip install random\n')


# In[3]:


import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import random
import torch
import torch.nn.functional as fn
from torch.optim import Adam
from tqdm import tqdm

from utils import MultiLayerPerceptron
from utils import clamp
from utils import entropic_loss
from utils import european_option_delta
from utils import generate_geometric_brownian_motion
from utils import to_premium


# In[4]:


seaborn.set_style("whitegrid")

FONTSIZE = 18
matplotlib.rcParams["figure.figsize"] = (10, 5)
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["figure.titlesize"] = FONTSIZE
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["legend.fontsize"] = FONTSIZE
matplotlib.rcParams["xtick.labelsize"] = FONTSIZE
matplotlib.rcParams["ytick.labelsize"] = FONTSIZE
matplotlib.rcParams["axes.labelsize"] = FONTSIZE
matplotlib.rcParams["axes.titlesize"] = FONTSIZE
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.1
matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.rcParams["axes.linewidth"] = 1.6


# In[13]:


if not torch.cuda.is_available():
    raise RuntimeWarning(
        "CUDA is not available. "
        "If you're using Google Colab, you can enable GPUs as: "
        "https://colab.research.google.com/notebooks/gpu.ipynb"
    )

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print("Default device:", DEVICE)
print(torch.cuda.memory_summary())


# ## Define derivatives

# We prepare a European option and a lookback option.
# 
# European option is the most popular option.
# 
# Lookback option is an exotic option whose payoff depends on the price history.

# In[6]:


def european_option_payoff(prices: torch.Tensor, strike=1.0) -> torch.Tensor:
    """
    Return the payoff of a European option.

    Parameters
    ----------
    prices : torch.Tensor, shape (n_steps, n_paths)
        Prices of the underlying asset.

    Returns
    -------
    payoff : torch.Tensor, shape (n_paths, )
    """
    return fn.relu(prices[-1, :] - strike)


# In[7]:


def lookback_option_payoff(prices: torch.Tensor, strike=1.03) -> torch.Tensor:
    """
    Return the payoff of a lookback option.

    Parameters
    ----------
    prices : torch.Tensor, shape (n_steps, n_paths)
        Prices of the underlying asset.

    Returns
    -------
    payoff : torch.Tensor, shape (n_paths, )
    """
    return fn.relu(torch.max(prices, dim=0).values - strike)


# ## Experiment: European Option

# ### Compute profit and loss with hedging
# 
# Bank sells option to client, incurring risk to hedge. Hedging can be managed dynamically by our hedging model.
# 
# The resulting profit and loss is obtained by adding up the payoff to the customer, capital gains from the underlying asset, and the transaction cost.

# In[14]:


def to_numpy(tensor: torch.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()

# N_PATHS are generated in each epoch
N_EPOCHS= 200
N_PATHS = 50000


# In[15]:


def compute_profit_and_loss(
    hedging_model: torch.nn.Module,
    payoff: typing.Callable[[torch.Tensor], torch.Tensor], # takes function with input torch.Tensor & output torch.Tensor
    cost: float, # transaction cost
    n_paths=N_PATHS,
    maturity=30/365,
    dt=1/365,
    volatility=0.2
) -> torch.Tensor:
    """
    Traces Pnl during whole lifetime of option
    ---
    Returns PnL of one option at expiry (pnl = gamma PL - hedging costs - final payoff)
    """
    prices = generate_geometric_brownian_motion(
        n_paths, maturity=maturity, dt=dt, volatility=volatility, device=DEVICE) # torch.Tensor
    # print("Prices shape:", prices.shape)  # torch.Size([29, 50000])
    
    # Visualise the Brownian paths
    # if hedging_model == model_ntb:
        # plt.figure(figsize=(12, 6))
        # for i in range(min(50, n_paths)):  # Plot only up to 50 paths
        #     plt.plot(prices[:, i].cpu().numpy(), alpha=0.6)  # Convert to numpy if on GPU
    
        # plt.title("First 50 Simulated Price Paths")
        # plt.xlabel("Days")
        # plt.ylabel("Price")
        # plt.show()

    hedge = torch.zeros_like(prices[:1]).reshape(-1) # torch.Size([1, 50000]) -> torch.Size([50000])
    pnl = 0

    for n in range(prices.shape[0] - 1):
        x_log_moneyness = prices[n, :, None].log()  # None converts torch.Size([50000]) to torch.Size([50000, 1])
        x_time_expiry = torch.full_like(x_log_moneyness, maturity - n * dt)  # torch.Size([50000, 1])
        x_volatility = torch.full_like(x_log_moneyness, volatility)  # torch.Size([50000, 1])
        x = torch.cat([x_log_moneyness, x_time_expiry, x_volatility], 1)  # torch.Size([50000, 3]) = ([50k* (spot, t, vol)])
        
        prev_hedge = hedge
        hedge = hedging_model(x, prev_hedge) # clamp

        # pnl from position
        pnl += hedge * (prices[n+1] - prices[n]) # reads: 50 shares held went up 3% in price
        # pay transation costs
        pnl -= cost * torch.abs(hedge - prev_hedge) * prices[n] # reads: 2% on 20 shares at $100 each

    pnl -= payoff(prices)
    
    return pnl


# ### Create hedging models
# 
# Now let us create `hedging_model` as `torch.nn.Module`.
# 
# We employ two models here:
# * **No-Transaction Band Network** (proposed architecture):
#     - A multi-layer perceptron outputs a no-transaction band, and the next hedge ratio is obtained by clamping the current hedge ratio into this band.
#     - Two outputs of the multi-layer perceptron are applied with [`LeakyReLU`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU), and then added/subtracted to the Black–Scholes’ delta to get the upper/lower-bound of the no-transaction band, respectively.
# * **Feed-forward network** (baseline):
#     - A multi-layer perception uses the current hedge ratio as an input to compute the next hedge ratio.
#     - The output of a multi-layer perceptron is applied with [`tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh) function and then added to Black–Scholes’ delta to get the next hedge ratio.

# In[16]:


class NoTransactionBandNet(torch.nn.Module):
    """
    - Feed-forward network & clamp
    - Called in compute_profit_and_loss()

    Inputs:
    in_features=x: (price, time, vol) by default
    
    x = torch.Size([50000, 3]) = ([50k* (spot, t, vol)])
      = torch.tensor([
    ...     [s0, t0, v0],
    ...     [s1, t1, v1],
    ...     [s2, t2, v2],
    ...                 )
    """

    def __init__(self, in_features=3):
        super().__init__()

        # defaults to a 4-layer MLP with 32 neurons in each hidden layer
        # [3, 32, 32, 32, 32, 2]
        self.mlp = MultiLayerPerceptron(in_features, 2)

    def forward(self, x, prev):  # define but don't call anywhere else, torch expects the forward method
        # find bounds using feed forward network
        band_width = self.mlp(x)

        # apply bounds to cost-free optimal hedge
        no_cost_delta = european_option_delta(x[:, 0], x[:, 1], x[:, 2])
        lower = no_cost_delta - fn.leaky_relu(band_width[:, 0])  # ReLU or leaky ReLU make sharper dist
        upper = no_cost_delta + fn.leaky_relu(band_width[:, 1])  # bc -ve band widths mostly removed
        # fn.leaky_relu(band_width[:, 0], ***negative_slope=1***)  =  band_width[:, 0]

        # keep our hedge within bounds 
        hedge = clamp(prev, lower, upper)
        
        return hedge


# In[17]:


class FeedForwardNet(torch.nn.Module):
    """
    Feed-forward network with Black-Scholes delta.

    Parameters
    ----------
    - in_features : int, default 3
        Number of input features.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> m = FeedForwardNet(3)
    >>> x = torch.tensor([
    ...     [-0.01, 0.1, 0.2],
    ...     [ 0.00, 0.1, 0.2],
    ...     [ 0.01, 0.1, 0.2]])
    >>> prev = torch.full_like(x[:, 0], 0.5)
    >>> m(x, prev)
    tensor([..., ..., ...], grad_fn=<AddBackward0>)
    """

    def __init__(self, in_features=3):
        super().__init__()

        # A four-layer MLP with 32 hidden neurons in each layer
        self.mlp = MultiLayerPerceptron(in_features + 1, 1)

    def forward(self, x, prev):
        # Black-Scholes' delta in the absence of transaction cost
        no_cost_delta = european_option_delta(x[:, 0], x[:, 1], x[:, 2])

        # Multi-layer perceptron directly computes the hedge ratio at the next time step
        x = torch.cat((x, prev.reshape(-1, 1)), 1)
        x = self.mlp(x).reshape(-1)
        x = torch.tanh(x)
        hedge = no_cost_delta + x

        return hedge


# ### Compute profit and loss before training

# In[18]:


from IPython.display import display, HTML
display(HTML("<style>.output_wrapper, .output {height:auto !important; max-height: none !important;}</style>"))



# transfer calculations to GPU. initialisation only, no calcualtion done here
torch.manual_seed(1337)
model_ntb = NoTransactionBandNet().to(DEVICE)
torch.manual_seed(1337)
model_ffn = FeedForwardNet().to(DEVICE)


# Calculate PL for different costs
results = []  # Store PnL data for both models and costs
for cost in [2e-2, 1e-2, 5e-3, 1e-3]:
    torch.manual_seed(1337)
    pnl_ntb = compute_profit_and_loss(model_ntb, european_option_payoff, cost=cost)
    torch.manual_seed(1337)
    pnl_ffn = compute_profit_and_loss(model_ffn, european_option_payoff, cost=cost)
    results.append((cost, pnl_ntb, pnl_ffn))

# Plot the data side by side
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns for 4 costs
fig.suptitle("PnLs of 50000 price paths for a European call (before fit)\n", fontsize=20)

for i, (cost, pnl_ntb, pnl_ffn) in enumerate(results):
    row = i // 2  # Determines the row index (0 or 1)
    col = i % 2   # Determines the column index (0 or 1)
    
    axs[row, col].hist(
        to_numpy(pnl_ntb),
        bins=100,
        range=(-0.08, -0.01),
        alpha=0.6,
        label="No-transaction Band Network",
    )
    axs[row, col].hist(
        to_numpy(pnl_ffn),
        bins=100,
        range=(-0.08, -0.01),
        alpha=0.6,
        label="Feed-forward Network",
    )
    axs[row, col].set_title(f"PnLs for Cost: {cost*100}%")
    axs[row, col].set_xlabel("Profit-loss")
    axs[row, col].set_ylabel("Number of events")
    axs[row, col].legend()
    axs[row, col].legend(fontsize='small')

plt.tight_layout()
plt.show()


# ### Fit hedging models
# 
# The profit and loss distributions with hedging are shown in the histograms above.
# 
# These distributions are not optimal since `hedging_model`s are not yet trained.
# 
# We train hedging models so that they minimize the `entropic_loss`, or equivalently, maximize the expected utility.

# In[ ]:


def fit(
    hedging_model: torch.nn.Module,
    payoff: typing.Callable[[torch.Tensor], torch.Tensor],
    cost: float,
    n_epochs=N_EPOCHS
) -> list:
    """
    Backpropagation

    Return:
    - list of errors  (floats) from loss function history at each epoch
    """

    optim = Adam(hedging_model.parameters())

    loss_history = []
    progress = tqdm(range(n_epochs))

    for _ in progress:
        optim.zero_grad() # clears previous gradients
        pnl = compute_profit_and_loss(hedging_model, payoff, cost=cost)
        loss = entropic_loss(pnl) # out utility function of preference, 
        loss.backward() # calculate gradients (backpropagation eq1 & eq2)
        optim.step()  # updates params based on gradients (backpropagation eq3 & eq4)

        progress.desc = f"cost: {cost*100}%, loss={loss:.5f}"
        loss_history.append(loss.item())

    return loss_history

def reset_weights(model):
    for layer in model.modules():  # Use .modules() to include all sub-layers
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            torch.cuda.empty_cache()
            # print("--- model parameters reset ---")


# In[ ]:


# Calculate and store loss histories for different costs
results = []  # Store loss history data for both models and costs
for cost in [1e-3]:# [1e-3, 2e-2, 1e-2, 5e-3, ]:    
    # Reset model weights to ensure same starting point
    reset_weights(model_ntb)
    reset_weights(model_ffn)

    torch.manual_seed(1337)  # Set a fixed seed for repeatability
    history_ntb = fit(model_ntb, european_option_payoff, cost=cost)
    torch.manual_seed(1337)  # Reset seed before training the second model
    
    history_ffn = fit(model_ffn, european_option_payoff, cost=cost)
    results.append((cost, history_ntb, history_ffn))


# In[ ]:


# Plot the data side by side in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns for 4 costs
fig.suptitle("Loss histories for a European option", fontsize=20)

for i, (cost, history_ntb, history_ffn) in enumerate(results):
    row = i // 2  # Determines the row index (0 or 1)
    col = i % 2   # Determines the column index (0 or 1)
    
    axs[row, col].plot(history_ntb, label="No-transaction Band Network")
    axs[row, col].plot(history_ffn, label="Feed-forward Network")
    axs[row, col].set_title(f"Cost: {cost*100}%")
    axs[row, col].set_xlabel("Number of epochs")
    axs[row, col].set_ylabel("Loss (Negative of expected utility)")
    axs[row, col].legend(fontsize='small')

plt.tight_layout()
plt.show()


# The learning histories above demonstrate that the no-transaction band network can be trained much quicker than the ordinary feed-forward network.
# 
# The fluctuations observed after around 100th epoch are mostly due to variances of Monte Carlo paths of the asset prices.

# ### Compute the profit-loss distributions with hedging

# In[ ]:


# Calculate PL for different costs
results = []  # Store PnL data for both models and costs
for cost in [1e-3]: # 1e-3, 2e-2, 1e-2, 5e-3, 
    torch.manual_seed(1337)
    pnl_ntb = compute_profit_and_loss(model_ntb, european_option_payoff, cost=cost)
    torch.manual_seed(1337)
    pnl_ffn = compute_profit_and_loss(model_ffn, european_option_payoff, cost=cost)
    results.append((cost, pnl_ntb, pnl_ffn))


# In[ ]:


# Plot the data side by side
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns for 4 costs
fig.suptitle("PnLs of 50000 price paths for a European call (after fit)\n", fontsize=20)

for i, (cost, pnl_ntb, pnl_ffn) in enumerate(results):
    row = i // 2  # Determines the row index (0 or 1)
    col = i % 2   # Determines the column index (0 or 1)
    
    axs[row, col].hist(
        to_numpy(pnl_ntb),
        bins=100,
        range=(-0.04, -0.0),
        alpha=0.6,
        label="No-transaction Band Network",
    )
    axs[row, col].hist(
        to_numpy(pnl_ffn),
        bins=100,
        range=(-0.04, -0.0),
        alpha=0.6,
        label="Feed-forward Network",
    )
    axs[row, col].set_title(f"PnLs for Cost: {cost*100}%")
    axs[row, col].set_xlabel("Profit-loss")
    axs[row, col].set_ylabel("Number of events")
    axs[row, col].legend()
    axs[row, col].legend(fontsize='small')

plt.tight_layout()
plt.show()


# The histograms of the profit and loss after hedging look like above.
# 
# The no-transaction band network saves on transaction cost while avoiding great losses.

# ### Evaluate the best premium of the derivative
# 
# Now, we are ready to define the premium of the derivative.
# 
# Premium of a derivative is defined as the guaranteed amount of cash which is as preferable as the profit-loss after hedging in terms of the exponential utility.

# In[ ]:


def evaluate_premium(
    hedging_model: torch.nn.Module,
    payoff: typing.Callable[[torch.Tensor], torch.Tensor],
    cost: float,
    n_times=20,
) -> float:
    """
    Evaluate the premium of the given derivative.

    Parameters
    ----------
    - hedging_model : torch.nn.Module
        Hedging model to fit.
    - payoff : callable[[torch.Tensor], torch.Tensor]
        Payoff function of the derivative to hedege.
    - cost : float, default 0.0
        Transaction cost of underlying asset.
    - n_times : int, default 20
        If `n_times > 1`, return ensemble mean of the results
        from multiple simulations.

    Returns
    -------
    premium : float
    """
    with torch.no_grad():
        p = lambda: -to_premium(
            compute_profit_and_loss(hedging_model, payoff, cost=cost)
        ).item()
        return float(np.mean([p() for _ in range(n_times)]))


# The no-transaction band network allows for a cheaper price:

# In[ ]:


torch.manual_seed(42)
premium_ntb = evaluate_premium(model_ntb, european_option_payoff, cost=1e-3)
torch.manual_seed(42)
premium_ffn = evaluate_premium(model_ffn, european_option_payoff, cost=1e-3)

print(f"Feed-forward network European call premium:\t {round(premium_ffn*100, 2)}%")
print(f"No-transaction band network European call premium:\t {round(premium_ntb*100, 2)}%")


# ## Experiment: Lookback Option
# 
# Let us carry out the same experiment for a lookback option.
# 
# Although we omit the cumulative maximum of the asset price, which is an important feature for a lookback option, for simlicity, the no-transaction band network attains a fairly good hedging strategy.

# ### Create hedging models

# In[ ]:


torch.manual_seed(42)
model_ntb = NoTransactionBandNet().to(DEVICE)
torch.manual_seed(42)
model_ffn = FeedForwardNet().to(DEVICE)


# ### Fit hedging models

# In[ ]:


torch.manual_seed(42)
history_ntb = fit(model_ntb, lookback_option_payoff, cost=1e-3)
torch.manual_seed(42)
history_ffn = fit(model_ffn, lookback_option_payoff, cost=1e-3)


# In[ ]:


plt.figure()
plt.plot(history_ntb, label="No-transaction band Network")
plt.plot(history_ffn, label="Feed-forward Network")
plt.xlabel("Number of epochs")
plt.ylabel("Loss (Negative of expected utility)")
plt.title("Learning histories for a lookback option")
plt.legend()
plt.show()


# Again, the above training histories exhibits that the no-transaction band network can be trained much quicker than the ordinary feed-forward network.
# 
# Surprisingly, the no-transaction band network achieves its optima as fast as it learns to hedge a European option, even though the lookback option bears further complication of path-dependence and needs more features.

# ### Evaluate the best premium of the derivative

# The no-transaction band network again allows for a cheaper price.

# In[ ]:


torch.manual_seed(42)
premium_ntb = evaluate_premium(model_ntb, lookback_option_payoff, cost=1e-3)
torch.manual_seed(42)
premium_ffn = evaluate_premium(model_ffn, lookback_option_payoff, cost=1e-3)

print(f"Feed-forward network lookback call premium:\t {round(premium_ffn*100, 2)}%")
print(f"No-transaction band network lookback call premium:\t {round(premium_ntb*100, 2)}%")


# In[ ]:




