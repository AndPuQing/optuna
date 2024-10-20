import math
import torch

def custom(mean: torch.Tensor, var: torch.Tensor, f0: float) -> torch.Tensor:
    # Calculate the expected improvement
    z = (mean - f0) / torch.sqrt(var + 1e-9)  # Adding a small value to avoid division by zero
    expected_improvement = (mean - f0) * torch.distributions.Normal(0, 1).cdf(z) + \
                           torch.sqrt(var + 1e-9) * torch.distributions.Normal(0, 1).log_prob(z)

    # Define the 25th and 75th percentile thresholds
    q_25 = 12.779
    q_75 = 21.153

    # Define hyperparameters
    beta = 0.1  # Weight for exploration term
    gamma = 0.5  # Penalty for high-scoring regions

    # Calculate the exploration term based on quantiles
    exploration_term = torch.where(mean < q_25, 
                                   torch.tensor(1.0, device=mean.device), 
                                   torch.where((mean >= q_25) & (mean <= q_75), 
                                                (mean - q_25) / (q_75 - q_25), 
                                                -gamma))

    # Combine expected improvement with exploration term
    adaptive_acquisition = expected_improvement + beta * exploration_term

    return adaptive_acquisition