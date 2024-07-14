import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """A neural network that estimates the mean and standard deviation of a normal distribution
        from which the agent's action is sampled."""

    def __init__(self, obs_dim, action_dim, hidden_size1=32, hidden_size2=32) -> None:
        """
        Args:
            obs_dim (int): Dimension of the observation space
            action_dim (int): Dimension of the action space
            hidden_size1 (int): Size of the first hidden layer
            hidden_size2 (int): Size of the second hidden layer
        """
        super().__init__()

        # Shared layers
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )

        # Mean output layer
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_size2, action_dim)
        )

        # Log of standard deviation output layer
        self.log_std_net = nn.Sequential(
            nn.Linear(hidden_size2, action_dim)
        )

    def forward(self, x: torch.Tensor):
        """Given an observation, this function returns the means and standard deviations of
        the normal distributions from which the action components are sampled.

        Args:
            x (torch.Tensor): Observation from the environment
        Returns:
            means: Predicted means of the normal distributions
            stddevs: Predicted standard deviations of the normal distributions
        """
        shared_features = self.shared_net(x)

        means = self.mean_net(shared_features)
        log_stddevs = self.log_std_net(shared_features)

        stddevs = torch.exp(log_stddevs) + 1e-6

        return means, stddevs
