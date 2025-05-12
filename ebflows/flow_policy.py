import torch
import torch.nn as nn

from .flows import initializeFlow

class FlowPolicy(nn.Module):
    def __init__(
        self,
        alpha,         # Temperature parameter
        log_sigma_max, # Maximum log standard deviation
        log_sigma_min, # Minimum log standard deviation
        action_sizes,  # Size of the action space
        state_sizes,   # Size of the state space
        device
    ):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.action_shape = action_sizes

        # We reuse this initialization function from the MEow paper
        flows, prior_dist = initializeFlow(log_sigma_max, log_sigma_min, action_sizes, state_sizes)

        # A cascade of energy-based normalizing flow layers
        self.flows = nn.ModuleList(flows).to(self.device)

        # Gaussian prior (ConditionalDiagLinearGaussian)
        self.prior_dist = prior_dist.to(self.device)

    def forward(self, obs, act_latent):
        """
        Uses a series of forward flows to determine the action-space
        representation of an action, given its latent-space representation

        Args:
            obs: a tensor of observations
            act: a tensor of proposed latent actions

        Returns:
            true action-space representations of the latent actions,
            with the log-determinant adjustment (chain rule)
        """
        num_actions = len(act_latent)
        act_true = act_latent
        log_det_adj = torch.zeros(num_actions, dtype=torch.float32).to(self.device)

        # Iterate over the flows, in forwards order
        for flow in self.flows:
            act_true, log_det = flow.forward(act_true, context=obs)

            # We keep adjusting the log determinant with each flow
            log_det_adj -= log_det

        # Return the true (action-space) action and true log-probability adjustment
        return act_true, log_det_adj

    def inverse(self, obs, act_true):
        """
        Reverses the forward function, by returning a latent representation
        (within the Gaussian prior) for a given action

        Args:
            obs: a tensor of observations
            act: a tensor of true actions

        Returns:
            latent-space representations of the true action-space actions,
            with the log-determinant adjustment (chain rule)
        """
        num_actions = len(act_true)
        act_latent = act_true
        log_det_adj = torch.zeros(num_actions, dtype=torch.float32).to(self.device)

        # Iterate over the flows, in backwards order
        for flow in reversed(self.flows):

            # We apply the inverse flow
            act_latent, log_det = flow.inverse(act_latent, context=obs)

            # We keep adjusting the log determinant with each flow
            log_det_adj += log_det

        # Return the latent action representation, with its log-prob adjustment
        return act_latent, log_det_adj

    def sample(self, num_samples, obs, deterministic=False):
        """
        Samples a set of actions from the space

        If deterministic, we use the means of many suggested latent actions.
        During training, we sample several latent actions.

        Args:
            num_samples: the number of samples
            obs: a tensor of observations
            deterministic: True for eval, False for train

        Returns:
            a tensor of sampled actions
        """

        # Take the average of many latent samples
        if deterministic:
            latent_samples = torch.randn((num_samples,) + self.prior_dist.shape, dtype=torch.float32).to(self.device)
            act_latent, _ = self.prior_dist.get_mean_std(latent_samples, context=obs)
            log_prob_latent = self.prior_dist.log_prob(act_latent, context=obs)

        # Directly sample from the latent
        else:
            act_latent, log_prob_latent = self.prior_dist.sample(context=obs, num_samples=num_samples)

        # Transform to the true (action-space) representation of actions
        act_true, log_det_adj = self.forward(obs=obs, act_latent=act_latent)
        log_prob_true = log_prob_latent - log_det_adj
        return act_true, log_prob_true

    def get_qv(self, obs, act_true):
        """
        Computes exact soft value functions (action-value and state-value)

        Args:
            obs: a tensor of observations
            act_true: a tensor of action-space actions

        Returns:
            soft Q-function, soft V-function
        """

        # Create empty tensors
        num_actions = len(act_true)
        q = torch.zeros((num_actions)).to(self.device)
        v = torch.zeros((num_actions)).to(self.device)

        # Compute the soft value functions using each flows, moving closer to the latent space
        act_latent = act_true
        for flow in reversed(self.flows):
            act_latent, q_adj, v_adj = flow.get_qv(act_latent, context=obs)

            # Adjust the current Q and V
            q += q_adj
            v += v_adj

        # Compute the Q and V in the latent space, and adjust the current Q and V
        q_adj, v_adj = self.prior_dist.get_qv(act_latent, context=obs)
        q += q_adj
        v += v_adj

        # Temperature scaling of the soft values, then we return
        q_scaled = q * self.alpha
        v_scaled = v * self.alpha
        return q_scaled[:, None], v_scaled[:, None]

    def get_v(self, obs):
        """
        Computes exact soft value functions (just state-value)

        Args:
            obs: a tensor of observations

        Returns:
            a tensor of sampled actions
        """

        # Calculate empty tensors
        num_observations = len(obs)
        act_true = torch.zeros((num_observations, self.action_shape)).to(self.device)
        v = torch.zeros((num_observations)).to(self.device)

        # Compute the soft-value functions
        act_latent = act_true
        for flow in reversed(self.flows):
            act_latent, _, v_adj = flow.get_qv(act_latent, context=obs)
            v += v_adj

        # Compute V in the latent space, and adjust the current V
        _, v_adj = self.prior_dist.get_qv(act_latent, context=obs)
        v += v_adj

        # Temperature scaling of the soft value, then we return
        v = v * self.alpha
        return v[:, None]