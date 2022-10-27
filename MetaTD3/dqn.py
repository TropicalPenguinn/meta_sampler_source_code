import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from common.utils import *
from common.networks import *


class Meta_Aent(object):
    """An implementation of the Deep Q-Network (DQN), Double DQN agents."""

    def __init__(self,
                 env,
                 args,
                 device,
                 obs_dim,
                 act_num,
                 steps=0,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 buffer_size=int(1e4),
                 batch_size=64,
                 target_update_step=100,
                 eval_mode=False,
                 q_losses=list(),
                 logger=dict(),
                 ):

        self.env = env
        self.args = args
        self.device = device
        self.obs_dim = obs_dim
        self.act_num = act_num
        self.steps = steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_step = target_update_step
        self.eval_mode = eval_mode
        self.q_losses = q_losses
        self.logger = logger

        # Main network
        self.qf = MLP(self.obs_dim, self.act_num).to(self.device)
        # Target network
        self.qf_target = MLP(self.obs_dim, self.act_num).to(self.device)

        # Initialize target parameters to match main parameters
        hard_target_update(self.qf, self.qf_target)

        # Create an optimizer
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=1e-3)



    def select_action(self, obs):
        """Select an action from the set of available actions."""
        # Decaying epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)

        if np.random.rand() <= self.epsilon:
            # Choose a random action with probability epsilon
            return np.random.randint(self.act_num)
        else:
            # Choose the action with highest Q-value at the current state
            action = self.qf(obs).argmax()
            return action.detach().cpu().numpy()

    def update_parameters(self,replay_buffer):
        batch = replay_buffer.sample(self.batch_size)
        obs1 = batch['obs1']
        obs2 = batch['obs2']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']

        if 0:  # Check shape of experiences
            print("obs1", obs1.shape)
            print("obs2", obs2.shape)
            print("acts", acts.shape)
            print("rews", rews.shape)
            print("done", done.shape)

        # Prediction Q(s)
        q = self.qf(obs1).gather(1, acts.long()).squeeze(1)

        # Target for Q regression
        q2 = self.qf(obs2)
        q_target = self.qf_target(obs2)
        q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))
        q_backup = rews + self.gamma * (1 - done) * q_target.max(1)[0]
        q_backup.to(self.device)

        if 0:  # Check shape of prediction and target
            print("q", q.shape)
            print("q_backup", q_backup.shape)

        # Update perdiction network parameter
        qf_loss = F.mse_loss(q, q_backup.detach())
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # Synchronize target parameters 𝜃‾ as 𝜃 every C steps
        if self.steps % self.target_update_step == 0:
            hard_target_update(self.qf, self.qf_target)

        # Save loss
        self.q_losses.append(qf_loss.item())
