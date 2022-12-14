import argparse
from tensorboardX import SummaryWriter
import os
import gym
import numpy as np
import itertools
import random
import torch
from sac import Agent
from common.buffers import *
import time

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--noise_scale', type=float, default=0.1, metavar='G')
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--test_interval', type=int, default=20000, metavar='N',
                    help='Test Steps')
parser.add_argument('--device',default='cuda')
parser.add_argument('--num_steps', type=int, default=2000001,metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--start_step',type=int,default=10000)
parser.add_argument('--lr',type=float,default=3e-4)
parser.add_argument('--alpha',type=float,default=0.2)
parser.add_argument('--automatic_entropy_tuning',type=bool,default=False)

parser.add_argument('--test_num', type=int, default=1)
args = parser.parse_args()

env = gym.make(args.env_name)
# Initialize environment
env = gym.make(args.env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]
device=torch.device(args.device)
interactions = []
rewards = []

print('---------------------------------------')
print('Environment:', args.env_name)
print('Algorithm:SAC')
print('State dimension:', obs_dim)
print('Action dimension:', act_dim)
print('Action limit:', act_limit)
print('---------------------------------------')

# Set a random seed
seed=random.randint(0,1e9)
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

test_env=list()
for i in range(args.test_num):
    test=gym.make(args.env_name)
    test.seed(seed+10*i)
    test_env.append(test)

i=0
while True:
    if os.path.exists('run/{}/sac/{}'.format(args.env_name, i)):
        i += 1
    else:
        break
writer = SummaryWriter('run/{}/sac/{}'.format(args.env_name, i))
agent = Agent(env, args, device, obs_dim, act_dim, act_limit,
              alpha=args.alpha,  # In HalfCheetah-v2 and Ant-v2, SAC with 0.2
              hidden_sizes=(args.hidden_size,args.hidden_size),  # shows the best performance in entropy coefficient
              buffer_size=args.replay_size,  # while, in Humanoid-v2, SAC with 0.05 shows the best performance.
              batch_size=args.batch_size,
              policy_lr=args.lr,
              qf_lr=args.lr,
              automatic_entropy_tuning=args.automatic_entropy_tuning)

# Experience buffer
memory = ReplayBuffer(obs_dim, act_dim, args.replay_size, device)

total_numsteps = 0
updates = 0
step_number=0

for i_episode in itertools.count(1):
    state = env.reset()
    if total_numsteps > args.num_steps:
        break

    step_number=0
    done=False
    while not done:


        if total_numsteps>args.start_step:
            action = agent.select_action(torch.Tensor(state).to(device))
        else:
            action=env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        total_numsteps += 1

        memory.add(state, action, reward, next_state, done)

        state = next_state

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                agent.update_parameters(memory)

        if total_numsteps % args.test_interval == 0:
            org_state=state
            avg_reward = 0.
            for test in test_env:
                state = test.reset()
                episode_reward= 0
                done = False
                while not done:
                    action = agent.select_action(torch.Tensor(state).to(device),evaluate=True)
                    next_state, reward, done, _ = test.step(action)
                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= args.test_num
            writer.add_scalar('score/score', avg_reward, total_numsteps)

            print("----------------------------------------")
            print("Total Numsteps: {}, Avg. Reward: {}".format(total_numsteps, round(avg_reward, 2)))
            print("----------------------------------------")
            done=False
            state=org_state

env.close()
