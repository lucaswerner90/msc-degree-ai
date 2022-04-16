#%%
import torch
import pandas as pd
import argparse
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from collections import namedtuple
from dataloader import train_dataloader, test_dataloader
from model.actor_critic import ActorCritic
from environment import DroneEnvironment


ACTIONS = ["LEFT","RIGHT","NONE"]
(WIDTH, HEIGHT, CHANNELS) = (640,360,3)
VALUE_LOSS_COEF = 0.5
GAMMA = 0.99

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='epochs (default: 20)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--learning-rate', type=float, default=1e-5, metavar='N',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


torch.manual_seed(args.seed)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
eps = np.finfo(np.float32).eps.item()


def main():
    num_training_images = len(train_dataloader)
    for epoch in range(1, args.epochs + 1):

        env = DroneEnvironment(train_dataloader)

        for i_episode in range(num_training_images):

            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0
            log_probs = []
            rewards = []
            values = []
            # for each episode, only run 99 steps so that we don't 
            # infinite loop while learning
            for t in range(100):

                # select action from policy
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)

                probs, state_value = model(state)
                probs, state_value = probs.squeeze(), state_value.squeeze()

                # create a categorical distribution over the list of probabilities of actions
                m = Categorical(probs)

                # and sample an action using the distribution
                action = m.sample()

                # take the action
                state, reward, done, _ = env.step(ACTIONS[action])

                log_probs.append(probs)
                rewards.append(reward)
                values.append(state_value)

                if done:
                    break
            
            print("Episode finished after {} timesteps".format(t+1))
            
            R = 0
            if not done:
                _, value = model(state)
                R = value.data

            values.append(R)
            policy_loss = 0
            value_loss = 0

            for i in reversed(range(len(rewards))):
                R = GAMMA * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)
                policy_loss = policy_loss - (log_probs[i] * Variable(advantage))

            optimizer.zero_grad()
            loss_fn = (policy_loss + VALUE_LOSS_COEF * value_loss)
            loss_fn.backward(retain_graph=True)
            optimizer.step()

            if i_episode % args.log_interval == 0 and i_episode > 0:
                print('Epoch {} Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    epoch, i_episode, ep_reward, np.mean(rewards)))

if __name__ == '__main__':
    main()