#%%
import torch
import argparse
import cv2
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from torch.distributions import Categorical
from torch.autograd import Variable
from dataloader import train_dataset, test_dataset
from model.actor_critic import ActorCritic
from environment import DroneEnvironment
from torch.utils.tensorboard import SummaryWriter

ACTIONS = ["LEFT","RIGHT","NONE"]
(WIDTH, HEIGHT, CHANNELS) = (640,360,3)
VALUE_LOSS_COEF = 0.8
GAMMA = 0.95

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--experiment-name', type=str, default='ac-discourage-reward-1', metavar='N',
                    help='epochs (default: 20)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='epochs (default: 20)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--learning-rate', type=float, default=1e-5, metavar='N',
                    help='learning rate (default: 1e-5)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='interval between training status logs (default: 5)')
args = parser.parse_args()


torch.manual_seed(args.seed)

experiment_name = args.experiment_name
images_testing_dir = f'data/training_images/actor_critic/{experiment_name}/testing'
images_training_dir = f'data/training_images/actor_critic/{experiment_name}/training'

if not os.path.exists(images_training_dir):
    os.makedirs(images_training_dir)
if not os.path.exists(images_testing_dir):
    os.makedirs(images_testing_dir)

writer = SummaryWriter(log_dir=f"runs/Actor-Critic-{experiment_name}", flush_secs=10)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4, eps=1e-5)
eps = np.finfo(np.float32).eps.item()

def main():
    num_training_images = len(train_dataset)
    model.train(True)
    for epoch in range(args.epochs + 1):
        actions_taken = {'LEFT':0, 'RIGHT':0, 'NONE':0}

        env = DroneEnvironment(train_dataset)
        env.current_image_index = 0
        rewards_mean = []

        for i_episode in range(num_training_images):

            # reset environment and episode reward
            state = env.reset()
            
            log_probs = []
            rewards = []
            values = []

            for t in range(50):

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

                actions_taken[ACTIONS[action]]+=1

                log_probs.append(m.log_prob(action))
                rewards.append(reward)
                values.append(state_value)

                if done:
                    break

            policy_losses = [] # list to save actor (policy) loss
            value_losses = [] # list to save critic (value) loss
            returns = [] # list to save the true values

            # calculate the true value using rewards returned from the environment
            R = 0
            for r in rewards[::-1]:
                # calculate the discounted value
                R = r + args.gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + eps)

            for log_prob, value, R in zip(log_probs, values, returns):
                advantage = R - value.item()

                # calculate actor (policy) loss 
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

            # reset gradients
            optimizer.zero_grad()

            # sum up all the values of policy_losses and value_losses
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

            writer.add_scalar("Train Loss", loss.item(), (epoch*num_training_images) + i_episode)
            writer.add_scalar("Train Rewards", np.mean(rewards), (epoch*num_training_images) + i_episode)
            writer.add_scalar("Episodes duration", t+1, (epoch*num_training_images) + i_episode)


            # perform backprop
            loss.backward()
            optimizer.step()

            rewards_mean.append(np.mean(rewards))

            if i_episode % args.log_interval == 0 and i_episode > 0:
                print('Epoch {} Episode {} Left:{} \t Right: {}\t None: {}\t'.format(epoch, i_episode, actions_taken['LEFT'], actions_taken['RIGHT'], actions_taken['NONE']))
                cv2.imwrite(
                    os.path.join(images_training_dir, "image_{}_epoch_{}_reward_{}.png".format(i_episode+1,epoch,reward)),
                    env.get_image()
                )
                rewards_mean = []
                actions_taken = {'LEFT':0, 'RIGHT':0, 'NONE':0}


        if (epoch+1) % 2 == 0:
            test(epoch)
            torch.save(model.state_dict(), "checkpoints/actor_critic_model_experiment_name_{}_epoch_{}.pth".format(experiment_name, epoch+1))


def test(epoch):
    num_test_images = len(test_dataset)
    env = DroneEnvironment(test_dataset)
    rewards = []
    actions_taken = []
    model.eval()
    with torch.no_grad():
        for idx in range(1, num_test_images+1):

            state = env.reset(eval=True)
            done = False
            num_actions = 0

            while not done:
                probs, state_value = model(state)
                probs, state_value = probs.squeeze(), state_value.squeeze()
                m = Categorical(probs)
                action = m.sample()
                state, reward, done, _ = env.step(ACTIONS[action])
                num_actions+=1
                
                if ACTIONS[action] == 'NONE':
                    done = True
                    rewards.append(reward)
                    actions_taken.append(num_actions)
                    cv2.imwrite(
                        os.path.join(images_testing_dir, "image_{}_epoch_{}_actions_taken_{}_reward_{}.png".format(idx, epoch, num_actions, reward)),
                        env.get_image()
                    )
                    break

        writer.add_scalar(
            "Test Mean Reward", 
            np.mean(rewards),
            epoch+1
        )
        writer.add_scalar(
            "Test Num Actions till finish",
            np.mean(actions_taken),
            epoch+1
        )

    model.train()

if __name__ == '__main__':
    main()