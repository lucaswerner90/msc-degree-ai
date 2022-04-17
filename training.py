#%%
import torch
import argparse
import cv2
import numpy as np
from collections import namedtuple

import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from collections import namedtuple
from dataloader import train_dataset, test_dataset
from model.actor_critic import ActorCritic
from environment import DroneEnvironment
from torch.utils.tensorboard import SummaryWriter

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

writer = SummaryWriter(log_dir="runs/Actor-Critic-v1", flush_secs=10)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
eps = np.finfo(np.float32).eps.item()


def main():
    num_training_images = len(train_dataset)
    model.train(True)
    for epoch in range(args.epochs + 1):

        env = DroneEnvironment(train_dataset)
        env.current_image_index = 0
        rewards_mean = []

        for i_episode in range(num_training_images):

            # reset environment and episode reward
            state = env.reset()
            
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
            print("-------------------------------------------------------")
            print("Episode {}\t finished after {} timesteps with reward {}".format(i_episode+1, t+1, reward))
            
            cv2.imwrite("data/training_images/actor_critic/v1/training/image_{}_epoch_{}_reward_{}.png".format(i_episode, epoch, reward),env.get_image())
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
            
            print("Loss: {}".format(loss_fn.mean()))
            print("-------------------------------------------------------")
            writer.add_scalar("Train Loss", loss_fn.mean(), (epoch*num_training_images) + i_episode)
            writer.add_scalar("Train Rewards", np.mean(rewards), (epoch*num_training_images) + i_episode)
            
            loss_fn.mean().backward(retain_graph=True)
            optimizer.step()


            rewards_mean.append(np.mean(rewards))

            if i_episode % args.log_interval == 0 and i_episode > 0:
                print('Epoch {} Episode {}\t Average reward: {:.2f}'.format(
                    epoch, i_episode, np.mean(rewards_mean)))


        if (epoch+1) % 2 == 0:
            test(epoch)
            torch.save(model.state_dict(), "checkpoints/actor_critic_model_epoch_{}.pth".format(epoch+1))

def test(epoch):
    num_test_images = len(test_dataset)
    env = DroneEnvironment(test_dataset)
    rewards = []
    actions_taken = []
    model.eval()
    with torch.no_grad():
        for idx in range(1,num_test_images):

            state = env.reset()
            done = False
            num_actions = 0

            while not done:
                probs, state_value = model(state)
                probs, state_value = probs.squeeze(), state_value.squeeze()
                m = Categorical(probs)
                action = m.sample()
                state, reward, done, _ = env.step(ACTIONS[action])
                num_actions+=1
                
                if done:
                    rewards.append(reward)
                    actions_taken.append(num_actions)
                    cv2.imwrite("data/training_images/actor_critic/v1/testing/image_{}_epoch_{}_actions_taken_{}.png".format(idx, epoch, num_actions),env.get_image())
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

    model.train(True)

if __name__ == '__main__':
    main()