#%%
import torch
import argparse
import cv2
import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from dataloader import train_dataset, test_dataset
from model.actor_critic import ActorCritic
# from model.visual_transformer_actor_critic import VisualEncoderActorCritic
from environment import DroneEnvironment
from torch.utils.tensorboard import SummaryWriter

ACTIONS = ["LEFT","RIGHT","NONE"]
(WIDTH, HEIGHT, CHANNELS) = (640,360,3)
VALUE_LOSS_COEF = 0.8
MAX_STEPS = 50
GAMMA = 0.99

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--experiment-name', type=str, default='ac-no-rewards-till-complete', metavar='N',
                    help='Whatever name you want')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='epochs (default: 100)')
parser.add_argument('--gamma', type=float, default=GAMMA, metavar='G',
                    help='discount factor (default: 0.95)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='N',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='interval between training status logs (default: 100)')
args = parser.parse_args()


torch.manual_seed(args.seed)

experiment_name = args.experiment_name
images_testing_dir = f'data/training_images/ac/{experiment_name}/testing'
images_training_dir = f'data/training_images/ac/{experiment_name}/training'

if not os.path.exists(images_training_dir):
    os.makedirs(images_training_dir)
if not os.path.exists(images_testing_dir):
    os.makedirs(images_testing_dir)

writer = SummaryWriter(log_dir=f"runs/Actor-Critic-{experiment_name}", flush_secs=10)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
eps = np.finfo(np.float32).eps.item()

def main():
    num_training_images = len(train_dataset)
    model.train()
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
            actions_taken_on_image = 0

            for t in range(MAX_STEPS):
                image, point = state
                logits, state_value = model(image,point)
                logits, state_value = logits.squeeze(), state_value.squeeze()
                log_prob,action = model.select_action(logits)
                # take the action
                selected_action = model.actions[action.item()]
                state, reward, done, _ = env.step(selected_action)

                actions_taken[selected_action]+=1
                actions_taken_on_image+=1

                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(state_value)

                if done:
                    break
            
            rewards = torch.tensor(rewards, dtype=torch.float)

            # calculate the true value using rewards returned from the environment
            R = 0
            policy_loss = 0
            value_loss = 0
            for i in reversed(range(len(rewards))):
                R = GAMMA * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)
                policy_loss = policy_loss - (log_probs[i] * Variable(advantage))

            # reset gradients
            optimizer.zero_grad()

            # sum up all the values of policy_losses and value_losses
            loss_fn = (policy_loss + VALUE_LOSS_COEF * value_loss)
            loss_fn.backward(retain_graph=True)

            writer.add_scalar("Train Loss", loss_fn.item(), (epoch*num_training_images) + i_episode)
            writer.add_scalar("Train Rewards", rewards.mean(), (epoch*num_training_images) + i_episode)
            writer.add_scalar("Episodes duration", t+1, (epoch*num_training_images) + i_episode)
            writer.add_scalar("Value Loss", value_loss.item(), (epoch*num_training_images) + i_episode)
            writer.add_scalar("Policy Loss", policy_loss.item(), (epoch*num_training_images) + i_episode)


            # perform backprop
            optimizer.step()
            rewards_mean.append(rewards.mean())

            if i_episode % args.log_interval == 0 and i_episode > 0:
                print('{} Epoch {} Episode {} Left:{} \t Right: {}\t None: {}\t'.format(experiment_name, epoch, i_episode, actions_taken['LEFT'], actions_taken['RIGHT'], actions_taken['NONE']))
                cv2.imwrite(
                    os.path.join(images_training_dir, "image_{}_epoch_{}_reward_{}.png".format(i_episode+1,epoch,reward)),
                    env.get_image()
                )
                rewards_mean = []
                actions_taken = {'LEFT':0, 'RIGHT':0, 'NONE':0}


        if (epoch+1) % 2 == 0:
            test(epoch)
            torch.save(model.state_dict(), "checkpoints/ac_{}_epoch_{}.pth".format(experiment_name, epoch+1))

        writer.flush()

def test(epoch):
    num_test_images = len(test_dataset)
    env = DroneEnvironment(test_dataset)
    rewards = []
    actions_taken = []
    model.eval()
    with torch.no_grad():
        for idx in range(1, num_test_images+1):

            state = env.reset(eval=True)
            num_actions = 0

            while True:
                image, point = state
                logits, state_value = model(image,point)
                logits, state_value = logits.squeeze(), state_value.squeeze()
                _, action = model.select_action(logits)
                state, reward, _, _ = env.step(model.actions[action.item()])
                num_actions+=1
                
                if model.actions[action.item()] == model.stop_action:
                    rewards.append(reward)
                    print(f'Epoch {epoch} \tTesting: {idx}/{num_test_images} - {num_actions} actions taken - Reward: {reward}')
                    actions_taken.append(num_actions)
                    cv2.imwrite(
                        os.path.join(images_testing_dir, "{}_epc_{}_actions_{}_rew_{}.png".format(idx, epoch, num_actions, reward)),
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