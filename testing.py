#%%
import torch
import os
import cv2
import numpy as np
from model.actor_critic import ActorCritic
from dataloader import validation_dataset
from environment import DroneEnvironment
from torch.distributions import Categorical

torch.manual_seed(42)

ACTIONS = ["LEFT","RIGHT","NONE"]



def run_experiment(experiment_name):
    model_file = f'checkpoints/{experiment_name}.pth'
    images_validation_dir = f'data/validation_images/actor_critic/{experiment_name}'

    if not os.path.exists(images_validation_dir):
        os.makedirs(images_validation_dir)


    model = ActorCritic()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    env = DroneEnvironment(validation_dataset)
    num_validation_images = len(validation_dataset)

    rewards = []
    num_actions_taken = []
    for i in range(1, num_validation_images+1):
        state = env.reset(True)
        num_actions = 0
        while True:
            probs, _ = model(state)
            # create a categorical distribution over the list of probabilities of actions
            m = Categorical(probs.squeeze())
            action = m.sample()
            _, reward, _, _ = env.step(ACTIONS[action])
            if ACTIONS[action] != "NONE":
                num_actions+=1
            else:
                rewards.append(reward)
                num_actions_taken.append(num_actions)
                image_description = "image_{}_actions_taken_{}_reward_{}.png".format(i, num_actions, reward)
                image_filename = os.path.join(images_validation_dir, image_description)
                cv2.imwrite(image_filename, env.get_image())
                num_actions = 0
                break

    return np.mean(rewards), np.mean(num_actions_taken)

if __name__ == "__main__":
    experiments = [
        'actor_critic_model_experiment_name_ac-discourage-reward-1_epoch_',
        'actor_critic_model_experiment_name_ac-discourage-reward-2_epoch_',
    ]
    epochs = [2,6,10,14,20]
    for experiment in experiments:    
        for epoch in epochs:
            experiment_name = f'{experiment}{epoch}'
            rewards, num_actions = run_experiment(experiment_name)
            print('-------------------------------------------------------')
            print(f'Experiment: {experiment} epoch: {epoch}')
            print(f'Experiment with epoch: {epoch}')
            print(f'Average reward: {np.mean(rewards)}')
            print(f'Average num actions taken: {np.mean(num_actions)}')
            print('-------------------------------------------------------')