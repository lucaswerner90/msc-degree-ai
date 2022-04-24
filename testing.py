#%%
import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
from model.actor_critic import ActorCritic
from dataloader import validation_dataset
from environment import DroneEnvironment

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
            logits, state_value = model(state)
            logits, state_value = logits.squeeze(), state_value.squeeze()
            prob = F.softmax(logits, -1)
            # create a categorical distribution over the list of probabilities of actions
            action = prob.multinomial(num_samples=1)
            log_prob = F.log_softmax(logits, -1)
            log_prob = log_prob.gather(0, action)

            # take the action
            selected_action = ACTIONS[action.item()]
            state, reward, _, _ = env.step(selected_action)
            if selected_action != "NONE":
                num_actions+=1
            else:
                rewards.append(reward)
                num_actions_taken.append(num_actions)
                image_description = "img_{}_ataken_{}_rew_{}.png".format(i, num_actions, reward)
                image_filename = os.path.join(images_validation_dir, image_description)
                cv2.imwrite(image_filename, env.get_image())
                num_actions = 0
                break

    return np.mean(rewards), np.mean(num_actions_taken)

if __name__ == "__main__":
    experiments = [
        'ac_ac-reward-2-normalized-dataframe-lr-e7_epoch_',
    ]
    epochs = [4,8,10,20,60,100]
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