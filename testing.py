#%%
import torch
import os
import cv2
import glob
import numpy as np
import torch.nn.functional as F
from model.visual_transformer_actor_critic import VisualEncoderActorCritic
# from model.actor_critic import ActorCritic
from dataloader import validation_dataset
from environment import DroneEnvironment
from PIL import Image

torch.manual_seed(42)

ACTIONS = ["LEFT","RIGHT","NONE"]

def save_image_for_gif(image, out_dir, image_number, step):
    image_description = "img_{}_step_{}.png".format(image_number, step)
    image_filename = os.path.join(
        out_dir,
        image_description
    )
    cv2.imwrite(
        image_filename,
        image
    )
def create_gif_from_images(src_dir, out_dir, image_number):
    # filepaths
    fp_in = f"{src_dir}/img_{image_number}_*.png"
    fp_out = f"{out_dir}/image_{image_number}.gif"
    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=400, loop=0)

def run_experiment(experiment_name):
    model_file = f'checkpoints/{experiment_name}.pth'
    images_validation_dir = f'data/validation_images/actor_critic/{experiment_name}'
    gif_validation_dir = f'data/gif_images/actor_critic/{experiment_name}'

    if not os.path.exists(images_validation_dir):
        os.makedirs(images_validation_dir)
    
    if not os.path.exists(gif_validation_dir):
        os.makedirs(gif_validation_dir)


    model = VisualEncoderActorCritic()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    env = DroneEnvironment(validation_dataset)
    num_validation_images = len(validation_dataset)

    rewards = []
    num_actions_taken = []
    for i in range(1, num_validation_images+1):
        print(f'Testing: {i}/{num_validation_images}')
        state = env.reset(True)
        num_actions = 0
        while True:
            image, point = state
            logits, _ = model(image,point)
            _, action = model.select_action(logits)
                # take the action
            selected_action = ACTIONS[action.item()]
            state, reward, _, _ = env.step(selected_action)
            save_image_for_gif(env.get_image(), gif_validation_dir, i, num_actions)
            if selected_action != "NONE":
                num_actions+=1
            else:
                rewards.append(reward)
                num_actions_taken.append(num_actions)
                image_description = "img_{}_ataken_{}_rew_{}.png".format(i, num_actions, reward)
                image_filename = os.path.join(
                    images_validation_dir,
                    image_description
                )
                cv2.imwrite(
                    image_filename,
                    env.get_image()
                )
                num_actions = 0
                break
        create_gif_from_images(images_validation_dir, gif_validation_dir, i)


    return np.mean(rewards), np.mean(num_actions_taken)

if __name__ == "__main__":
    experiments = [
        'ac_ac-vit-encoder_epoch_',
    ]
    epochs = [2]
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