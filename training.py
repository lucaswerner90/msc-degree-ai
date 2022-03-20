import argparse
from itertools import count
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from model.agent import PolicyNet
from sklearn.model_selection import train_test_split
from torch.distributions import Categorical
from torch.autograd import Variable
from torchvision import transforms

NUM_EPISODES = 50
MAX_STEPS_PER_IMAGE = 500
ACTIONS = ["LEFT", "RIGHT", "NONE"]

NUM_ACTIONS = len(ACTIONS)
# Make the results reproducible
np.random.seed(42)

writer = SummaryWriter()

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def calculate_new_position(image_width, point_of_view, action):
    distance = 15  # distancia en pixeles
    if action == "LEFT":
        return max(1, point_of_view - distance)  # 200 => 200 - distancia
    if action == "RIGHT":
        return min(image_width - 1, point_of_view + distance)
    return point_of_view


def calculate_reward(image_width, real_x_coordinate, predicted_x_coordinate):
    """
    Divides the images into 32 sections of fixed length and returns
    1 if the predicted_x_coordinate is in the same section as
    the real_x_coordinate.
    Otherwise it returns a result directly proportional to the distance
    between the two sections, the closer to 1, the closer the 2 sections are.
    """
    num_sections = 32
    section_width = image_width // num_sections
    num_sections_away = abs(predicted_x_coordinate - real_x_coordinate) // section_width
    return 1 if num_sections_away == 0 else 1 - (num_sections_away / num_sections)


def generate_image(image,score,real_coordinates,predicted_coordinates):
	real_x, real_y = real_coordinates
	predicted_x, predicted_y = predicted_coordinates
	image = cv2.circle(image, (predicted_x,predicted_y), radius=4, color=(0, 255, 255), thickness=10)
	image = cv2.circle(image, (real_x,real_y), radius=4, color=(0, 0, 255), thickness=10)
	image = cv2.putText(image,f'Score: {score}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
	return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/dataframe.csv")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--steps", type=int, default=MAX_STEPS_PER_IMAGE)

    # If we want to resume a previous training we can load the model weights
    # instead of starting from scratch
    parser.add_argument("--model-weights", type=str)
    opt = parser.parse_args()

    if opt.steps < 1:
        raise ValueError("You need to go through each image at least once")

    if opt.episodes < 1:
        raise ValueError("You need to train at least 1 epoch")

    if not opt.data:
        raise ValueError("You must specify a CSV path")

    if not os.path.exists(opt.data):
        raise FileNotFoundError(f'"{opt.data}" doesn\'t exist')

    # TODO: Load the model weights
    if opt.model_weights:
        pass

    # Read the dataframe and split into train/test/validation
    df = pd.read_csv(opt.data)
    train, validation = train_test_split(
        df, test_size=0.1, random_state=42, shuffle=True
    )
    train, test = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)

    agent = PolicyNet(NUM_ACTIONS)

    hparams = dict(
        learning_rate = 1e-4,
        train_epochs = len(train),
        batch_size = 16,
        gamma = 0.99
    )
    
    optimizer = torch.optim.RMSprop(agent.parameters(), lr=hparams["learning_rate"])

    # Batch history
    episodes_duration = []
    action_pool = []
    reward_pool = []
    state_pool = []
    steps = 0

    # writer.add_hparams(hparams)
    actions_taken = {
        "LEFT":0,
        "RIGHT":0,
        "NONE":0
    }
    for episode in range(hparams["train_epochs"]):
        row = train.iloc[episode]
        filename, real_x, real_y = (
            row["filename"],
            row["prediction_x"],
            row["prediction_y"],
        )

        # Read and transform the original image to be able to fit it
        # into our model correctly
        original_image = cv2.imread(filename)
        _, img_width, img_height = original_image.shape
        img = agent.preprocess(original_image)
        img = agent.pretrained_model(torch.unsqueeze(img, 0))
        img = img.squeeze()
        point_of_view = torch.Tensor([0.5])  # img_width/2

        initial_state = torch.concat((img, point_of_view))
        state = Variable(initial_state)
        # For now we will consider only the X position as the point of view of
        # our model.
        # point_of_view = np.array([round(img_width/2), round(img_height/2)])
        print('-------------------------------------------------')
        for t in count():
            # Approximate the model prediction to the real value
            probs = agent(torch.unsqueeze(state, 0))
            probs = probs.squeeze()
            m = Categorical(probs)
            action = m.sample().item()

            actions_taken[ACTIONS[action]]+=1
            # Calculate reward based on the current point of view and the action selected
            point_of_view = round(point_of_view.item() * img_width)
            next_point_of_view = calculate_new_position(
                img_width, point_of_view, ACTIONS[action]
            )
            reward = calculate_reward(img_width, real_x, next_point_of_view)

            reward_pool.append(reward)
            action_pool.append(action)
            state_pool.append(state.squeeze())

            point_of_view = torch.Tensor([next_point_of_view / img_width])
            state = torch.concat((img, point_of_view))
            state = Variable(state)

            steps += 1

            if reward == 1 or t > MAX_STEPS_PER_IMAGE:
                # image = generate_image(
                #     original_image,
                #     reward,
                #     (real_x, real_y),
                #     (round(point_of_view.item()*img_width),round(img_height/2))
                # )
                # image = (image * 255).astype(np.uint8)
                # writer.add_image('Predicted vs real', image, steps)
            
                episodes_duration.append(t + 1)
                print('-------------------------------------------------')
                print(f'Episode {episode} \t duration:{t} \t\t Last Reward: {reward}')
                print(f'Action LEFT taken:{actions_taken["LEFT"]}')
                print(f'Action RIGHT taken:{actions_taken["RIGHT"]}')
                print(f'Action NONE taken:{actions_taken["NONE"]}')
                print('-------------------------------------------------')

                for act in ACTIONS:
                    actions_taken[act]=0

                break
        
        # Update the policy after batch size steps
        if episode > 0 and episode % hparams["batch_size"] == 0:
            running_add = 0
            for i in reversed(range(steps)):
                running_add = running_add * hparams["gamma"] + reward_pool[i]
                reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)

            writer.add_scalar('Reward mean', reward_mean, episode)
            writer.add_scalar('Reward std', reward_std, episode)

            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Descent
            optimizer.zero_grad()

            print('Calculating loss...')

            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.Tensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = agent(torch.unsqueeze(state, 0))
                probs = probs.squeeze()
                m = Categorical(probs)
                loss = -m.log_prob(action) * reward  # Negative score function x reward
                loss.backward()

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0
            
            # Save the model
            print('Saving the model...')
            torch.save(agent, "policy_gradient.pth")

    writer.close()
