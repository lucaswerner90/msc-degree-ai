import random
import os
from itertools import count

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.autograd import Variable

from torchvision import transforms
import torchvision.models as models

MAX_STEPS_PER_IMAGE = 500
MAX_REWARD = 10
np.random.seed(42)

class PolicyNet(nn.Module):
    """
    PolicyNet class contains our Policy Gradient agent implementation
    that will run the training and the testing over our dataframe
    """
    def __init__(self, 
    actions, 
    hparams,
    writer = SummaryWriter(),
    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
    pretrained_model = models.vgg19(pretrained=True)
    ):
        super(PolicyNet, self).__init__()
        self.transforms = transforms
        self.writer = writer
        self.hparams = hparams
        self.actions = actions
        self.num_actions = len(self.actions)

        # Load the pretrained model that will preprocess the image
        # before moving it to the agent
        pretrained_model.classifier = nn.Sequential(
            *list(pretrained_model.classifier.children())[:-2]
        )
        for params in pretrained_model.parameters():
            params.requires_grad = False

        self.pretrained_model = pretrained_model

        # Define the model structure
        self.fc1 = nn.Linear(4096+1, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, self.num_actions)

    def calculate_next_movement(self, image_width:int, point_of_view:int, action:str) -> int:
        """
        Given the current position of our view, the total image width in pixels
        and the next action, we return the next point of view
        """
        distance = 20  # pixels distance
        if action == "LEFT":
            return max(1, point_of_view - distance)  # 200 => 200 - distancia
        if action == "RIGHT":
            return min(image_width - 1, point_of_view + distance)
        return point_of_view

    def calculate_reward(self, image_width, real_x_coordinate, predicted_x_coordinate):
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
        return MAX_REWARD if num_sections_away == 0 else 1 - (num_sections_away / num_sections)

    def select_action(self, probs: torch.Tensor) -> int:
        """
        Given the output of the model, we select the action
        based on a probability distribution and we return the
        position of the action in the actions array
        """
        m = Categorical(probs)
        return m.sample().item()


    def get_row(self,df,row_number):
        row = df.iloc[row_number]
        image = cv2.imread(row["filename"])
        return (
            image,
            row["prediction_x"],
            row["prediction_y"],
        )

    def train_model(self, df_train, df_test):
        self.train()
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams["learning_rate"])
    
        actions_taken = {
            "LEFT":0,
            "RIGHT":0,
            "NONE":0
        }

        # Batch history
        episodes_duration = []
        action_pool = []
        reward_pool = []
        state_pool = []
        steps = 0
        total_episodes = len(df_train)

        for episode in range(total_episodes):
            original_image, real_x, real_y = self.get_row(df_train,episode)
            # Read and transform the original image to be able to fit it
            # into our model correctly
            _, img_width, img_height = original_image.shape
            img = self.transforms(original_image)
            img = self.pretrained_model(torch.unsqueeze(img, 0))
            img = img.squeeze()

            # Calculate the initial point of view
            point_of_view = torch.Tensor([random.random()])  # img_width/2
            # point_of_view = torch.Tensor([0.5])  # img_width/2

            initial_state = torch.concat((img, point_of_view))
            state = Variable(initial_state)

            print('-------------------------------------------------')
            for t in count():
                # Approximate the model prediction to the real value
                probs = self.forward(torch.unsqueeze(state, 0))
                action = self.select_action(probs.squeeze())

                actions_taken[self.actions[action]]+=1

                # Calculate reward based on the current point of view and the action selected
                point_of_view = round(point_of_view.item() * img_width)
                next_point_of_view = self.calculate_next_movement(
                    img_width,
                    point_of_view, 
                    self.actions[action]
                )
                reward = self.calculate_reward(
                    img_width,
                    real_x,
                    next_point_of_view
                )

                reward_pool.append(reward)
                action_pool.append(action)
                state_pool.append(state.squeeze())

                point_of_view = torch.Tensor([next_point_of_view / img_width])
                state = torch.concat((img, point_of_view))
                state = Variable(state)

                steps += 1

                # If we reach the correct segment of the image, we know
                # that we're doing good and we can stop there
                if reward == MAX_REWARD or t > MAX_STEPS_PER_IMAGE:
                    episodes_duration.append(t + 1)
                    print('-------------------------------------------------')
                    print(f'Episode {episode+1}/{total_episodes} \t duration:{t} \t\t Last Reward: {reward}')
                    print(f'Action LEFT taken:{actions_taken["LEFT"]}')
                    print(f'Action RIGHT taken:{actions_taken["RIGHT"]}')
                    print(f'Action NONE taken:{actions_taken["NONE"]}')
                    print('-------------------------------------------------')

                    # Reset the action count 
                    for act in self.actions:
                        actions_taken[act]=0

                    break
            
            # Update the policy after batch size steps
            if episode > 0 and episode % self.hparams["batch_size"] == 0:
                running_add = 0
                for i in reversed(range(steps)):
                    running_add = running_add * self.hparams["gamma"] + reward_pool[i]
                    reward_pool[i] = running_add

                # Normalize reward
                reward_mean = np.mean(reward_pool)
                reward_std = np.std(reward_pool)

                self.writer.add_scalar('Training reward mean', reward_mean, episode)
                self.writer.add_scalar('Training reward std', reward_std, episode)

                for i in range(steps):
                    reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

                # Gradient Descent
                optimizer.zero_grad()

                print('Calculating loss...')

                for i in range(steps):
                    state = state_pool[i]
                    action = Variable(torch.Tensor([action_pool[i]]))
                    reward = reward_pool[i]

                    probs = self.forward(torch.unsqueeze(state, 0))
                    probs = probs.squeeze()
                    m = Categorical(probs)
                    loss = -m.log_prob(action) * reward  # Negative score function x reward
                    loss.backward()

                optimizer.step()

                state_pool = []
                action_pool = []
                reward_pool = []
                steps = 0
                

                # Test the network
                test_rewards = self.test_model(df_test)
                self.writer.add_scalar(
                    'Testing reward mean',
                    np.mean(test_rewards),
                    episode
                )
                self.writer.add_scalar(
                    'Testing reward std',
                    np.std(test_rewards),
                    episode
                )

    def predict_image(
            self,
            image,
            initial_point_of_view = random.random(),
            use_action_break = True,
            max_actions = 10
        ):
        """
        Gets the file name of the image and returns the predicted points
        and actions till we reach a number of tries or the action is NONE 
        which would mean that we're in the correct spot
        """
        with torch.no_grad():
            _, img_width, img_height = image.shape
            img = self.transforms(image)
            img = self.pretrained_model(torch.unsqueeze(img, 0))
            img = img.squeeze()

            # Calculate the initial point of view
            point_of_view = torch.Tensor([initial_point_of_view])
            state = torch.concat((img, point_of_view))

            actions = []
            points = [round(point_of_view.item() * img_width)]

            for _ in range(max_actions):
                probs = self.forward(torch.unsqueeze(state, 0))
                action = self.select_action(probs.squeeze())

                if use_action_break and self.actions[action] == 'NONE':
                    break

                point_of_view = round(point_of_view.item() * img_width)
                next_point_of_view = self.calculate_next_movement(
                    img_width,
                    point_of_view, 
                    self.actions[action]
                )
                point_of_view = torch.Tensor([next_point_of_view / img_width])

                actions.append(self.actions[action])
                points.append(round(point_of_view.item()*img_width))

                state = torch.concat((img, point_of_view))

        return actions, points


    def test_model(self,df):
        with torch.no_grad():
            rewards = []
            print(f'Init testing on test dataframe with {len(df)} images')
            for i in range(len(df)):
                image, real_x, _ = self.get_row(df,i)
                _, img_width, _ = image.shape
                actions, points = self.predict_image(image)
                last_reward = self.calculate_reward(img_width, real_x, points[-1])
                rewards.append(last_reward)
                print(f'Reward for image number {i} \t\t -> \t{last_reward}')

            return rewards


    def eval_model(self,df,images_dir='./data/model_test_images/reward_10'):
        """
        Uses a validation dataframe to test the model after training.
        Saves the images with the prediction information into the images_dir param
        """
        num_images = len(df)
        for row in range(num_images):
            image, real_x, _ = self.get_row(df, row)
            img_height, img_width, _ = image.shape
            image_y_position = round(img_height/2)
            actions, points = self.predict_image(image, 0.5)

            # Print the real and predicted point in the image
            # and text with the real point and the predicted point
            image = cv2.circle(
                image,
                (points[-1],image_y_position),
                radius=4,
                color=(0, 255, 255),
                thickness=5
            )
            image = cv2.circle(
                image,
                (real_x,image_y_position),
                radius=4,
                color=(0, 0, 255),
                thickness=5
            )
            image = cv2.putText(
                image,
                "Predicted",
                (points[-1],image_y_position+30),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            image = cv2.putText(
                image,
                f"Predicted: {points[-1]}",
                (20,img_height-50),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            image = cv2.putText(
                image,
                f"Real: {real_x}",
                (20,img_height-20),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            # Save the image into the directory
            filename = os.path.join(
                images_dir,
                f'{row}_real_{real_x}_predicted_{points[-1]}.jpg'
            )
            cv2.imwrite(filename,image)

            print(f'{row}/{num_images}\t Predicted:{points[-1]}\tReal:{real_x}\tNum of actions:{len(actions)}\t')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=-1)
        return x