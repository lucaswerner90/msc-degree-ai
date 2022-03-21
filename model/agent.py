from itertools import count
import pandas as pd
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

np.random.seed(42)

class PolicyNet(nn.Module):
    """
    PolicyNet class contains our Policy Gradient agent implementation
    that will run the training and the testing over our dataframe
    """
    def __init__(self, 
    actions, 
    hparams,
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
    writer = SummaryWriter(), 
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
        return 1 if num_sections_away == 0 else 1 - (num_sections_away / num_sections)

    def select_action(self, probs: torch.Tensor) -> int:
        """
        Given the output of the model, we select the action
        based on a probability distribution and we return the
        position of the action in the actions array
        """
        m = Categorical(probs)
        return m.sample().item()

    def train_model(self, df):
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
        total_episodes = self.hparams["train_epochs"]

        for episode in range(total_episodes):
            row = df.iloc[episode]
            filename, real_x, real_y = (
                row["filename"],
                row["prediction_x"],
                row["prediction_y"],
            )

            # Read and transform the original image to be able to fit it
            # into our model correctly
            original_image = cv2.imread(filename)
            _, img_width, img_height = original_image.shape
            img = self.transforms(original_image)
            img = self.pretrained_model(torch.unsqueeze(img, 0))
            img = img.squeeze()

            # Calculate the initial point of view
            point_of_view = torch.Tensor([0.5])  # img_width/2

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
                if reward == 1 or t > MAX_STEPS_PER_IMAGE:
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

                self.writer.add_scalar('Reward mean', reward_mean, episode)
                self.writer.add_scalar('Reward std', reward_std, episode)

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
                
                # Save the model
                print('Saving the model...')
                torch.save(self.state_dict(), "policy_gradient_agent.pth")

    def eval_model(self,):
        self.eval()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=-1)
        return x