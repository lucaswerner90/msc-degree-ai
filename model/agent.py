import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision import transforms
class PolicyNet(nn.Module):
    """
    Policy gradient algorithm
    """
    def __init__(self, num_actions):
        super(PolicyNet, self).__init__()
        
        # Load the pretrained model that will preprocess the image
        # before moving it to the agent
        pretrained_model = models.vgg19(pretrained=True)
        pretrained_model.classifier = nn.Sequential(
            *list(pretrained_model.classifier.children())[:-2]
        )
        for params in pretrained_model.parameters():
            params.requires_grad = False

        self.pretrained_model = pretrained_model
        
        # Initialize the transforms pipeline
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Define the model structure
        self.fc1 = nn.Linear(4096+1, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_actions)


    def select_action(self,probs):
        pass

    def train(self,df,hparams):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=hparams["learning_rate"])
        pass

    def eval_model(self,):
        self.eval()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=-1)
        return x