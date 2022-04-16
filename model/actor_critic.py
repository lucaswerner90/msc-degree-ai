from torch import nn
import torch.nn.functional as F


ACTIONS = ["LEFT","RIGHT","NONE"]

class ActorCritic(nn.Module):
    """
    Implements both actor and critic in one model
    """
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.common_model = nn.Sequential(
            nn.Linear(4096+1, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Softmax(dim=-1)
        )

        # actor's layer
        self.actor = nn.Linear(64, len(ACTIONS))

        # critic's layer
        self.critic = nn.Linear(64, 1)


    def forward(self, x):
        x = F.relu(self.common_model(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        policy = F.softmax(self.actor(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_value = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return policy, state_value

	