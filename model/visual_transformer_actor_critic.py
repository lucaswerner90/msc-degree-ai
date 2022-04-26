"""
Defines the actor-critic model according to the needs of using a Vision Transformer 
for the encoder part.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class VisualEncoderActorCritic(nn.Module):
    def __init__(self):
        super(VisualEncoderActorCritic, self).__init__()
        self.actions = ["LEFT","RIGHT","NONE"]
        self.stop_action = "NONE"

        self.common_model = nn.Sequential(
            nn.Linear(197*768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )

        # actor's layer
        self.actor = nn.Sequential(
            nn.Linear(256+1, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.actions))
        )

        # critic's layer
        self.critic =nn.Sequential(
            nn.Linear(256+1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def select_action(self, logits):
        prob = F.softmax(logits, -1)
        # create a categorical distribution over the list of probabilities of actions
        action = prob.multinomial(num_samples=1)
        log_prob = F.log_softmax(logits, -1)
        log_prob = log_prob.gather(0, action)
        return log_prob, action

    def forward(self, image, point):
        x = self.common_model(image).squeeze()
        x = torch.cat((x, point))
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        policy = self.actor(x)

        # critic: evaluates being in the state s_t
        state_value = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return policy, state_value

	