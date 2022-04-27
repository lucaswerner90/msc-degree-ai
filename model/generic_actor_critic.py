import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericActorCriticAgent(nn.Module):
    def __init__(self, common_model=None, actor=None, critic=None):
        super(GenericActorCriticAgent, self).__init__()
        self.actions = ["LEFT","RIGHT","NONE"]
        self.stop_action = "NONE"

        self.common_model = common_model

        # actor's layer
        self.actor = actor

        # critic's layer
        self.critic = critic

    def select_action(self, logits):
        prob = F.softmax(logits, -1)
        # create a categorical distribution over the list of probabilities of actions
        action = prob.multinomial(num_samples=1)
        log_prob = F.log_softmax(logits, -1)
        log_prob = log_prob.gather(0, action)
        return log_prob, action

    def forward(self, image, point):
        if image.shape[0] == 1:
            image = image.squeeze()
        x = torch.cat((image, point))
        x = self.common_model(x).squeeze()

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        policy = self.actor(x)

        # critic: evaluates being in the state s_t
        state_value = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return policy, state_value

	