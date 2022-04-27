from torch import nn
from model.generic_actor_critic import GenericActorCriticAgent
class ActorCritic(GenericActorCriticAgent):
    """
    Basic implementation of the actor critic agent model
    """
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.common_model= nn.Sequential(
            nn.Linear(4096+1, 1024), # 4096 => salida de la ultima capa de la VGG16 + punto de vista
            nn.Linear(1024, 256),
        )
        self.actor = nn.Linear(256, len(self.actions))
        self.critic = nn.Linear(256, 1)

	