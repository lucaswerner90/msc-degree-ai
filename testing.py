#%%
import pandas as pd
from model.agent import PolicyNet
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torch

dataframe = pd.read_csv('./data/dataframe.csv')

_, df_validation = train_test_split(
    dataframe, 
    test_size=0.1,
    random_state=42,
    shuffle=True
)

hparams = dict(
    learning_rate = 1e-4,
    batch_size = 16,
    gamma = 0.99
)

actions = ["LEFT", "RIGHT", "NONE"]

writer = SummaryWriter(
	log_dir='runs',
	comment='PG_basic_reward_10'
)
agent = PolicyNet(
	actions,
	hparams,
	writer
)

# Load the model and test the agent
agent.load_state_dict(
	torch.load('policy_gradient_agent_reward_10.pth')
)

#%%
agent.eval_model(df_validation)


# %%
