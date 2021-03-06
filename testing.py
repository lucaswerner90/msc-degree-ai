#%%
import torch
import pandas as pd
from model.policy_gradient_agent import PolicyNet
from sklearn.model_selection import train_test_split
# %%
# Read the dataframe and split into train/test/validation
dataframe = pd.read_csv('./data/dataframe.csv')

_, df_validation = train_test_split(
    dataframe, 
    test_size=0.1,
    random_state=42,
    shuffle=True
)

#%%
hparams = dict(
    learning_rate = 1e-5,
    batch_size = 16,
    gamma = 0.95,
    epochs=50
)

actions = ["LEFT", "RIGHT", "NONE"]
agent = PolicyNet(actions, hparams, experiment_name="policy_gradient_50epochs_lr_1e-5_batch_16_gamma_95")
agent.load_state_dict(torch.load('policy_gradient_reward_50_steps_per_image_v2.pth'))
agent.eval_model(df_validation)