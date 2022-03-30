#%%
import torch
import pandas as pd
from model.policy_gradient_agent import PolicyNet
from sklearn.model_selection import train_test_split
# %%
# Read the dataframe and split into train/test/validation
dataframe = pd.read_csv('./data/dataframe.csv')

df_train, df_validation = train_test_split(
    dataframe, 
    test_size=0.1,
    random_state=42,
    shuffle=True
)

train, test = train_test_split(
    df_train,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

#%%
hparams = dict(
    learning_rate = 1e-4,
    batch_size = 16,
    gamma = 0.99
)

actions = ["LEFT", "RIGHT", "NONE"]
agent = PolicyNet(actions, hparams)

save_file = 'policy_gradient_reward_10.pth'
agent.train_model(train, test)
torch.save(agent.state_dict(), save_file)
agent.eval_model(df_validation)