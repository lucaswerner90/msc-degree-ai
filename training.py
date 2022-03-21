import pandas as pd
from model.agent import PolicyNet
from sklearn.model_selection import train_test_split



if __name__ == "__main__":
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

    hparams = dict(
        learning_rate = 1e-4,
        train_epochs = len(train),
        batch_size = 16,
        gamma = 0.99
    )
    actions = ["LEFT", "RIGHT", "NONE"]
    agent = PolicyNet(actions, hparams)
    agent.train_model(train)
