#%%
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataloader import DroneImages
from model.policy_gradient_lightning import PolicyGradientLightning
if __name__ == '__main__':
    # Read the dataframe and split into train/test/validation
    dataset = DroneImages("./data/dataframe.csv")
    train_length = int(len(dataset) * 0.8)
    train, validation = random_split(
        dataset,
        [train_length, len(dataset) - train_length]
    )
    train_loader = DataLoader(train, batch_size=16)
    val_loader = DataLoader(validation, batch_size=16)

    actions = ["LEFT", "RIGHT", "NONE"]
    model = PolicyGradientLightning(actions)

    logger = TensorBoardLogger("runs", name="policy_gradient_lightning")
    checkpoint_callback = ModelCheckpoint(dirpath="./model/checkpoints/policy_gradient", save_top_k=2, monitor="validation_reward_mean")
    # early_stopping = EarlyStopping(monitor="validation_reward_mean", mode="min", patience=3)

    trainer = pl.Trainer(
        min_epochs=10,
        max_epochs=100,
        logger=logger,
        auto_lr_find=False,
        auto_scale_batch_size=False,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

# Usar una sola acción por imagen => en vez de ejecutar una accion y mover el punto central, ejecutar solamente esa acción por cada imagen
# Empezar a probar con diferentes algoritmos => probar con el actor critic

# %%
