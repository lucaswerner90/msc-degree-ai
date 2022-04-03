#%%
from distutils.util import strtobool
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataloader import DroneImages
from model.policy_gradient_lightning import PolicyGradientLightning

#%%
# Read the dataframe and split into train/test/validation
dataset = DroneImages("./data/dataframe.csv")
train_length = int(len(dataset) * 0.8)
train, validation = random_split(
    dataset,
    [train_length, len(dataset) - train_length]
)
train_loader = DataLoader(train, batch_size=32, num_workers=4)
val_loader = DataLoader(validation, batch_size=32, num_workers=4)

actions = ["LEFT", "RIGHT", "NONE"]
model = PolicyGradientLightning(actions)

logger = TensorBoardLogger("runs", name="policy_gradient_lightning")
checkpoint_callback = ModelCheckpoint(dirpath="./model/checkpoints/policy_gradient", save_top_k=2, monitor="val_loss")
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3)

trainer = pl.Trainer(
    gpus=0,
    min_epochs=10,
    max_epochs=100,
    logger=logger,
    log_every_n_steps=5,
    callbacks=[checkpoint_callback, early_stopping]
)
trainer.fit(model, train_loader, val_loader)

# Usar una sola acción por imagen => en vez de ejecutar una accion y mover el punto central, ejecutar solamente esa acción por cada imagen
# Empezar a probar con diferentes algoritmos => probar con el actor critic

# %%
