import configparser
import pathlib as path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from idao.data_module import IDAODataModule
from idao.aka_cnn import AkaCnn

# Define some custom callbacks for use in training

class HistCallback(Callback):
    """ custom callback for making histograms of model weights before training.
    """
    def on_pretrain_routine_end(self, trainer, pl_module):
        pl_module.custom_histogram_adder()

########################## PTL Trainer setup #######################


def trainer(logger, mode: ["classification", "regression"], cfg, dataset_dm):
    """ trainer - This function builds model and executes training
    """
    
    # Build model
    model = AkaCnn(mode=mode)
    
    # Set number of epochs
    if mode == "classification":
        epochs = cfg["TRAINING"]["ClassificationEpochs"]
    else:
        epochs = cfg["TRAINING"]["RegressionEpochs"]
        
    # Build dataloaders
    train = dataset_dm.train_dataloader()
    valid = dataset_dm.val_dataloader()
    
    # Build callbacks
    check_dir = "./checkpoints/" + mode + "/" + logger.name + "/version_" + str(logger.version)
    print("Will save checkpoints at ", check_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=check_dir,
                                          monitor='valid_loss',
                                          mode='min',
                                          save_top_k=5)
    hist_callback = HistCallback()
        
    # Build pytorch lightening trainer
    trainer = pl.Trainer(
        callbacks=[hist_callback, checkpoint_callback],
        gpus=int(cfg["TRAINING"]["NumGPUs"]),
        max_epochs=int(epochs),
        progress_bar_refresh_rate=1,
        logger=logger
    )

    # Train the model ⚡
    trainer.fit(model, train, valid)


def main():
    config = configparser.ConfigParser()
    config.read("./config.ini")

    PATH = path.Path(config["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=int(config["TRAINING"]["BatchSize"]), cfg=config
    )
    dataset_dm.prepare_data()
    dataset_dm.setup()
    
    logger = TensorBoardLogger('runs', 'AkaCnn-reg1', log_graph=True)
    
    # trainer(logger, "classification", cfg=config, dataset_dm=dataset_dm)
    trainer(logger, "regression", cfg=config, dataset_dm=dataset_dm)


if __name__ == "__main__":
    main()
