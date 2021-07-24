""" aka_cnn.py - This code defines a CNN to be used in the MLHEP competitions. It is based on
the baseline classifier, but some changes will be made to hopefully make it perform better!

Authors: Kevin Greif
Adapted from model.py at https://github.com/yandexdataschool/mlhep-2021-baseline
7/23/21
python3
"""

import pytorch_lightning as pl
import torch
import torchmetrics as tm
from torch import nn
from torch.nn import functional as F


class AkaCnn(pl.LightningModule):
    """ This class defines a CNN. It will have 2 modes, regression and classification.
    Classification is for challenge 1 and regression is for challenge 2.
    """
    
    def __init__(self, mode: ["classification", "regression"] = "classification", lr=1e-4):
        
        super().__init__()
        self.mode = mode
        self.learning_rate = lr
        
        # Convoluational layers
        self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(8),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(19, stride=6),
                    nn.Flatten()
                    )

        # Fully connected layers
        self.drop_out1 = nn.Dropout()
        self.fc1 = nn.Linear(4624, 500)

        # Define stem network, common to both tasks
        self.stem = nn.Sequential(
                self.layer1, self.drop_out1, self.fc1,
                )
        
        # Now define different final layers, one for each task
        self.drop_out2 = nn.Dropout()
        self.fc2 = nn.Linear(500, 2)  # for classification
        self.fc3 = nn.Linear(500, 1)  # for regression
        
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem, self.drop_out2, self.fc2)
        else:
            self.regression = nn.Sequential(self.stem, self.drop_out2, self.fc3)

        # Initialize some metrics
        self.train_acc = tm.Accuracy()
        self.valid_acc = tm.Accuracy()
        
        # Now initialize random noise picture for making graph
        self.example_input_array = torch.rand((1, 1, 120, 120))

        
    def training_step(self, batch, batch_idx):
        """ training_step - Training steps differ slightly between tasks,
        so implement custom training step for lightening.
        """
        
        # Pull images and targets from batch
        x_target, class_target, reg_target, _ = batch
        
        # Classification training
        if self.mode == "classification":
            
            # Forward pass
            class_pred = self.classification(x_target.float())
            
            # Calculate loss
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            
            # Find correct classifications
            classes = torch.sigmoid(class_pred).argmax(dim=1)
            correct = (classes == class_target[:,1]).sum()
            
            # Build batch dictionary
            batch_dict = {
                'loss': class_loss,
                'correct': correct,
                'total': len(classes)
            }

            return batch_dict

        # Regression training
        else:
            
            # Predict energies
            reg_pred = self.regression(x_target)
            
            # Calculate loss (what is the ideal loss here? Regularization?)
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))
            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            
            # Build batch dictionary again, this time we're only interested in training loss.
            batch_dict = {'loss': reg_loss}
            
            return batch_dict

    def validation_step(self, batch, batch_idx):
        """ validation_step - Here we'll evaluate model over validation set and log metrics.
        """
        
        # Pull data from batch
        x_target, class_target, reg_target, _ = batch
        
        # Classification validation
        if self.mode == "classification":
            
            # Forward pass
            class_pred = self.classification(x_target.float())
            
            # Calculate loss
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            
            # Calculate accuracy
            self.valid_acc(torch.sigmoid(class_pred), class_target)
            
            # Log loss and accuracy
            self.log("valid_acc", self.valid_acc.compute())
            self.log("classification_loss", class_loss)
            
            return class_loss

        # Regression validation
        else:
            
            # Forward pass
            reg_pred = self.regression(x_target.float())
            
            # Calculate loss
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))
            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            
            # Log loss
            self.log("valid_loss", reg_loss, prog_bar=True)
            
            return reg_loss

    def configure_optimizers(self):
        """ configure_optimizers - Simply initialize adam optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def custom_histogram_adder(self):
        """ custom_histogram_adder - Just adds histograms of all model parameters to tensorboard.
        """
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
            
    def training_epoch_end(self, outputs):
        """ training_epoch_end - This runs after every epoch, only to save histograms of weights
        and log some metrics. In classification mode, train loss, and train_acc. In regression
        mode, just train loss.
        """
        
        # Save histograms of weights after each epoch
        self.custom_histogram_adder()
        
        # Calculate average train loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        # Log train loss
        self.logger.experiment.add_scalar("train_loss",
                                            avg_loss,
                                            self.current_epoch)

        # Only for classification
        if self.mode == 'classification':
            
            # Calculate train ACC
            correct = sum([x["correct"] for  x in outputs])
            total = sum([x["total"] for  x in outputs])
            train_acc = correct / total
            
            self.logger.experiment.add_scalar("train_acc",
                                                train_acc,
                                                self.current_epoch)

    def forward(self, x):
        """ forward - Custom forward call required for saving graph at end of 0th epoch.
        How to fix this??
        """
        
        if self.mode == "classification":
            class_pred = self.classification(x)
            return {"class": torch.sigmoid(class_pred)}
        else:
            reg_pred = self.regression(x)
            return reg_pred
