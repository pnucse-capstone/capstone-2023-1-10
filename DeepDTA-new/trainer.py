from dataset import Datahelper, DTAdataset
import pandas as pd
import json
import torch
import torch.nn as nn
import logging
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from lifelines.utils import concordance_index


def collate_fn(batch):
    """
    Collate function for the DataLoader
    """
    proteins, ligands, targets = zip(*batch)
    proteins = torch.stack(proteins, dim=0)
    ligands = torch.stack(ligands, dim=0)
    targets = torch.stack(targets, dim=0)
    return proteins, ligands, targets

class Trainer:
    def __init__(self, fpath,  model, channel, protein_kernel, ligand_kernel,log_file, smilen=100, seqlen=1000):
        
        #Datahelper와 DTAdataset을 통해 머신러닝이 가능한 형태로 바꿔준다(flattening & torch화)
        self.datahelper = Datahelper(fpath, seqlen, smilen)
        self.xD, self.xT, self.xY = self.datahelper.data_to_tensor()
        self.dataset = DTAdataset(self.xD, self.xT, self.xY)

        self.protein_kernel = protein_kernel
        self.ligand_kernel = ligand_kernel

        self.smilen = smilen
        self.seqlen = seqlen
        
        self.protein_vocab_len , self.ligand_vocab_len = self.datahelper.get_vocab_length()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model(self.protein_vocab_len, self.ligand_vocab_len, channel, protein_kernel, ligand_kernel).to(self.device)

        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)


    def train_kfold(self,lr, num_epochs, batch_size, save_path):  
        
        writer = SummaryWriter()

        #dataset을 train/test로 나누기
        X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.xY, test_size=0.2, random_state=42)


        #K-fold cross validation 
        k_folds = 6
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Initialize a list to store the evaluation results for each fold
        validation_losses = []
        test_losses_total = []
        #optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        for fold, (train_indices, val_indices) in enumerate(kf.split(X_train)):
            print(f"==================== Fold {fold + 1}/{k_folds} ====================")

            # Subset the training and validation data for this fold
            train_data = Subset(X_train, train_indices)
            val_data = Subset(X_train, val_indices)


            train_loader = DataLoader(train_data, batch_size=batch_size, drop_last = False, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_data, batch_size=batch_size, drop_last = False, collate_fn=collate_fn)
            test_loader = DataLoader(X_test, batch_size= batch_size, drop_last = False, collate_fn=collate_fn)

            # Training loop for this epoch
            for epoch in range(num_epochs):
                print("<<<<<<< epoch num: ",epoch," >>>>>>>>>>")
                self.model.train()
                train_loss = 0.0
                
                with tqdm(total=len(train_loader)) as pbar:

                    for protein, ligand, labels in train_loader:

                        protein, ligand, target = protein.to(self.device), ligand.to(self.device), labels.to(self.device)

                        optimizer.zero_grad()
                        # Forward pass
                        output = self.model(protein, ligand)
                        
                        # Compute the loss
                        loss = criterion(output, target)

                        # Backpropagation and optimization
                        
                        loss.backward()
                        print(loss.item())
                        optimizer.step()
                        train_loss += loss.item()

                        pbar.update(1)
                    
                train_loss /= len(train_loader)
                self.logger.info('Epoch: {} - Training Loss: {:.6f}'.format(epoch+1, train_loss))
                writer.add_scalar('train_loss', train_loss, epoch)

                # Validation loop for this epoch
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    for protein, ligand, target in val_loader:
                        protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                        output = self.model(protein, ligand)
                        loss = criterion(output, target)
                        val_losses.append(loss.item())

                epoch_val_loss = np.mean(val_losses)
                validation_losses.append(epoch_val_loss)

            # CI scores
            predicted_values = []
            true_labels = []

            # Evaluate the model on the test set
            test_losses = []
            with torch.no_grad():
                for protein, ligand, target in test_loader:
                    protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                    output = self.model(protein, ligand)

                    test_loss = criterion(output, target)
                    test_losses.append(test_loss.item())

                    predicted_values.extend(output.to(self.device).numpy())
                    true_labels.extend(target.numpy())


            # Calculate and print the average test loss
            avg_test_loss = np.mean(test_losses)
            test_losses_total.append(avg_test_loss)
            print(f"Average Test(MSE) Loss for fold {fold+1}: {avg_test_loss}")

            c_index = concordance_index(true_labels, -np.array(predicted_values))
            print(f"C-Index for fold {fold+1}: {c_index}")

            avg_validation_loss = np.mean(validation_losses)
            print(f"Average Validation Loss for fold {fold+1}: {avg_validation_loss}")

        