from model import DeepDTA
from trainer import Trainer
import pandas as pd

# this CSV file has 4 columns, protein, ligands, affinity, split.

fp = 'C:\Users\hanul\Downloads\DeepDTA-new (1)\DeepDTA-new\davis'

model = DeepDTA
channel = 10
protein_kernel = [8, 12]
ligand_kernel = [4, 8]

for prk in protein_kernel:
    for ldk in ligand_kernel:
        # epoch 50 is enough for convergence in this case, but may need more for other datasets
        trainer = Trainer(fp, model, channel, prk, ldk, "training_logs-prk{}-ldk{}.log".format(prk, ldk))
        trainer.train_kfold(num_epochs=30, batch_size=128, lr=0.001, save_path='training_result-prk{}-ldk{}.pt'.format(prk, ldk))
