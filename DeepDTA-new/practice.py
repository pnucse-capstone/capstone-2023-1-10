from dataset import Dataset
import numpy as np

fp = 'C:/Users/hanul/OneDrive/바탕 화면/deepdta_new/kiba/'

data = Dataset(fp)
label_ligand, label_protein, affinity = data.parse_data()
label_ligand, label_protein, affinity = np.array(label_ligand), np.array(label_protein), np.array(affinity)
label_row_inds, label_col_inds = np.where(np.isnan(affinity)==False)


print(len(label_col_inds))