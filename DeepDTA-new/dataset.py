import json
import numpy as np
import pickle
import math
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, Subset, DataLoader


class DTAdataset(Dataset):
    def __init__(self, ligand_data, protein_data, affinities):
        self.ligand_data = ligand_data
        self.protein_data = protein_data
        self.affiniy_data = affinities

    def __len__(self):
        return len(self.protein_data)

    def __getitem__(self, index):
        d1 =  torch.tensor(self.ligand_data[index])
        d2 =  torch.tensor(self.protein_data[index])
        d3 =  torch.tensor(self.affiniy_data[index], dtype=torch.float)
        return d2,d1,d3


class Datahelper:
    def __init__(self, fpath, seqlen=1000, smilen=100):

        #파일 위치
        self.fpath = fpath

        #ligand(smile)/protein(sequence)의 최대 길이
        self.smilelen = smilen
        self.seqlen = seqlen

        #protein/ligand의 단어 종류 집합이랑 encoding dict
        self.protein_vocab = set()
        self.ligand_vocab = set()
        self.protein_dict = {}
        self.ligand_dict = {}

    def encoding(self, line, MAX_LEN, dict): #ligand/protein 한개에 대한 encoding

        #padding
        X = np.zeros(MAX_LEN)

        #cutting & encoding
        for index, char in enumerate(line[:MAX_LEN]):
            X[index] = dict[char]
        
        return X
	  
    def parse_data(self, with_label=True):  #데이터셋 폴더로 부터 D/T label, Affinity값 받아오기
        
        #데이터 폴더 읽기 시작
        print("Read %s start" % self.fpath)


        ligands = json.load(open(self.fpath+"ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(self.fpath+"proteins.txt"), object_pairs_hook=OrderedDict)

        Y = pickle.load(open(self.fpath + "Y","rb"), encoding='latin1') ### TODO: read from raw
        Y = -(np.log10(Y/(math.pow(10,9))))

        XD = []
        XT = []

        #Drug(ligand), Target(Protein) vocab 만들기
        if with_label:
            for d in ligands.keys():
                for letter in ligands[d]:
                    self.ligand_vocab.update(letter)

            for t in proteins.keys():
                for letter in proteins[t]:
                    self.protein_vocab.update(letter)

        #ligand, protein dict 만들기
        self.protein_dict = {x: i+1 for i, x in enumerate(self.protein_vocab)}
        self.ligand_dict = {x: i+1 for i, x in enumerate(self.ligand_vocab)}
                
        #dict를 토대로 encoding 해서 리스트 XD, XT, Y 만들기
        for d in ligands.keys():
            XD.append(self.encoding(ligands[d], self.smilelen, self.ligand_dict))
        for t in proteins.keys():
            XT.append(self.encoding(proteins[t], self.seqlen, self.protein_dict))

        return XD, XT, Y
    
    def data_to_tensor(self):    #데이터셋을 tensor형식으로 바꿔주기 + 데이터셋에서 nanvalue 없애기

        XD, XT, xY = self.parse_data()
        int_XD = [array.astype(int) for array in XD]
        int_XT = [array.astype(int) for array in XT]
        xD = np.array(int_XD)
        xT = np.array(int_XT)
        xY = np.array(xY)

        #데이터셋에서 nan이 아닌 값들의 index구하기
        label_row_inds, label_col_inds = np.where(np.isnan(xY)==False)

        protein_data = []
        ligand_data = []
        affinity = []
        
        #구한 index값을 바탕으로 kiba데이터셋을 tensorDataset으로 바꾸기
        for i in label_row_inds:
            ligand_data.append(xD[i])
    
        for j in label_col_inds:
            protein_data.append(xT[j])
    
        for i in range(len(label_col_inds)):
            affinity.append(xY[label_row_inds[i]][label_col_inds[i]])

        return ligand_data, protein_data, affinity

    def get_vocab_length(self):
        return len(self.protein_vocab)+1, len(self.ligand_vocab)+1
