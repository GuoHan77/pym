import  os
import json
import torch
from torch import nn
import  pandas as pd
import multiprocessing
from d2l import torch as d2l
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
device_count = torch.cuda.device_count()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
device_count = torch.cuda.device_count()
print(f'Available GPUs: {device_count}')
class MyDataset(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, index_col=0).values
        self.features = data[:,:-1].astype(int)
        self.labels = data[:, -1].astype(int)
        self.n_features = self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        labels = self.labels[index]
        return torch.tensor(features > 0, dtype=torch.int), torch.tensor(labels > 0, dtype=torch.float)
