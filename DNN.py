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
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.dropout=nn.Dropout(0.5)
        self.fc1 = nn.Linear(603, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 1)
    def forward(self, x):
        x = F.gelu (self.dropout((self.fc1(x))))
        x = F.gelu(self.dropout((self.fc2(x))))
        x =self.fc3(x)
        return x
