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
class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 ):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size,
                                   query_size=query_size, value_size=value_size)
        self.output_layer1 = nn.Linear(768, 1, bias=False)
        self.mlp=DNN()
    def forward(self, tokens):
        encoded_X = self.encoder(tokens)
        x=self.output_layer1(encoded_X)
        x=x.view(x.size(0), -1)
        mlp_y=self.mlp(x)
        return mlp_y
