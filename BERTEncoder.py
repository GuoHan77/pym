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

class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens).to(device)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
               num_hiddens,ffn_num_hiddens, num_heads, dropout, True))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens)).to(device)

    def forward(self, tokens, valid_lens=None):
        X = self.token_embedding(tokens)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
