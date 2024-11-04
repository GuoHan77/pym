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
def load_pretrained_model(data_dir):
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 603, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    bert = BERTModel(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                ffn_num_hiddens, num_heads, num_layers, dropout).to(device)
    bert.load_state_dict(torch.load(os.path.join(data_dir,"AML_model.pth")))
    return bert
def train_batch_ch13(model, criterion , optimizer, device,dataloader_train):
    model.train()
    total_loss_train = 0
    total_correct_train = 0
    total_samples_train = 0
    for i,data in enumerate(dataloader_train):
        optimizer.zero_grad()
        features, labels = data
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits.view(-1), labels)
        loss.backward()
        optimizer.step()
        total_loss_train += loss.item()
        total_samples_train += labels.size(0)
        predicted = (logits > 0.5).view(-1).float()
        total_correct_train += (predicted == labels).sum().item()
   
    loss_train = total_loss_train / len(dataloader_train)
    return  loss_train

def train_ch13(model, train_iter, test_iter, criterion , optimizer, num_epochs,device,plot=False):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],legend=['train loss', 'test auc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        train_loss=train_batch_ch13(model, criterion , optimizer, device,train_iter)
        model.eval()
        total_correct_val = 0
        total_samples_val = 0
        total_loss_val = 0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for inputs, targets in test_iter:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), targets)
                total_loss_val += loss.item()
                total_samples_val += targets.size(0)
                output=F.sigmoid(outputs)
                predicted = (output > 0.5).view(-1).float()
                total_correct_val += (predicted == targets).sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        accuracy_val = total_correct_val / total_samples_val
        auc_val = roc_auc_score(all_targets, all_outputs)
        print(f"train loss: {train_loss:.4f}  test acc: {accuracy_val:.2f}  test auc: {auc_val:.2f}")
model = load_pretrained_model("模型参数")
my_dataset = MyDataset("Blusom/AML_Adrenal Gland.csv")
targets = [int(data[1]) for data in train_dataset]
class_counts = torch.bincount(torch.tensor(targets))
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[torch.tensor(targets)]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(my_dataset)):
    
    print(f"Fold {fold + 1}")
    train_subset = torch.utils.data.Subset(my_dataset, train_idx)
    val_subset = torch.utils.data.Subset(my_dataset, val_idx)

    batch_size = 32
    dataloader_train = DataLoader(train_subset, batch_size=batch_size,sampler=sampler)
    dataloader_val = DataLoader(val_subset, batch_size=batch_size)

    num_epochs = 5
    loss = nn.BCEWithLogitsLoss()
    trainer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_ch13(model, dataloader_train, dataloader_val, loss, trainer, num_epochs, device)
