import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch_geometric.nn import MessagePassing, RGCNConv, GATConv
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import math

def get_accuracy(probs, labels):
    probs = torch.argmax(probs, dim = 1)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(probs)):
        if probs[i] == 1 and labels[i] == 1:
            TP += 1
        if probs[i] == 0 and labels[i] == 0:
            TN += 1
        if probs[i] == 1 and labels[i] == 0:
            FP += 1
        if probs[i] == 0 and labels[i] == 1:
            FN += 1
    accuracy = (TP + TN) / (TP+TN+FP+FN)
    try:
        precision = TP / (TP + FP)
    except: 
        precision = TP / (TP + FP + 1)
    try:
        recall = TP / (TP + FN)
    except:
        recall = TP / (TP + FN + 1)
    try:
        F1 = (2 * precision * recall) / (precision + recall)
    except:
        F1 = (2 * precision * recall) / (precision + recall + 1)
    try:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP + 1) * (TP + FN + 1) * (TN + FP + 1) * (TN + FN + 1))
    return {'accuracy': accuracy, 'F1': F1, 'MCC': MCC}

class TwiBot20Dataset(Dataset):
    def __init__(self, node_feature_set, batch_size, label_user_count, name): # name = train/dev
        path = '/new_temp/fsb/fsb_asonam/'
        self.P1_features = torch.load(path + 'user_features1.pt').squeeze(1).float()
        self.P1_features = (self.P1_features - self.P1_features.mean()) / sqrt(self.P1_features.var())
        self.P2_features = torch.load(path + 'user_features4.pt').squeeze(1).float()
        self.S1_features = torch.load(path + 'user_features2.pt').squeeze(1).float()
        self.S2_features = torch.load(path + 'user_features3.pt').squeeze(1).float()
        self.S2_features = self.S2_features.squeeze(1)
        
        self.edge_index = torch.load(path + 'edge_index.pt')
        self.edge_type = torch.cat((torch.zeros(1,len(self.edge_index[0])), torch.ones(1,len(self.edge_index[0]))), dim = 1).long()
        #bidirectional edges???
        self.edge_index = torch.cat((self.edge_index, torch.stack([self.edge_index[1], self.edge_index[0]])), dim = 1)
        #print(self.edge_index.size())
        
        self.label_list = torch.load(path + 'label_list.pt')
        self.batch_size = batch_size
        self.name = name
        if self.name == 'train':
            self.length = int(label_user_count * 0.7 / self.batch_size)
        else:
            self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.P1_features, self.P2_features, self.S1_features, self.S2_features, self.edge_index, self.edge_type, self.label_list

class GATClassifier(pl.LightningModule):
    def __init__(self, in_channels, out_channels, dropout, label_user_count, batch_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_user_count = label_user_count
        self.batch_size = batch_size
        
        #self.linearP1 = nn.Linear(6, int(out_channels / 4))
        #self.linearP2 = nn.Linear(11, int(out_channels / 4))
        self.linearS1 = nn.Linear(768, int(out_channels / 2))
        self.linearS2 = nn.Linear(768, int(out_channels / 2))
        
        self.linear1 = nn.Linear(out_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        
        #self.linearP = nn.Linear(6, out_channels / 2)
        #torch.nn.init.kaiming_uniform(self.linearP1.weight, nonlinearity='leaky_relu')
        #torch.nn.init.kaiming_uniform(self.linearP2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform(self.linearS1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform(self.linearS2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform(self.linear1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform(self.linear2.weight, nonlinearity='leaky_relu')
        
        #self.GAT1 = GATConv(in_channels=out_channels, out_channels=out_channels, dropout=0 * dropout)
        #self.GAT2 = GATConv(in_channels=out_channels, out_channels=out_channels, dropout=0 * dropout)
        self.RGCN1 = RGCNConv(in_channels=out_channels, out_channels=out_channels, num_relations=2)
        #self.RGCN2 = RGCNConv(in_channels=out_channels, out_channels=out_channels, num_relations=2)
        #self.GAT3 = GATConv(in_channels=out_channels, out_channels=out_channels, dropout=0 * dropout)
        self.output = nn.Linear(out_channels, 2) #binary classification
        self.dropout_layer = nn.Dropout(dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.relu = nn.LeakyReLU()

    def forward(self, x): 
        P1 = x[0].squeeze(0)
        P2 = x[1].squeeze(0)
        S1 = x[2].squeeze(0)
        S2 = x[3].squeeze(0)
    

        edge_index = x[4].squeeze(0)
        edge_type = x[5].squeeze(0).squeeze(0)
        
        #P1 = self.dropout_layer(self.relu(self.linearP1(P1)))
        #P2 = self.dropout_layer(self.relu(self.linearP2(P2)))
        S1 = self.dropout_layer(self.relu(self.linearS1(S1)))
        S2 = self.dropout_layer(self.relu(self.linearS2(S2)))
        
        node_features = torch.cat((S1,S2),dim=1)
        #node_features = torch.cat((P1,P2),dim=1)
        #node_features = torch.cat((node_features, S1), dim=1)
        #node_features = torch.cat((node_features, S2), dim=1)
        #node_features = S1
        
        node_features = self.dropout_layer(self.relu(self.linear1(node_features)))
        
        node_features = self.dropout_layer(self.relu(self.RGCN1(node_features, edge_index, edge_type)))
        node_features = self.dropout_layer(self.relu(self.RGCN1(node_features, edge_index, edge_type)))
        
        node_features = self.dropout_layer(self.relu(self.linear2(node_features)))
        
        probs = self.output[node_features[int(0.9 * self.label_user_count) : self.label_user_count]]
        return torch.argmax(probs, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay = 1e-5)
        #optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        P1 = train_batch[0].squeeze(0)
        P2 = train_batch[1].squeeze(0)
        S1 = train_batch[2].squeeze(0)
        S2 = train_batch[3].squeeze(0)
    

        edge_index = train_batch[4].squeeze(0)
        edge_type = train_batch[5].squeeze(0).squeeze(0)
        label = train_batch[6].squeeze(0).tolist()
        label = torch.LongTensor(label[:int(0.7 * self.label_user_count)])
        
        #P1 = self.dropout_layer(self.relu(self.linearP1(P1)))
        #P2 = self.dropout_layer(self.relu(self.linearP2(P2)))
        S1 = self.dropout_layer(self.relu(self.linearS1(S1)))
        S2 = self.dropout_layer(self.relu(self.linearS2(S2)))
        
        node_features = torch.cat((S1,S2),dim=1)
        #node_features = torch.cat((P1,P2),dim=1)
        #node_features = torch.cat((node_features, S1), dim=1)
        #node_features = torch.cat((node_features, S2), dim=1)
        #node_features = S1
        
        node_features = self.dropout_layer(self.relu(self.linear1(node_features)))
        
        node_features = self.dropout_layer(self.relu(self.RGCN1(node_features, edge_index, edge_type)))
        node_features = self.dropout_layer(self.relu(self.RGCN1(node_features, edge_index, edge_type)))
        
        node_features = self.dropout_layer(self.relu(self.linear2(node_features)))
        
        #!! residual connection
        #node_features = node_features + self.relu(self.linear(train_batch[0].squeeze(0)))
        
        batch_id = torch.randperm(int(0.7 * self.label_user_count))[:self.batch_size].tolist()
        
        probs = self.output(node_features[batch_id])
        #print(probs.size())
        #print(label[batch_id].size())
        loss = self.CELoss(probs, label[batch_id].cuda())
        acc = get_accuracy(probs, label[batch_id])['accuracy']
        
        #regularization
        #lambda1 = 1e-5
        #lambda2 = 0.01
        #layers = [model.linear, model.linear2, model.linear3, model.linearP]
        #l1_regularization = 0
        #l2_regularization = 0
        #for layer in layers:
            #l1_regularization += lambda1 * torch.norm(torch.cat([x.view(-1) for x in layer.parameters()]), 1)
            #l2_regularization += lambda2 * torch.norm(torch.cat([x.view(-1) for x in layer.parameters()]), 2)
        #loss = loss + l1_regularization

        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        P1 = val_batch[0].squeeze(0)
        P2 = val_batch[1].squeeze(0)
        S1 = val_batch[2].squeeze(0)
        S2 = val_batch[3].squeeze(0)
    

        edge_index = val_batch[4].squeeze(0)
        edge_type = val_batch[5].squeeze(0).squeeze(0)
        label = val_batch[6].squeeze(0).tolist()
        label = torch.LongTensor(label[int(0.9 * self.label_user_count) : int(1 * self.label_user_count)])
        
        #P1 = self.relu(self.linearP1(P1))
        #P2 = self.relu(self.linearP2(P2))
        S1 = self.relu(self.linearS1(S1))
        S2 = self.relu(self.linearS2(S2))
        
        node_features = torch.cat((S1,S2),dim=1)
        #node_features = torch.cat((P1,P2),dim=1)
        #node_features = torch.cat((node_features, S1), dim=1)
        #node_features = torch.cat((node_features, S2), dim=1)
        #node_features = S1
        
        node_features = self.relu(self.linear1(node_features))
        
        node_features = self.relu(self.RGCN1(node_features, edge_index, edge_type))
        node_features = self.relu(self.RGCN1(node_features, edge_index, edge_type))
        
        node_features = self.relu(self.linear2(node_features))
        
        #residual connection
        #node_features = node_features + self.relu(self.linear(val_batch[0].squeeze(0)))
        
        probs = self.output(node_features[int(0.9 * self.label_user_count) : int(1 * self.label_user_count)])
        #print(probs.size())
        #print(label.size())
        #print(label[int(0.7 * self.label_user_count) : int(0.9 * self.label_user_count)].size())
        loss = self.CELoss(probs, label.cuda())
        temp = get_accuracy(probs, label)
        acc = temp['accuracy']
        F1 = temp['F1']
        MCC = temp['MCC']
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        self.log('val_F1', F1)
        self.log('val_MCC', MCC)


# data
dataset1 = TwiBot20Dataset(node_feature_set=2, batch_size=128, label_user_count = 11826, name = 'train') #batch_size governs (7xxx / 64) batches of training
dataset2 = TwiBot20Dataset(node_feature_set=2, batch_size=1, label_user_count = 11826, name = 'dev') #batch_size governs nothing

train_loader = DataLoader(dataset1, batch_size=1) # always should be 1
val_loader = DataLoader(dataset2, batch_size=1) #always should be 1

# model
model = GATClassifier(in_channels = 768, out_channels = 512, dropout = 0.5, label_user_count = 11826, batch_size = 32) #batch_size governs each batch how many nodes

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16)
print('training begins')
trainer.fit(model, train_loader, val_loader)

