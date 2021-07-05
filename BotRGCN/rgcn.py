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

def get_accuracy(probs, labels):
    probs = torch.argmax(probs, dim = 1)
    correct = 0
    for i in range(len(probs)):
        if probs[i] == labels[i]:
            correct += 1
    return correct / len(probs)

class TwiBot20Dataset(Dataset):
    def __init__(self, node_feature_set, batch_size, label_user_count, name): # name = train/dev
        path = '/new_temp/fsb/fsb_asonam/'
        self.node_features = torch.load(path + 'user_features' + str(node_feature_set) + '.pt').squeeze(1).float()
        if node_feature_set == 1:
            self.node_features = (self.node_features - self.node_features.mean()) / sqrt(self.node_features.var())
        if node_feature_set == 3:
            self.node_features = self.node_features.squeeze(1)
        self.edge_index = torch.load(path + 'edge_index.pt')
        self.edge_type = torch.cat((torch.zeros(1,len(self.edge_index[0])), torch.ones(1,len(self.edge_index[0]))), dim = 1).long()
        #bidirectional edges???
        self.edge_index = torch.cat((self.edge_index, torch.stack([self.edge_index[1], self.edge_index[0]])), dim = 1)
        print(self.edge_index.size())
        
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
        return self.node_features, self.edge_index, self.label_list, self.edge_type

class GATClassifier(pl.LightningModule):
    def __init__(self, in_channels, out_channels, dropout, label_user_count, batch_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_user_count = label_user_count
        self.batch_size = batch_size
        self.linear = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.linear3 = nn.Linear(out_channels, out_channels)
        #self.linearP = nn.Linear(6, out_channels / 2)
        torch.nn.init.kaiming_uniform(self.linear.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform(self.linear2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform(self.linear3.weight, nonlinearity='leaky_relu')
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
        node_features = x[0].squeeze(0)
        edge_index = x[1].squeeze(0)
        node_features = self.relu(self.linear(node_features))
        node_features = self.relu(self.linear2(node_features))
        edge_type = x[3].squeeze(0).squeeze(0)
        
        #no dropout for validation
        #node_features = self.dropout_layer(node_features)
        
        node_features = self.relu(self.RGCN1(node_features, edge_index, edge_type))
        node_features = self.relu(self.RGCN1(node_features, edge_index, edge_type))
        
        node_features = self.relu(self.linear3(node_features))
        
        #!! redisual connection
        #node_features = node_features + self.relu(self.linear(x[0].squeeze(0)))
        
        
        probs = self.output[node_features[int(0.9 * self.label_user_count) : self.label_user_count]]
        return torch.argmax(probs, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        node_features = train_batch[0].squeeze(0)
        edge_index = train_batch[1].squeeze(0)
        edge_type = train_batch[3].squeeze(0).squeeze(0)
        label = train_batch[2].squeeze(0).tolist()
        label = torch.LongTensor(label[:int(0.7 * self.label_user_count)])
        node_features = self.dropout_layer(self.relu(self.linear(node_features)))
        node_features = self.dropout_layer(self.relu(self.linear2(node_features)))
        
        
        node_features = self.dropout_layer(self.relu(self.RGCN1(node_features, edge_index, edge_type)))
        node_features = self.dropout_layer(self.relu(self.RGCN1(node_features, edge_index, edge_type)))
        
        node_features = self.dropout_layer(self.relu(self.linear3(node_features)))
        
        #!! residual connection
        #node_features = node_features + self.relu(self.linear(train_batch[0].squeeze(0)))
        
        batch_id = torch.randperm(int(0.7 * self.label_user_count))[:self.batch_size].tolist()
        
        probs = self.output(node_features[batch_id])
        #print(probs.size())
        #print(label[batch_id].size())
        loss = self.CELoss(probs, label[batch_id].cuda())
        acc = get_accuracy(probs, label[batch_id])
        
        #regularization
        #lambda1 = 1e-5
        #lambda2 = 0.01
        #layers = [model.linear, model.linear2, model.linear3]
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
        node_features = val_batch[0].squeeze(0)
        edge_index = val_batch[1].squeeze(0)
        edge_type = val_batch[3].squeeze(0).squeeze(0)
        #print(val_batch[2].size())
        label = val_batch[2].squeeze(0).tolist()
        label = torch.LongTensor(label[int(0.9 * self.label_user_count) : int(1 * self.label_user_count)])
        node_features = self.relu(self.linear(node_features))
        node_features = self.relu(self.linear2(node_features))
        #print(node_features.size())
        #print(edge_index.size())
        
        #no dropout for validation
        #node_features = self.dropout_layer(node_features)
        
        node_features = self.relu(self.RGCN1(node_features, edge_index, edge_type))
        node_features = self.relu(self.RGCN1(node_features, edge_index, edge_type))
        
        node_features = self.dropout_layer(self.relu(self.linear3(node_features)))
        
        #residual connection
        #node_features = node_features + self.relu(self.linear(val_batch[0].squeeze(0)))
        
        probs = self.output(node_features[int(0.9 * self.label_user_count) : int(1 * self.label_user_count)])
        #print(probs.size())
        #print(label.size())
        #print(label[int(0.7 * self.label_user_count) : int(0.9 * self.label_user_count)].size())
        loss = self.CELoss(probs, label.cuda())
        acc = get_accuracy(probs, label)
        self.log('val_acc', acc)
        self.log('val_loss', loss)


# data
dataset1 = TwiBot20Dataset(node_feature_set=2, batch_size=128, label_user_count = 11826, name = 'train') #batch_size governs (7xxx / 64) batches of training
dataset2 = TwiBot20Dataset(node_feature_set=2, batch_size=1, label_user_count = 11826, name = 'dev') #batch_size governs nothing

train_loader = DataLoader(dataset1, batch_size=1) # always should be 1
val_loader = DataLoader(dataset2, batch_size=1) #always should be 1

# model
model = GATClassifier(in_channels = 768, out_channels = 128, dropout = 0.5, label_user_count = 11826, batch_size = 128) #batch_size governs each batch how many nodes

# training
trainer = pl.Trainer(gpus=1, num_nodes=1)
print('training begins')
trainer.fit(model, train_loader, val_loader)

