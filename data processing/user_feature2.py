# user feature generation: RoBERTa with bio
from transformers import pipeline
import torch
from transformers import *
pretrained_weights = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
feature_extractor = pipeline('feature-extraction', model = RobertaModel.from_pretrained(pretrained_weights), tokenizer = tokenizer, device = 1)

import torch
import json
import numpy
import datetime
file_list = ['train', 'dev', 'test', 'support']
user_idlist = torch.load('user_idlist.pt').tolist()
user_features = []
now = 0
for file in file_list:
    print(file)
    f = open(file + '.json')
    users = json.load(f)
    #now = 0
    count = 0
    for user in users:
        if count % 1000 == 0:
            print(count)
        count += 1
        assert user_idlist[now] == int(user['ID'])
        now += 1
        try:
            feature_temp = torch.tensor(feature_extractor(user['profile']['description']))
            feature_temp = torch.mean(feature_temp.squeeze(0), dim=0).unsqueeze(0)
            user_features.append(feature_temp)
        except:
            user_features.append(torch.randn(1,768)) # if there is no profile bio
            print('no profile')
user_features = torch.stack(user_features)
torch.save(user_features, 'user_features2.pt')
temp = torch.load('user_features2.pt')
print(temp.size())