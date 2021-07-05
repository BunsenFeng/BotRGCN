# user feature generation: RoBERTa with tweets
from transformers import pipeline
import torch
from transformers import *
pretrained_weights = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights, model_max_length = 500)
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
        if count % 100 == 0:
            print(count, 'users')
        count += 1
        assert user_idlist[now] == int(user['ID'])
        now += 1
        try:
            temp = len(user['tweet'])
            #print(len(user['tweet']))
        except:
            user_features.append(torch.randn(1,768)) # if there is no tweet
            #print('zero tweet')
            continue
        try:
            this_feature = torch.zeros(1,768)
            too_long = 0
            for tweet in user['tweet'][:20]: #first 20 tweets only
                result = tokenizer(tweet)
                if len(result['input_ids']) > 500:
                    too_long += 1
                    continue
                feature_temp = torch.tensor(feature_extractor(tweet))
                feature_temp = torch.mean(feature_temp.squeeze(0), dim=0).unsqueeze(0)
                #print('feature_temp', feature_temp.size())
                this_feature = this_feature + feature_temp
                #print('this_feature', this_feature.size())
            #print('individual tweet finish encoding')
            this_feature = this_feature / (len(user['tweet']) - too_long)
            #feature_temp = torch.tensor(feature_extractor(user['profile']['description']))
            #feature_temp = torch.mean(feature_temp.squeeze(0), dim=0).unsqueeze(0)
            user_features.append(this_feature)
        except:
            user_features.append(torch.randn(1,768)) # if there is no tweet
            print('error')
user_features = torch.stack(user_features)
torch.save(user_features, 'user_features3.pt')
temp = torch.load('user_features3.pt')
print(temp.size())