# user feature generation: Alhosseini
import torch
import json
import numpy
import datetime
file_list = ['train', 'dev', 'test', 'support']
user_idlist = torch.load('user_idlist.pt').tolist()
user_features = []
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

now = 0
for file in file_list:
    print(file)
    f = open(file + '.json')
    users = json.load(f)
    for user in users:
        assert user_idlist[now] == int(user['ID'])
        now += 1
        feature_temp = []
        try:
            #age
            thing = user['profile']['created_at'].split(' ')
            then_time = datetime.date(int(thing[5]), months.index(thing[1])+1, int(thing[2]))
            feature_temp.append((datetime.date(2021, 5, 30) - then_time).days)
            #favorites count
            feature_temp.append(int(user['profile']['favourites_count']))
            #statues count
            feature_temp.append(int(user['profile']['statuses_count']))
            #account length name
            feature_temp.append(len(user['profile']['screen_name']))
            #followers count
            feature_temp.append(int(user['profile']['followers_count']))
            #friends count
            feature_temp.append(int(user['profile']['friends_count']))
        except:
            feature_temp = [0, 0, 0, 0, 0, 0]
            print('profile missing')
        user_features.append(feature_temp)
        #user_features.append(torch.tensor(feature_temp))
torch.save(torch.tensor(user_features), 'user_feature1.pt')
temp = torch.load('user_feature1.pt')
print(temp.size())
