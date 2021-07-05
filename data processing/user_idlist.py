#user idlist
import torch
import json
file_list = ['train', 'dev', 'test', 'support']
user_idlist = []
for file in file_list:
    print(file)
    f = open(file + '.json')
    users = json.load(f)
    for user in users:
        user_idlist.append(int(user['ID']))
torch.save(torch.LongTensor(user_idlist), 'user_idlist.pt')
print(len(user_idlist))
temp = torch.load('user_idlist.pt').tolist()
print(temp[0])
