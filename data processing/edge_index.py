#edge index
import torch
import json
file_list = ['train', 'dev', 'test']
user_idlist = torch.load('user_idlist.pt').tolist()
user_iddict = {}
for i in range(len(user_idlist)):
    user_iddict[user_idlist[i]] = i
print(len(user_iddict))
edge_index = []

for file in file_list:
    print(file)
    f = open(file + '.json')
    users = json.load(f)
    cnt = 0
    for user in users:
        if cnt % 100 == 0:
            print(cnt)
        cnt += 1
        user_id = int(user['ID'])
        if not user['neighbor']:
            continue
        for following in user['neighbor']['following']:
            #try:
            edge_index.append([user_iddict[user_id], user_iddict[int(following)]]) #R-GAT so that bidirectional?
            #except:
                #print('np')
                #continue
    print('half done')
    cnt = 0
    for user in users:
        if cnt % 100 == 0:
            print(cnt)
        cnt += 1
        user_id = int(user['ID'])
        if not user['neighbor']:
            continue
        for following in user['neighbor']['follower']:
            try:
                edge_index.append([user_iddict[int(follower)], user_iddict[user_id]]) #R-GAT so that bidirectional?
            except:
                continue
torch.save(torch.tensor(edge_index).t(), 'edge_index.pt')
temp = torch.load('edge_index.pt')
print(temp.size())
        
                