#label list
import torch
user_idlist = torch.load('user_idlist.pt').tolist()
print(user_idlist[0])

label_dict = {}
f = open('label.txt')
for line in f:
    temp = line.split(' ')
    label_dict[int(temp[1])] = int(temp[2])

label_list = []
count = 0
for id in user_idlist:
    try:
        label_list.append(label_dict[id])
        count += 1
    except:
        label_list.append(-1) #support set
print(count)
torch.save(torch.tensor(label_list), 'label_list.pt')