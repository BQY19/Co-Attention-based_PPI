import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import random

class mydataset(Dataset):
    def __init__(self,datalist,embpath):
        self.datalist = datalist  
        self.len = len(self.datalist)
        self.emb = np.load(r"%s" % embpath, allow_pickle=True).item()

    def __len__(self):
        return self.len

    def __getitem__(self, item):
                    
            data_ = self.datalist[item]
            content=data_[0]    #content1\2?embeding lenth
            padmask1=torch.zeros(2065-len(content[0][1]))
            content_mask1=torch.cat([torch.Tensor((content[0])[1]),padmask1],dim=0)
            padmask2 = torch.zeros(2415 - len(content[1][1]))
            content_mask2 = torch.cat([torch.Tensor((content[1])[1]), padmask2], dim=0)
            pademb1 = torch.zeros([2065 - len(content[0][1]), 1024])
            pademb2 = torch.zeros([2415- len(content[1][1]), 1024])
            target=data_[1]
            id1=content[0][2]
            id2=content[1][2]
            if target==1:
               target=torch.ones(1)
               content_emb1 = torch.cat([torch.Tensor(self.emb["%s" %(content[0])[2]]),pademb1])
               content_emb2 = torch.cat([torch.Tensor(self.emb["%s"%(content[1])[2]]),pademb2])
            else :
               target=torch.zeros(1)
               content_emb1 = torch.cat([torch.Tensor(self.emb["%s" %(content[0])[2]]),pademb1])
               content_emb2 = torch.cat([torch.Tensor(self.emb["%s"%(content[1])[2]]),pademb2])
   
            return content_emb1,content_mask1,content_emb2,content_mask2,target,id1,id2


class mydataset_site(Dataset):
    def __init__(self, datalist, embpath):
        self.datalist = datalist
        self.len = len(self.datalist)
        self.emb = np.load(r"%s" % embpath, allow_pickle=True).item()

    def __len__(self):
        return self.len

    def __getitem__(self, item):

        data_ = self.datalist[item]
        content = data_[0]
        padmask1 = torch.zeros(766 - len(content[0][1]))
        content_mask1 = torch.cat([torch.Tensor((content[0])[1]), padmask1], dim=0)
        padmask2 = torch.zeros(766 - len(content[1][1]))
        content_mask2 = torch.cat([torch.Tensor((content[1])[1]), padmask2], dim=0)
        id1 = content[0][2]
        id2 = content[1][2]
        seq1 = content[0][0]
        seq2 = content[1][0]

        target1 = content[0][3]
        target2 = content[1][3]
        target1pad = torch.zeros(766 - len(target1))
        target2pad = torch.zeros(766 - len(target2))
        target1 = torch.cat([torch.Tensor(target1), target1pad])
        target2 = torch.cat([torch.Tensor(target2), target2pad])
        target = torch.cat([torch.Tensor(target1), torch.Tensor(target2)])
        id1 = content[0][2]
        id2 = content[1][2]
        pademb1 = torch.zeros([766 - len(self.emb["%s" % (content[0])[2]]), 1024])
        pademb2 = torch.zeros([766 - len(self.emb["%s" % (content[1])[2]]), 1024])
        content_emb1 = torch.cat([torch.Tensor(self.emb["%s" % (content[0])[2]]), pademb1])
        content_emb2 = torch.cat([torch.Tensor(self.emb["%s" % (content[1])[2]]), pademb2])
        return content_emb1, content_mask1, content_emb2, content_mask2, target, id1, id2


def shuffle_split(data_path, trainrate, validrate, seed):
    data = np.load(data_path, allow_pickle=True)
    datalist = data.tolist()
    lenth = len(datalist)
    # index_shuffle=[i for i in range(lenth)]
    if not seed == False:
        random.seed(seed)
    random.shuffle(datalist)
    trainlen = int(lenth * trainrate)
    validlen = int(lenth * validrate)
    train = datalist[0:trainlen]
    valid = datalist[trainlen:]
    test = datalist[trainlen + validlen:]
    return train, valid, test


def getBinaryTensor(imgTensor, boundary):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)

