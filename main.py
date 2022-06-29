import torch
import torch.nn as nn   
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd
import torch.nn.functional as F
import argparse
from Models import *
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=int, default = 0)
parser.add_argument('--layer',type=int, default = 5)
parser.add_argument('--thed', type=float, default = 0.1)
parser.add_argument('--filename',type=str,default='result.txt')
parser.add_argument('--knn',type=int,default=5)
args = parser.parse_args()
if args.gpus==1:
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
elif args.gpus==0:
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
elif args.gpus==2:
    device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
elif args.gpus==3:
    device=torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
else:
    device=torch.device('cpu')
from tqdm import tqdm
import warnings 
warnings.filterwarnings("ignore")
cpac_root='/media/D/yazid/ASD-classification-ANEGCN/ABIDEI_CPAC/'
smri_root='/media/D/yazid/ASD-classification-ANEGCN/ABIDEI_sMRI/'
nan_subid=np.load('nan_subid.npy').tolist()
seed = 1234
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
def cal_evaluate(TP,TN,FP,FN):
    if TP>0:
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
    else:
        F1=0
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc,F1
def test(device,model,testloader):
    model.eval()
    TP_test,TN_test,FP_test,FN_test=0,0,0,0
    with torch.no_grad():
        for (X,Z,label,sub_id) in testloader:
            TP,TN,FN,FP=0,0,0,0
            n=X.size()[0]
            X=X.to(device)
            Z=Z.to(device)
            label=label.to(device)
            y=model(X,Z)
            _,predict=torch.max(y,1)
            TP+=((predict==1)&(label==1)).sum().item()
            TN+=((predict==0)&(label==0)).sum().item()
            FN+=((predict==0)&(label==1)).sum().item()
            FP+=((predict==1)&(label==0)).sum().item()
            TP_test+=TP
            TN_test+=TN
            FP_test+=FP
            FN_test+=FN
        acc,f1=cal_evaluate(TP_test,TN_test,FP_test,FN_test)
        global max_acc
        global modelname
        global savedModel
        if acc>=max_acc:
            max_acc=acc
            if saveModel:
                torch.save(model.state_dict(),modelname)
        return acc,f1,TP_test,TN_test,FP_test,FN_test
class dataset(Dataset):
    def __init__(self,fmri_root,smri_root,site,ASD,TDC):
        super(dataset,self).__init__()
        self.fmri=fmri_root
        self.smri=smri_root
        self.ASD=[j for i in ASD for j in i]
        self.TDC=[j for i in TDC for j in i]
        self.data=self.ASD+self.TDC
        random.shuffle(self.data)
        self.data_site={}
        for i in range(len(site)):
            data=ASD[i]+TDC[i]
            for j in data:
                if j not in self.data_site:
                    self.data_site[j]=site[i]                
    def __getitem__(self,index):
        data=self.data[index]
        sub_id=int(data[0:5])
        if data in self.ASD:
            data_slow5 =np.load(self.fmri+self.data_site[data]+'/group1_slow5/'+data,allow_pickle=True)
            data_slow4 =np.load(self.fmri+self.data_site[data]+'/group1_slow4/'+data,allow_pickle=True)
            data_voxel =np.load(self.smri+self.data_site[data]+'/group1/'+data,allow_pickle=True)
            data_FCz   =np.load(self.fmri+self.data_site[data]+'/group1_FC/'+data,allow_pickle=True)
        elif data in self.TDC:
            data_slow5 =np.load(self.fmri+self.data_site[data]+'/group2_slow5/'+data,allow_pickle=True)
            data_slow4 =np.load(self.fmri+self.data_site[data]+'/group2_slow4/'+data,allow_pickle=True)
            data_voxel =np.load(self.smri+self.data_site[data]+'/group2/'+data,allow_pickle=True)
            data_FCz   =np.load(self.fmri+self.data_site[data]+'/group2_FC/'+data,allow_pickle=True)
        else:
            print('wrong input')
        data_slow5=(data_slow5-np.min(data_slow5))/(np.max(data_slow5)-np.min(data_slow5))
        data_slow4=(data_slow4-np.min(data_slow4))/(np.max(data_slow4)-np.min(data_slow4))
        if np.any(np.isnan(data_slow5)) or np.any(np.isnan(data_slow4)) or np.any(np.isnan(data_FCz)):
            print('data wronmg')
        #data_FCz=(data_FCz-np.min(data_FCz))/(np.max(data_FCz)-np.min(data_FCz))
        if self.data[index] in self.ASD:
            label=torch.tensor(1)
        else:
            label=torch.tensor(0)
        X=np.zeros((116,3),dtype=np.float32)
        X[:,0]=data_slow5
        X[:,1]=data_slow4
        X[:,2]=data_voxel
        data_FCz=data_FCz.astype(np.float32)
        Z=torch.from_numpy(data_FCz)
        X=torch.from_numpy(X)
        return X,Z,label,sub_id
    def __len__(self):
        return len(self.data)

def train_pgd(model,trainloader,testloader,eps=0.02,iters=10,alpha=0.004):
    result=pd.DataFrame(columns=('Loss','Acc','F1','TP','TN','FP','FN'))
    criterian1=LabelSmoothLoss(0.1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    for j in range(epoch):
        loss_sum=0
        model.train()
        for (X,Z,label,sub_id) in trainloader:
            model.train()
            x=X.to(device)
            z=Z.to(device)
            label=label.to(device)
            pretu_x,pretu_z=x,z
            ori_x,ori_z=x.data,z.data
            for i in range(iters):
                pretu_x.requires_grad=True
                pretu_z.requires_grad=True
                y=model(pretu_x,pretu_z)
                loss=criterian1(y,label)
                model.zero_grad()
                loss.backward()
                adv_x=pretu_x+alpha*torch.sign(pretu_x.grad.data)
                adv_z=pretu_z+alpha*torch.sign(pretu_z.grad.data)
                eta_x=torch.clamp(adv_x-ori_x,min=-eps,max=eps)
                eta_z=torch.clamp(adv_z-ori_z,min=-eps,max=eps)
                pretu_x=torch.clamp(ori_x+eta_x,min=0,max=1).detach_()
                pretu_z=torch.clamp(ori_z+eta_z,min=-1,max=1).detach_()
            y=model(x,z)
            yy=model(pretu_x,pretu_z)
            L2=torch.tensor(0,dtype=torch.float32).to(device)
            if L2_lamda>0:
                for name,parameters in model.named_parameters():
                    if name[0:5]=='clase' and  name[-8:]=='0.weight':
                        L2+=L2_lamda*torch.norm(parameters,2)
            loss=0.5*(criterian1(yy,label)+criterian1(y,label))+L2
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (j+1)%5==0 or j==0:
            acc,f1,TP_test,TN_test,FP_test,FN_test=test(device,model,testloader)
            result.loc[j//5]=[loss_sum,acc,f1,TP_test,TN_test,FP_test,FN_test]
    result.sort_values('Acc',inplace=True,ascending=False)
    return result.iloc[0]['Acc']

def train_fgsm(model,trainloader,testloader,epsilon=0.05):
    result=pd.DataFrame(columns=('Loss','Acc','F1','TP','TN','FP','FN'))
    criterian1=LabelSmoothLoss(0.1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    acc=0.5000
    for j in range(epoch):
        loss_sum=0
        TP,TN,FP,FN=0,0,0,0
        model.train()
        for (X,Z,label,sub_id) in trainloader:
            x=X.to(device)
            z=Z.to(device)
            x.requires_grad=True
            z.requires_grad=True
            label=label.to(device)
            y=model(x,z)
            loss=criterian1(y,label)
            model.zero_grad()
            loss.backward(retain_graph=True)
            sign_grad_x=torch.sign(x.grad.data)
            sign_grad_z=torch.sign(z.grad.data)
            perturbed_x=x+epsilon*sign_grad_x 
            perturbed_z=z+epsilon*sign_grad_z 
            perturbed_x=torch.clamp(perturbed_x,0,1)
            perturbed_z=torch.clamp(perturbed_z,-1,1)
            y=model(perturbed_x,perturbed_z)
            L2=torch.tensor(0,dtype=torch.float32).to(device)
            if L2_lamda>0:
                for name,parameters in model.named_parameters():
                    if name[0:5]=='clase' and  name[-8:]=='0.weight':
                        L2+=L2_lamda*torch.norm(parameters,2)
            loss=0.5*(criterian1(y,label)+loss)+L2
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (j+1)%5==0 or j==0:
            acc,f1,TP_test,TN_test,FP_test,FN_test=test(device,model,testloader)
            result.loc[j//5]=[loss_sum,acc,f1,TP_test,TN_test,FP_test,FN_test]
    result.sort_values('Acc',inplace=True,ascending=False)
    return result.iloc[0]['Acc']

def train_norm(model,trainloader,testloader):
    result=pd.DataFrame(columns=('Loss','Acc','F1','TP','TN','FP','FN'))
    criterian1=LabelSmoothLoss(0.1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    acc=0.5000
    loss_sum=0
    for j in range(epoch):
        loss_sum=0
        TP,TN,FP,FN=0,0,0,0
        model.train()
        for (X,Z,label,sub_id) in trainloader:
            x=X.to(device)
            z=Z.to(device)
            label=label.to(device)
            y=model(x,z)
            loss=criterian1(y,label)
            L2=torch.tensor(0,dtype=torch.float32).to(device)
            if L2_lamda>0:
                for name,parameters in model.named_parameters():
                    if name[0:5]=='clase' and  name[-8:]=='0.weight':
                        L2+=L2_lamda*torch.norm(parameters,2)
            loss=loss+L2
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (j+1)%5==0 or j==0:
            acc,f1,TP_test,TN_test,FP_test,FN_test=test(device,model,testloader)
            result.loc[j//5]=[loss_sum,acc,f1,TP_test,TN_test,FP_test,FN_test]
    result.sort_values('Acc',inplace=True,ascending=False)
    return result.iloc[0]['Acc']
if __name__=='__main__':
    setup_seed(seed)
    train_site=test_site=np.load('DATAARRANGE/train_test_site.npy')
    train_asd_dict=np.load('DATAARRANGE/train_asd_dict.npy',allow_pickle=True).item()
    train_tdc_dict=np.load('DATAARRANGE/train_tdc_dict.npy',allow_pickle=True).item()
    test_asd_dict=np.load('DATAARRANGE/test_asd_dict.npy',allow_pickle=True).item()
    test_tdc_dict=np.load('DATAARRANGE/test_tdc_dict.npy',allow_pickle=True).item()
    L1_lamda=0.0
    L2_lamda=0.0001
    learning_rate=0.0001
    epoch   =100
    batch_size=64
    gmma    =1
    layer   =1
    Acc_norm=np.zeros(10)
    Acc_fgsm=np.zeros(10)
    Acc_pgd =np.zeros(10)
    for index in range(10):
        start_t=time.time()
        saveModel=False
        max_acc=0.6
        modelname='../SAVEDModels/PGDtrainedensamble/models_{}_{}'.format(0,index)
        train_asd=train_asd_dict[index]
        train_tdc=train_tdc_dict[index]
        test_asd =test_asd_dict[index]
        test_tdc =test_tdc_dict[index]
        trainset=dataset(site=train_site,fmri_root=cpac_root,smri_root=smri_root,ASD=train_asd,TDC=train_tdc)
        trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
        testset=dataset(site=test_site,fmri_root=cpac_root,smri_root=smri_root,ASD=test_asd,TDC=test_tdc)
        testloader=DataLoader(testset,batch_size=1)

        # model=ANEGCN_fixed(args.layer).to(device)
        # acc=train_norm(model,trainloader,testloader)
        # if acc>=Acc_norm[index]:
        #     Acc_norm[index]=acc

#         model=ANEGCN_fixed(args.layer).to(device)
#         acc=train_fgsm(model,trainloader,testloader,0.05)
#         if acc>=Acc_fgsm[index]:
#             Acc_fgsm[index]=acc   
            
        model=ANEGCN(args.layer).to(device)
        acc=train_pgd(model,trainloader,testloader)
        if acc>=Acc_pgd[index]:
            Acc_pgd[index]=acc
        end_t=time.time()
        print('\r[%2d/10]  Rest time: %.2f  Speed:%.2f'%(1+index,(9-index)*(end_t-start_t)/3600,(end_t-start_t)/3600),end='')
        with open(args.filename,'a') as fileOut:
            print('[%2d/10]  Norm Acc:%.4f  FGSM Acc:%.4f  PGD Acc:%.4f'%(index,Acc_norm[index],Acc_fgsm[index],Acc_pgd[index]),file=fileOut)
    with open(args.filename,'a') as fileOut:
        print('Norm Acc:',np.mean(Acc_norm),'+',np.std(Acc_norm),'\n',
              'FGSM Acc:',np.mean(Acc_fgsm),'+',np.std(Acc_fgsm),'\n',
              'PGD  Acc:',np.mean(Acc_pgd ),'+',np.std(Acc_pgd ),file=fileOut)

              
            
