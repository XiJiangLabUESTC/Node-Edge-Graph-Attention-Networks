import torch
import torch.nn as nn 
class GraphNorm(nn.Module):
    def __init__(self,features):
        super(GraphNorm,self).__init__()
        self.weight = nn.Parameter(torch.randn(features))
        self.bias = nn.Parameter(torch.randn(features))
        self.alpha = nn.Parameter(torch.randn(features))       
    def forward(self,X):
        X=X.transpose(0,1)
        X=self.weight*(X-self.alpha*X.mean(0))/X.std(0)
        X=X.transpose(0,1)+self.bias
        return X

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.conv1=nn.Conv1d(in_channels=3,out_channels=3,kernel_size=1,padding=0)
        self.conv2=nn.Conv1d(in_channels=116,out_channels=116,kernel_size=1,padding=0)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,Z,X):
        K=self.conv1(X.permute(0,2,1))# BS,x_c,x_dim
        Q=K.permute(0,2,1)# BS,x_dim,x_c
        V=self.conv2(Z.permute(0,2,1))# Bs,z_c,z_dim
        attention=self.softmax(torch.matmul(Q,K))#BS,x_dim,x_dim
        out=torch.bmm(attention,V).permute(0,2,1)#BS,z_dim,z_c
        return out

class NEGAN(nn.Module):
    def __init__(self,layer):
        super(NEGAN,self).__init__()
        self.layer =layer
        self.relu  =nn.ReLU()
        self.atten =nn.ModuleList([Attention() for i in range(layer)])
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.node_w=nn.ParameterList([nn.Parameter(torch.randn((3,3),dtype=torch.float32)) for i in range(layer)])
        self.edge_w=nn.ParameterList([nn.Parameter(torch.randn((116,116),dtype=torch.float32)) for i in range(layer)])
        self.line_n=nn.ModuleList([nn.Sequential(nn.Linear(116*3,128),nn.ReLU(),nn.BatchNorm1d(128)) for i in range(layer+1)])
        self.line_e=nn.ModuleList([nn.Sequential(nn.Linear(116*116,128*3),nn.ReLU(),nn.BatchNorm1d(128*3)) for i in range(layer+1)])
        self.clase =nn.Sequential(nn.Linear(128*4*(self.layer+1),1024),nn.Dropout(0.2),nn.ReLU(),
                                   nn.Linear(1024,2))
        self.ones=nn.Parameter(torch.ones((116),dtype=torch.float32),requires_grad=False)
        self._initialize_weights()
    # params initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def normalized(self,Z):
        n=Z.size()[0]
        A=Z[0,:,:]
        A=A+torch.diag(self.ones)
        d=A.sum(1)
        D=torch.diag(torch.pow(d,-1))
        A=D.mm(A).reshape(1,116,116)
        for i in range(1,n):
            A1=Z[i,:,:]+torch.diag(self.ones)
            d=A1.sum(1)
            D=torch.diag(torch.pow(d,-1))
            A1=D.mm(A1).reshape(1,116,116)
            A=torch.cat((A,A1),0)
        return A
        
    def update_A(self,Z):
        n=Z.size()[0]
        A=Z[0,:,:]
        Value,_=torch.topk(torch.abs(A.view(-1)),int(116*116*args.thed))
        A=(torch.abs(A)>=Value[-1])+torch.tensor(0,dtype=torch.float32)
        A=A.reshape(1,116,116)
        for i in range(1,n):
            A2=Z[i,:,:]
            Value,_=torch.topk(torch.abs(A2.view(-1)),int(116*116*args.thed))
            A2=(torch.abs(A2)>=Value[-1])+torch.tensor(0,dtype=torch.float32)
            A2=A2.reshape(1,116,116)
            A=torch.cat((A,A2),0)
        return A
        
    def forward(self,X,Z):
        n=X.size()[0]
        XX=self.line_n[0](X.view(n,-1))
        ZZ=self.line_e[0](Z.view(n,-1))
        for i in range(self.layer):
            A=self.atten[i](Z,X)
            Z1=torch.matmul(A,Z)
            Z2=torch.matmul(Z1,self.edge_w[i])
            Z=self.relu(self.norm_e[i](Z2))+Z
            ZZ=torch.cat((ZZ,self.line_e[i+1](Z.view(n,-1))),dim=1)
            X1=torch.matmul(A,X)
            X1=torch.matmul(X1,self.node_w[i])
            X=self.relu(self.norm_n[i](X1))+X
            XX=torch.cat((XX,self.line_n[i+1](X.view(n,-1))),dim=1)
        XZ=torch.cat((XX,ZZ),1)
        y=self.clase(XZ)
        return y


class ANEGCN_fixed(nn.Module):
    """相较于初始模型，只用一层attention,后面的层使用邻接矩阵与第一个相同"""
    def __init__(self,layer):
        super(ANEGCN_fixed,self).__init__()
        self.layer =layer
        self.relu  =nn.ReLU()
        self.atten =Attention()
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.node_w=nn.ModuleList([nn.Linear(3,3) for i in range(layer)])
        self.edge_w=nn.ModuleList([nn.Linear(116,116) for i in range(layer)])
        self.line_n=nn.ModuleList([nn.Sequential(nn.Linear(116*3,128),nn.ReLU(),nn.BatchNorm1d(128)) for i in range(layer+1)])
        self.line_e=nn.ModuleList([nn.Sequential(nn.Linear(116*116,128*3),nn.ReLU(),nn.BatchNorm1d(128*3)) for i in range(layer+1)])
        self.clase =nn.Sequential(nn.Linear(128*4*(self.layer+1),1024),nn.Dropout(0.2),nn.ReLU(),
                                   nn.Linear(1024,2))
        self.ones=nn.Parameter(torch.ones((116),dtype=torch.float32),requires_grad=False)
        self._initialize_weights()
    # params initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def normalized(self,Z):
        n=Z.size()[0]
        A=Z[0,:,:]
        A=A+torch.diag(self.ones)
        d=A.sum(1)
        D=torch.diag(torch.pow(d,-1))
        A=D.mm(A).reshape(1,116,116)
        for i in range(1,n):
            A1=Z[i,:,:]+torch.diag(self.ones)
            d=A1.sum(1)
            D=torch.diag(torch.pow(d,-1))
            A1=D.mm(A1).reshape(1,116,116)
            A=torch.cat((A,A1),0)
        return A
        
    def update_A(self,Z):
        n=Z.size()[0]
        A=Z[0,:,:]
        Value,_=torch.topk(torch.abs(A.view(-1)),int(116*116*self.thed))
        A=(torch.abs(A)>=Value[-1])+torch.tensor(0,dtype=torch.float32)
        A=A.reshape(1,116,116)
        for i in range(1,n):
            A2=Z[i,:,:]
            Value,_=torch.topk(torch.abs(A2.view(-1)),int(116*116*self.thed))
            A2=(torch.abs(A2)>=Value[-1])+torch.tensor(0,dtype=torch.float32)
            A2=A2.reshape(1,116,116)
            A=torch.cat((A,A2),0)
        return A
        
    def forward(self,X,Z):
        n=X.size()[0]
        XX=self.line_n[0](X.view(n,-1))
        ZZ=self.line_e[0](Z.view(n,-1))
        A=self.atten(Z,X)
        for i in range(self.layer):
            Z1=self.edge_w[i](torch.matmul(A,Z))
            Z=self.relu(self.norm_e[i](Z1))+Z
            ZZ=torch.cat((ZZ,self.line_e[i+1](Z.view(n,-1))),dim=1)
            X1=self.node_w[i](torch.matmul(A,X))
            X=self.relu(self.norm_n[i](X1))+X
            #X.register_hook(grad_X_hook)
            #feat_X_hook(X)
            XX=torch.cat((XX,self.line_n[i+1](X.view(n,-1))),dim=1)
        XZ=torch.cat((XX,ZZ),1)
        y=self.clase(XZ)
        #print(self.clase[0].weight)
        return y
class ANEGCN_noatt(nn.Module):
    """与初始模型相比较，去掉了Attention，转而只使用通过阈值的方法来确定"""
    def __init__(self,layer,thed):
        super(ANEGCN_noatt,self).__init__()
        self.layer =layer
        self.thed = thed
        self.relu  =nn.ReLU()
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        
#         self.node_w=nn.ParameterList([nn.Parameter(torch.randn((3,3),dtype=torch.float32)) for i in range(layer)])
#         self.edge_w=nn.ParameterList([nn.Parameter(torch.randn((116,116),dtype=torch.float32)) for i in range(layer)])    
        self.node_w=nn.ModuleList([nn.Linear(3,3) for i in range(layer)])
        self.edge_w=nn.ModuleList([nn.Linear(116,116) for i in range(layer)])
        
        self.line_n=nn.ModuleList([nn.Sequential(nn.Linear(116*3,128),nn.ReLU(),nn.BatchNorm1d(128)) for i in range(layer+1)])
        self.line_e=nn.ModuleList([nn.Sequential(nn.Linear(116*116,128*3),nn.ReLU(),nn.BatchNorm1d(128*3)) for i in range(layer+1)])
        self.clase =nn.Sequential(nn.Linear(128*4*(self.layer+1),1024),nn.Dropout(0.2),nn.ReLU(),
                                   nn.Linear(1024,2))
        self.ones=nn.Parameter(torch.ones((116),dtype=torch.float32),requires_grad=False)
        self._initialize_weights()
    # params initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def normalized(self,Z):
        n=Z.size()[0]
        A=Z[0,:,:]
        A=A+torch.diag(self.ones)
        d=A.sum(1)
        D=torch.diag(torch.pow(d,-1))
        A=D.mm(A).reshape(1,116,116)
        for i in range(1,n):
            A1=Z[i,:,:]+torch.diag(self.ones)
            d=A1.sum(1)
            D=torch.diag(torch.pow(d,-1))
            A1=D.mm(A1).reshape(1,116,116)
            A=torch.cat((A,A1),0)
        return A
        
    def update_A(self,Z):
        n=Z.size()[0]
        A=Z[0,:,:]
        Value,_=torch.topk(torch.abs(A.view(-1)),int(116*116*self.thed))
        A=(torch.abs(A)>=Value[-1])+torch.tensor(0,dtype=torch.float32)
        A=A.reshape(1,116,116)
        for i in range(1,n):
            A2=Z[i,:,:]
            Value,_=torch.topk(torch.abs(A2.view(-1)),int(116*116*self.thed))
            A2=(torch.abs(A2)>=Value[-1])+torch.tensor(0,dtype=torch.float32)
            A2=A2.reshape(1,116,116)
            A=torch.cat((A,A2),0)
        return A
        
    def forward(self,X,Z):
        n=X.size()[0]
        XX=self.line_n[0](X.view(n,-1))
        ZZ=self.line_e[0](Z.view(n,-1))
        for i in range(self.layer):
            A=self.update_A(Z)
            A=self.normalized(Z)
            Z1=self.edge_w[i](torch.matmul(A,Z))
            Z=self.relu(self.norm_e[i](Z1))+Z
            ZZ=torch.cat((ZZ,self.line_e[i+1](Z.view(n,-1))),dim=1)
            X1=self.node_w[i](torch.matmul(A,X))
            X=self.relu(self.norm_n[i](X1))+X
            #X.register_hook(grad_X_hook)
            #feat_X_hook(X)
            XX=torch.cat((XX,self.line_n[i+1](X.view(n,-1))),dim=1)
        XZ=torch.cat((XX,ZZ),1)
        y=self.clase(XZ)
        #print(self.clase[0].weight)
        return y

class ANEGCN_1(nn.Module):
    '''不在使用Feedforward 作为降采样的方法，使用线性层'''
    def __init__(self,layer):
        super(ANEGCN_1,self).__init__()
        self.layer =layer
        self.celu  =nn.ReLU()
        self.atten =nn.ModuleList([Attention() for i in range(layer)])
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.node_w=nn.ModuleList([nn.Linear(3,3) for i in range(layer)])
        self.edge_w=nn.ModuleList([nn.Linear(116,116) for i in range(layer)])
        self.line_n=nn.ModuleList([nn.Sequential(nn.Linear(3,1),nn.ReLU(),nn.BatchNorm1d(116)) for i in range(layer+1)])
        self.line_e=nn.ModuleList([nn.Sequential(nn.Linear(116,3),nn.ReLU(),nn.BatchNorm1d(116)) for i in range(layer+1)])
        self.clase=nn.Sequential(nn.Linear(116*4*(self.layer+1),1024),nn.Dropout(0.2),nn.ReLU(),
                                   nn.Linear(1024,2))
        
    def forward(self,X,Z):
        # X: bs*N*n_features
        # Z: bs8N*N
        n=X.size()[0]
        XX=self.line_n[0](X).view(n,-1)
        ZZ=self.line_e[0](Z).view(n,-1)
        for i in range(self.layer):
            A=self.atten[i](Z,X)
            Z1=torch.matmul(A,Z)
            Z2=self.edge_w[i](Z1)
            Z=self.celu(self.norm_e[i](Z2))+Z
            ZZ=torch.cat((ZZ,self.line_e[i+1](Z).view(n,-1)),dim=1)
            X1=torch.matmul(A,X)
            X1=self.node_w[i](X1)
            X=self.celu(self.norm_n[i](X1))+X
            XX=torch.cat((XX,self.line_n[i+1](X).view(n,-1)),dim=1)
        XZ=torch.cat((XX,ZZ),1)
        y=self.clase(XZ)
        return y

    
class ConvDownSample(nn.Module):
    def __init__(self,in_feat):
        super(ConvDownSample,self).__init__()
        if in_feat==3:           
            self.Conv=nn.Conv1d(1,1,3)
        elif in_feat==116:
            self.Conv=nn.Conv1d(1,1,39*2,39,39)
        self.activeFunc=nn.ReLU()
        self.norm=nn.BatchNorm1d(1)
    def forward(self,X):
        h=self.norm(self.activeFunc(self.Conv(X.reshape(X.shape[0]*X.shape[1],1,X.shape[2]))))
        return h
class ANEGCN_2(nn.Module):
    '''使用卷积层作为降采样的方法'''
    def __init__(self,layer):
        super(ANEGCN_2,self).__init__()
        self.layer =layer
        self.celu  =nn.ReLU()
        self.atten =nn.ModuleList([Attention() for i in range(layer)])
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.node_w=nn.ModuleList([nn.Linear(3,3) for i in range(layer)])
        self.edge_w=nn.ModuleList([nn.Linear(116,116) for i in range(layer)])
        self.line_n=nn.ModuleList([ConvDownSample(3) for i in range(layer+1)])
        self.line_e=nn.ModuleList([ConvDownSample(116) for i in range(layer+1)])
        self.clase=nn.Sequential(nn.Linear(116*4*(self.layer+1),1024),nn.Dropout(0.2),nn.ReLU(),
                                   nn.Linear(1024,2))

    def forward(self,X,Z):
        # X: bs*N*n_features
        # Z: bs8N*N
        n=X.size()[0]
        XX=self.line_n[0](X).view(n,-1)
        ZZ=self.line_e[0](Z).view(n,-1)
        for i in range(self.layer):
            A=self.atten[i](Z,X)
            Z1=self.edge_w[i](torch.matmul(A,Z))
            Z=self.celu(self.norm_e[i](Z1))+Z
            ZZ=torch.cat((ZZ,self.line_e[i+1](Z).view(n,-1)),dim=1)
            X1=self.node_w[i](torch.matmul(A,X))
            X=self.celu(self.norm_n[i](X1))+X
            XX=torch.cat((XX,self.line_n[i+1](X).view(n,-1)),dim=1)
        XZ=torch.cat((XX,ZZ),1)
        y=self.clase(XZ)
        return y  

class AvgPoolDownSample(nn.Module):
    def __init__(self,in_feat):
        super(AvgPoolDownSample,self).__init__()
        if in_feat==3:           
            self.pool=nn.AvgPool1d(3,1,0)
        elif in_feat==116:
            self.pool=nn.AvgPool1d(40,38,0)
        self.activeFunc=nn.ReLU()
        self.norm=nn.BatchNorm1d(1)
    def forward(self,X):
        h=self.norm(self.activeFunc(self.pool(X.reshape(X.shape[0]*X.shape[1],1,X.shape[2]))))
        return h   
class ANEGCN_3(nn.Module):
    '''使用均值池化层作为降采样的方法'''
    def __init__(self,layer):
        super(ANEGCN_3,self).__init__()
        self.layer =layer
        self.celu  =nn.ReLU()
        self.atten =nn.ModuleList([Attention() for i in range(layer)])
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.node_w=nn.ModuleList([nn.Linear(3,3) for i in range(layer)])
        self.edge_w=nn.ModuleList([nn.Linear(116,116) for i in range(layer)])
        self.line_n=nn.ModuleList([AvgPoolDownSample(3) for i in range(layer+1)])
        self.line_e=nn.ModuleList([AvgPoolDownSample(116) for i in range(layer+1)])
        self.clase=nn.Sequential(nn.Linear(116*4*(self.layer+1),1024),nn.Dropout(0.2),nn.ReLU(),
                                   nn.Linear(1024,2))

    def forward(self,X,Z):
        # X: bs*N*n_features
        # Z: bs8N*N
        n=X.size()[0]
        XX=self.line_n[0](X).view(n,-1)
        ZZ=self.line_e[0](Z).view(n,-1)
        for i in range(self.layer):
            A=self.atten[i](Z,X)
            Z1=self.edge_w[i](torch.matmul(A,Z))
            Z=self.celu(self.norm_e[i](Z1))+Z
            ZZ=torch.cat((ZZ,self.line_e[i+1](Z).view(n,-1)),dim=1)
            X1=self.node_w[i](torch.matmul(A,X))
            X=self.celu(self.norm_n[i](X1))+X
            XX=torch.cat((XX,self.line_n[i+1](X).view(n,-1)),dim=1)
        XZ=torch.cat((XX,ZZ),1)
        y=self.clase(XZ)
        return y

class MaxPoolDownSample(nn.Module):
    def __init__(self,in_feat):
        super(MaxPoolDownSample,self).__init__()
        if in_feat==3:           
            self.pool=nn.MaxPool1d(3,1,0)
        elif in_feat==116:
            self.pool=nn.MaxPool1d(40,38,0)
        self.activeFunc=nn.ReLU()
        self.norm=nn.BatchNorm1d(1)
    def forward(self,X):
        h=self.norm(self.activeFunc(self.pool(X.reshape(X.shape[0]*X.shape[1],1,X.shape[2]))))
        return h   
class ANEGCN_4(nn.Module):
    '''使用最大池化层作为降采样的方法'''
    def __init__(self,layer):
        super(ANEGCN_4,self).__init__()
        self.layer =layer
        self.celu  =nn.ReLU()
        self.atten =nn.ModuleList([Attention() for i in range(layer)])
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.node_w=nn.ModuleList([nn.Linear(3,3) for i in range(layer)])
        self.edge_w=nn.ModuleList([nn.Linear(116,116) for i in range(layer)])
        self.line_n=nn.ModuleList([MaxPoolDownSample(3) for i in range(layer+1)])
        self.line_e=nn.ModuleList([MaxPoolDownSample(116) for i in range(layer+1)])
        self.clase=nn.Sequential(nn.Linear(116*4*(self.layer+1),1024),nn.Dropout(0.2),nn.ReLU(),
                                   nn.Linear(1024,2))

    def forward(self,X,Z):
        # X: bs*N*n_features
        # Z: bs8N*N
        n=X.size()[0]
        XX=self.line_n[0](X).view(n,-1)
        ZZ=self.line_e[0](Z).view(n,-1)
        for i in range(self.layer):
            A=self.atten[i](Z,X)
            Z1=self.edge_w[i](torch.matmul(A,Z))
            Z=self.celu(self.norm_e[i](Z1))+Z
            ZZ=torch.cat((ZZ,self.line_e[i+1](Z).view(n,-1)),dim=1)
            X1=self.node_w[i](torch.matmul(A,X))
            X=self.celu(self.norm_n[i](X1))+X
            XX=torch.cat((XX,self.line_n[i+1](X).view(n,-1)),dim=1)
        XZ=torch.cat((XX,ZZ),1)
        y=self.clase(XZ)
        return y
