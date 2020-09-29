import subprocess as sp
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from torch.utils.checkpoint import checkpoint

class every_model(nn.Module):
    def __init__(self, device):
        super(every_model,self).__init__()
        self.device = device
        self.m0 = nn.Sequential(nn.Linear(2,200),
                              nn.ReLU(),
                              ).to(device)
        self.m1 = nn.Sequential(nn.Linear(200,500),
                              nn.ReLU(),
                              ).to(device)
        self.m2 = nn.Sequential(nn.Linear(500,1000),
                              nn.ReLU(),
                              ).to(device)
        # Use ~~3~~8 times
        self.m3 = nn.Sequential(nn.Linear(1000,1000),
                              nn.ReLU(),
                              ).to(device)
        self.m4 = nn.Sequential(nn.Linear(1000,500),
                              nn.ReLU(),
                              ).to(device)
        self.m5 = nn.Sequential(nn.Linear(500,200),
                              nn.ReLU(),
                              ).to(device)
        self.m6 = nn.Sequential(nn.Linear(200,100),
                              nn.ReLU(),
                              ).to(device)
        self.m7 = nn.Sequential(nn.Linear(100,50),
                              nn.ReLU(),
                              ).to(device)
        self.m8 = nn.Sequential(nn.Linear(50,2),
                              nn.ReLU(),
                              nn.Sigmoid(),
                              ).to(device)

    def forward(self, x):
        x = x.to(self.device)
        x.requires_grad=True
        o = checkpoint(self.m0, x)
        o = checkpoint(self.m1, o)
        o = checkpoint(self.m2, o)
        for i in range(8):
            o = checkpoint(self.m3, o)
        o = checkpoint(self.m4, o)
        o = checkpoint(self.m5, o)
        o = checkpoint(self.m6, o)
        o = checkpoint(self.m7, o)
        o = checkpoint(self.m8, o)
        return o

class eo_model(nn.Module):
    def __init__(self, device):
        super(eo_model,self).__init__()
        self.device = device
        self.m0 = nn.Sequential(nn.Linear(2,200),
                              nn.ReLU(),
                              nn.Linear(200,500),
                              nn.ReLU(),
                              ).to(device)
        self.m1 = nn.Sequential(nn.Linear(500,1000),
                              nn.ReLU(),
                              nn.Linear(1000,1000),
                              nn.ReLU(),
                              ).to(device)
        # Use 3 times
        self.m2 = nn.Sequential(nn.Linear(1000,1000),
                              nn.ReLU(),
                              nn.Linear(1000,1000),
                              nn.ReLU(),
                              ).to(device)
        self.m3 = nn.Sequential(nn.Linear(1000,1000),
                              nn.ReLU(),
                              nn.Linear(1000,500),
                              nn.ReLU(),
                              ).to(device)
        self.m4 = nn.Sequential(nn.Linear(500,200),
                              nn.ReLU(),
                              nn.Linear(200,100),
                              nn.ReLU(),
                              ).to(device)
        self.m5 = nn.Sequential(nn.Linear(100,50),
                              nn.ReLU(),
                              nn.Linear(50,2),
                              nn.ReLU(),
                              nn.Sigmoid(),
                              ).to(device)


    def forward(self,x):
        x = x.to(self.device)
        x.requires_grad=True
        o = checkpoint(self.m0.to(self.device), x)
        o = checkpoint(self.m1.to(self.device), o)
        for i in range(3):
            o = checkpoint(self.m2.to(self.device), o)
        o = checkpoint(self.m3.to(self.device), o)
        o = checkpoint(self.m4.to(self.device), o)
        o = checkpoint(self.m5.to(self.device), o)
        return o

class three_block_model(nn.Module):
    def __init__(self, device):
        super(three_block_model, self).__init__()
        self.device = device
        self.m0 = nn.Sequential(nn.Linear(2,200),
                                nn.ReLU(),
                                nn.Linear(200,500),
                                nn.ReLU(),
                                nn.Linear(500,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                ).to(device)
        self.m1 = nn.Sequential(nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                ).to(device)
        self.m2 = nn.Sequential(nn.Linear(1000,500),
                                nn.ReLU(),
                                nn.Linear(500,200),
                                nn.ReLU(),
                                nn.Linear(200,100),
                                nn.ReLU(),
                                nn.Linear(100,50),
                                nn.ReLU(),
                                nn.Linear(50,2),
                                nn.ReLU(),
                                nn.Sigmoid(),
                                ).to(device)

    def forward(self, x):
        x = x.to(self.device)
        x.requires_grad=True
        o = checkpoint(self.m0,x)
        o = checkpoint(self.m1, o)
        o = checkpoint(self.m2, o)
        return o

class two_block_model(nn.Module):
    def __init__(self, device):
        super(two_block_model, self).__init__()
        self.device = device
        self.m0 = nn.Sequential(nn.Linear(2,200),
                                nn.ReLU(),
                                nn.Linear(200,500),
                                nn.ReLU(),
                                nn.Linear(500,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                ).to(device)
        self.m1 = nn.Sequential(nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,1000),
                                nn.ReLU(),
                                nn.Linear(1000,500),
                                nn.ReLU(),
                                nn.Linear(500,200),
                                nn.ReLU(),
                                nn.Linear(200,100),
                                nn.ReLU(),
                                nn.Linear(100,50),
                                nn.ReLU(),
                                nn.Linear(50,2),
                                nn.ReLU(),
                                nn.Sigmoid(),
                                ).to(device)

    def forward(self, x):
        x = x.to(self.device)
        x.requires_grad=True
        o = checkpoint(self.m0,x)
        o = checkpoint(self.m1, o)
        return o

def train(m=0):
    numepochs = 2
    if m is 0:
        net = every_model(device=torch.device('cuda')).half()
    if m is 1:
        net = eo_model(device=torch.device('cuda')).half()
    elif m is 2:
        net = three_block_model(device=torch.device('cuda')).half()
    elif m is 3:
        net = two_block_model(device=torch.device('cuda')).half()
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    for epoch in range(numepochs):
        # Make a binary operator
        x = torch.randint(0,2,(100000,1,2),dtype=torch.float)#, requires_grad=True)
        # XOR solution
        y = torch.Tensor([int(a[0][0].item())^int(a[0][1].item()) for a in x]).to('cuda').long()
    
        #y = y.view(-1,1).long()
        optimizer.zero_grad()
        out = net(x.half()).squeeze()
        #out = out.squeeze()
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        print(f"Allocated: {torch.cuda.memory_allocated()//2**20} MB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated()//2**20} MB")
        smi = int(sp.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],encoding='utf-8').split('\n')[0])
        print(f"SMI: {smi} MB")
    # Eval
    #   with torch.no_grad():
    #       x =  torch.randint(0,2,(10,2), dtype=torch.long)
    #       y = torch.FloatTensor([int(a[0].item())^int(a[1].item()) for a in x]).to('cuda').long()
    #       out = net(x.float())
    #       pred = torch.argmax(out,dim=1).view(-1)

    #       #print(f"True: {y}")
    #       #print(f"Pred: {pred}")
    #   #print(f"Max memory {torch.cuda.max_memory_allocated()/1000000} MB")
    print(f"Max memory {torch.cuda.max_memory_allocated()/1024**2} MB")

def main():
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"Every Checkpoint")
    train(m=0)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"Every other model")
    train(m=1)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"Three Checkpoints")
    train(m=2)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"Two Checkpoints")
    train(m=3)
if __name__ == '__main__':
    main()
