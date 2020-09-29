import subprocess as sp
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.checkpoint import checkpoint

class simple_model(nn.Module):
    def __init__(self, device, checkpoint=False, segments=2):
        super(simple_model,self).__init__()
        self.device = device
        self.checkpoint = checkpoint
        self.segments = segments
        model = nn.Sequential(nn.Linear(2,200),
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
                              nn.Linear(1000,1000),
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
                              )

        self.model = model.to(device)

    def forward(self,x):
        x = x.to(self.device)
        x.requires_grad=True
        if self.checkpoint:
            o = checkpoint_sequential(self.model, self.segments, x)
        else:
            o = self.model(x)
        return o

class simple2(simple_model):
    def __init__(self, device, checkpoint=False, segments=2):
        super(simple2,self).__init__()
        self.device = device
        self.checkpoint = checkpoint
        self.segments = segments

    def forward(self,x):
        x = x.to(self.device)
        x.requires_grad=True
        if self.checkpoint:
            o = checkpoint

def train(checkpoint=False, segments=None):
    numepochs = 2
    net = simple_model(device=torch.device('cuda'), checkpoint=checkpoint,
            segments=segments)
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
        out = net(x).squeeze()
        #out = out.squeeze()
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        print(f"Allocated: {torch.cuda.memory_allocated()//2**20} MB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated()//2**20} MB")
        smi = int(sp.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],encoding='utf-8').split('\n')[0])
        print(f"SMI: {smi} MB")
    # Eval
    with torch.no_grad():
        x =  torch.randint(0,2,(10,2), dtype=torch.long)
        y = torch.FloatTensor([int(a[0].item())^int(a[1].item()) for a in x]).to('cuda').long()
        out = net(x.float())
        pred = torch.argmax(out,dim=1).view(-1)

        #print(f"True: {y}")
        #print(f"Pred: {pred}")
    #print(f"Max memory {torch.cuda.max_memory_allocated()/1000000} MB")
    print(f"Max memory {torch.cuda.max_memory_allocated()/1024**2} MB")

def main():
    #torch.cuda.empty_cache()
    #torch.cuda.reset_max_memory_allocated()
    #print(f"No Segments")
    #train()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"2 Segments")
    train(checkpoint=True, segments=2)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"4 Segments")
    train(checkpoint=True, segments=4)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"8 Segments")
    train(checkpoint=True, segments=8)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"10 Segments")
    train(checkpoint=True, segments=10)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print(f"16 Segments")
    train(checkpoint=True, segments=16)

if __name__ == '__main__':
    main()
