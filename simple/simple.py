import torch
from torch import nn
from torch import optim

class simple_model(nn.Module):
    def __init__(self, device):
        super(simple_model,self).__init__()
        self.device = device

        model = nn.Sequential(nn.Linear(2,200),
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
        o = self.model(x)
        return o

def train():
    numepochs = 1000
    net = simple_model(device=torch.device('cuda'))
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    for epoch in range(numepochs):
        # Make a binary operator
        x = torch.randint(0,2,(1000,1,2),dtype=torch.long)
        # XOR solution
        y = torch.Tensor([int(a[0][0].item())^int(a[0][1].item()) for a in x]).to('cuda').long()
    
        #y = y.view(-1,1).long()
        optimizer.zero_grad()
        out = net(x.float()).squeeze()
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        print(torch.cuda.memory_summary)
    # Eval
    print("===\nEVAL TIME\n===")
    with torch.no_grad():
        x =  torch.randint(0,2,(10,2), dtype=torch.long)
        y = torch.FloatTensor([int(a[0].item())^int(a[1].item()) for a in x]).to('cuda').long()
        out = net(x.float())
        pred = torch.argmax(out,dim=1).view(-1)

        print(f"True: {y}")
        print(f"Pred: {pred}")

def main():
    train()

if __name__ == '__main__':
    main()
