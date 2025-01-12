import torch
import torch.nn as nn


class EOM(nn.Module):
    def __init__(self, size_in, device):
        super().__init__()
        self.size = size_in
        self.device = device
        mag = torch.empty(self.size).normal_(mean=5,std=1)
        phase = 2*torch.pi*torch.rand(self.size)-torch.pi
        self.mag = nn.Parameter(mag.to(device))
        self.phase = nn.Parameter(phase.to(device))

    def forward(self, x):
        #weights = torch.polar(torch.ones(self.size).to(self.device), self.phase)
        weights = torch.polar(self.mag, self.phase)

        return weights*x
    

class SpectralShaper(nn.Module):
    def __init__(self, size_in, device):
        super().__init__()
        self.size = size_in
        self.device = device
        #mag = torch.empty(self.size//2).normal_(mean=1,std=0.5)
        phase = 2*torch.pi*torch.rand(self.size//2)-torch.pi
        #self.mag = nn.Parameter(mag.to(device))
        self.phase = nn.Parameter(phase.to(device))

    def forward(self, x):
        weights = torch.polar(torch.ones(self.size//2).to(self.device), self.phase)
        #weights = torch.polar(self.mag, self.phase)
        weights2 = torch.cat((weights, weights.flip(0)), 0)

        return torch.fft.ifft(weights2*torch.fft.fft(x))
    

class SA(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.1
        self.I0 = 1

    def forward(self, x):
        x_fft = torch.fft.fft(x)
        return torch.fft.ifft(x_fft*(1-self.alpha/(1+torch.abs(x_fft)/self.I0)/torch.abs(x_fft)))
    

class Transpose(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    return torch.transpose(x,0,1)
  

class SelectTopN(nn.Module):
  def __init__(self,n,n_dim):
    super().__init__()
    self.n = n
    self.n_dim = n_dim

  def forward(self,x):
    return x[:,self.n_dim//2-self.n//2:self.n_dim//2+self.n//2]
    

class MyComplexCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, size_avg = None):
      loss = nn.CrossEntropyLoss(reduce = size_avg)
      #loss = nn.MSELoss(reduce = size_avg)
      #loss = nn.L1Loss(reduce = size_avg)
      #targets2 = torch.zeros(size=inputs.size()).to(device)
      #print(torch.transpose(targets,0,1).size())
      #idx = torch.stack((torch.arange(inputs.size(0)).to(device),targets),1)
      #print(idx)
      #targets2[idx] = 1
      if torch.is_complex(inputs):
        real_loss = loss(inputs.real, targets)
        abs_loss = loss(inputs.abs()**2, targets)
        return abs_loss
      else:
        return loss(inputs, targets)


class Model(nn.Module):
    def __init__(self, size_in, size_out, num_layers, device):
        super().__init__()
        layer = [EOM(size_in, device), SpectralShaper(size_in, device), SA()]
        layers = layer*num_layers
        self.network = nn.Sequential(
           *layers,
           SelectTopN(size_out, size_in)
        )

    def forward(self, x):
       return self.network(x)