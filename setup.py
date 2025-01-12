import torch
import torch.optim as optim

from dataLoader import get_loader
from PNN_classes import *

# Check for CUDA
if torch.cuda.is_available():
  print('CUDA is available. GPU will be used.')
  device = torch.device('cuda')
else:
  print('CUDA is not available. CPU will be used.')
  device = torch.device('cpu')


# training parameters
n_epochs = 300
batch_size_train = 64
batch_size_test = 274
learning_rate = 0.001
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# get data loaders
#vowel_types=['ae', 'ah', 'aw', 'eh', 'er', 'iy', 'uw']
vowel_types = ['ah', 'uw', 'iy']
train_loader, test_loader, len_sound = get_loader(batch_size_train, batch_size_test, sr=8000,  vowel_types=vowel_types)

# define model
network = Model(len_sound, len(vowel_types)+1, 3, device)


# define optimizer
optimizer = optim.Adam(network.parameters(),lr=learning_rate)
#optimizer = optim.SGD(network.parameters(),lr = learning_rate, momentum=momentum)

# record stats
train_losses = []
train_counter = []
test_losses = []
test_acc = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = network(data.to(torch.cfloat))
    criterion = MyComplexCrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    pred = torch.abs(output).data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).sum()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: ({:.0f}%)'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item(),100. * correct / len(target)))
    train_losses.append(loss.item())
    train_counter.append(
      (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = network(data.to(torch.cfloat))
      criterion = MyComplexCrossEntropyLoss()
      test_loss += criterion(output, target, size_avg=False).sum().item()
      pred = torch.abs(output).data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  test_acc.append(correct.to('cpu'))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()