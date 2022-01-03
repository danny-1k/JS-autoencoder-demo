import torch
import torch.nn as nn
from torch.optim import Adam
from model import Autoencoder
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]
)

trainmnist = datasets.FashionMNIST('~/',train=True,transform=trans)
testmnist = datasets.FashionMNIST('~/',train=False,transform=trans)

train_loader = DataLoader(trainmnist,batch_size=64)
test_loader = DataLoader(testmnist,batch_size=64)

model = Autoencoder()

epochs = 30
lr = 3e-4
loss_fn = nn.MSELoss()
optim = Adam(model.parameters(),lr=lr)
print('Started training!')
best_loss = 100000
total_train = []
total_test = []
for e in range(epochs):
    running_train_loss = []
    running_test_loss = []
    for x,_ in train_loader:
        x = x.view(-1,28*28)
        p = model(x)[0]
        loss = loss_fn(p,x)
        running_train_loss.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        for x,_ in test_loader:
            x = x.view(-1,28*28)
            p = model(x)[0]
            loss = loss_fn(p,x)
            running_test_loss.append(loss.item())
    train_loss = torch.mean(torch.Tensor(running_train_loss))
    test_loss = torch.mean(torch.Tensor(running_test_loss))
    total_train.append(train_loss)
    total_test.append(test_loss)
    utils.save_loss_plot(total_train,total_test)
     
    print(f'Epoch : {e+1} Train-Loss : {train_loss:.5f} Test-Loss : {test_loss:.5f}')
    if test_loss < best_loss:
        torch.save(model, f'saved/model.pt')

print('Done training!')