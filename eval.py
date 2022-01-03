import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Autoencoder

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]
)

model = Autoencoder()
model.load_state_dict(torch.load('saved/model.pt').state_dict())
model.eval()
testmnist = datasets.FashionMNIST('~/',train=False,transform=trans)
#test_loader = DataLoader(testmnist,batch_size=1)
real = []
fig = plt.figure(figsize=(35,4))
idx = 0
for image,_ in testmnist:
    if idx == 30:
        break
    else:
        if(idx<=14):
            ax = fig.add_subplot(2,30/2,idx+1)
            real.append(image.reshape(28,28))
            ax.imshow(model(image.reshape(-1,28*28))[0].detach().squeeze().reshape(28,28),cmap='gray')
            ax.set_title(f'Encoder')
        else:
            ax = fig.add_subplot(2,30/2,idx+1)
            ax.imshow(real[idx-15],cmap='gray')
            ax.set_title('Real')
            
    idx +=1

plt.show()