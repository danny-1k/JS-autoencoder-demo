import torch
import torchvision.datasets as datasets
from torchvision import transforms

# just gonna do this the lazy way :)

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]
)

testmnist = datasets.FashionMNIST('~/',train=False,transform=trans)


sandal = testmnist[8][0].reshape(28,28).tolist()
sneaker = testmnist[9][0].reshape(28,28).tolist()
trouser = testmnist[2][0].reshape(28,28).tolist()
pullover = testmnist[1][0].reshape(28,28).tolist()
ankle_boot = testmnist[0][0].reshape(28,28).tolist()


open('static/sandal.txt','w').write(str(sandal))
open('static/sneaker.txt','w').write(str(sneaker))
open('static/trouser.txt','w').write(str(trouser))
open('static/pullover.txt','w').write(str(pullover))
open('static/ankle_boot.txt','w').write(str(ankle_boot))