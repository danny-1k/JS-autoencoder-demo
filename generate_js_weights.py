import torch
from model import Autoencoder

net = Autoencoder()
net.load_state_dict(torch.load('saved/model.pt').state_dict())
net.eval()



for sub in ['encoder','decoder']:
    for i in range(0,9,2):

        weights = eval(f'net.{sub}[{i}].weight.T.tolist()')
        bias = eval(f'net.{sub}[{i}].bias.tolist()')

        f = open(f'static/{sub}/w{i}.txt','w').write(str(weights))

        f = open(f'static/{sub}/b{i}.txt','w').write(str(bias))