import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self,encoder=None,decoder=None):
        super().__init__()
        if encoder == None:
            self.encoder = nn.Sequential(
				nn.Linear(28*28,512),
				nn.LeakyReLU(0.01),
				nn.Linear(512,256),
				nn.LeakyReLU(0.01),
				nn.Linear(256,128),
    			nn.LeakyReLU(0.01),
				nn.Linear(128,64),
    			nn.LeakyReLU(0.01),
				nn.Linear(64,10),
			)
        else:
            self.encoder = encoder
        if decoder == None:
            self.decoder = nn.Sequential(
				nn.Linear(10,64),
    			nn.LeakyReLU(0.01),	
				nn.Linear(64,128),
				nn.LeakyReLU(0.01),
				nn.Linear(128,256),
				nn.LeakyReLU(0.01),
				nn.Linear(256,512),
    			nn.LeakyReLU(0.01),
				nn.Linear(512,28*28),
			)
        else:
            self.decoder = decoder
    def forward(self,x):
        x = self.encoder(x)
        latent = x
        x = self.decoder(x)
        return x,latent
        