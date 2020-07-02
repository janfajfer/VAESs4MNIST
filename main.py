"""
Variational Autoencoder for the MNIST dataset.

Author: Jan Fajfer

Sources:
- Allen Lee: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
- Carl Doersch: https://arxiv.org/pdf/1606.05908.pdf
"""

# Imports
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets, models, transforms


class CVariationalAutoEncoder(nn.Module):
    def __init__(self):
        """ Initializes the structrue of the VAE """
        super(CVariationalAutoEncoder, self).__init__()
        # Encoder
        self.encoder_net = nn.Sequential(nn.Linear(784, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256),
                                     nn.ReLU())
        self.layer_mean = nn.Linear(256, 2)
        self.layer_var = nn.Linear(256, 2)

        # Decoder
        self.decoder_net = nn.Sequential(nn.Linear(2, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 784),
                                     nn.Sigmoid())

    def encoder(self, x):
        """ Propagates input through the encoder and returns mean and variance vectors """
        h = self.encoder_net(x)
        return self.layer_mean(h), self.layer_var(h)

    def generate_samples(self, mu, log_var):
        """ Generates random samples from a standard distribution """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # random numbers from uniform distribution [0,1) with the same size as std
        z = eps * std + mu
        return z

    def decoder(self, z):
        """ Propagates sample from encoder through the decoder """
        return self.decoder_net(z)

    def forward(self, x):
        """ VAE forward pass """
        mu, log_var = self.encoder(x.flatten(start_dim=1, end_dim=-1))
        z = self.generate_samples(mu, log_var)
        return self.decoder(z), mu, log_var

def load_mnist(batch_size: int):
    """ Loads the dataset """
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def loss_function(recon_x, x, mu, log_var):
    """ Loss function """
    # Reconstruction error
    BCE = nn.functional.binary_cross_entropy(recon_x, x.flatten(start_dim=1, end_dim=-1), reduction='sum')
    # Kullback–Leibler divergence
    KLD = - 0.5 * torch.sum(-torch.exp(log_var) - mu**2 + log_var + 1)
    return BCE + KLD

def train(vae, optimizer, num_epochs, device, train_loader, test_loader):
    """ Train and evaluate VAE """
    for epoch in range(1, num_epochs+1):
        # Training -----------------------------
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        # Evaluation -----------------------------
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon, mu, log_var = vae(data)
                test_loss += loss_function(recon, data, mu, log_var).item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


def generate_images(vae):
    """ Generates images using the trained VAE """
    with torch.no_grad():
        z = torch.randn(8, 2)
        sample = vae.decoder(z)
        save_image(sample.view(8, 1, 28, 28), 'sample.png')

def main():
    # parameters
    batch_size = 100
    num_epochs = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # query for GPU
    print('Using device:', device)

    # build model
    vae = CVariationalAutoEncoder()
    vae.to(device)
    print(vae)

    optimizer = optim.Adam(vae.parameters())

    train_loader, test_loader = load_mnist(batch_size)
    train(vae, optimizer, num_epochs, device, train_loader, test_loader)
    generate_images(vae)

if __name__ == "__main__":
    main()


