import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from models import BaseVAE
import math


class VanillaVAE(BaseVAE):

    def __init__(self, encoder_configs, decoder_configs, fc_configs, latent_dim):
        
        super(VanillaVAE, self).__init__()

        self.enc_layers = encoder_configs
        self.dec_layers = decoder_configs
        self.fc_layer = fc_configs
        self.latent_dim = latent_dim
        
        encoder = []
        
        for i in range(len(self.enc_layers)):
            layer = self.enc_layers[i]
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            kernel_size = layer["kernel_size"]
            stride = layer["stride"]
            padding = layer["padding"]
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size= kernel_size,
                            stride= stride, 
                            padding =  padding),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU()
                )
            ) 
        self.encoder = nn.Sequential(*encoder)

        if self.fc_layer["out_channels"] != self.latent_dim:
            raise(RuntimeError("latent dim error"))

        self.fc_mu = nn.Linear(self.fc_layer["in_channels"], self.fc_layer["out_channels"])
        self.fc_var = nn.Linear(self.fc_layer["in_channels"], self.fc_layer["out_channels"])


        decoder = []
        self.z_to_dec = nn.Linear(self.fc_layer["out_channels"], self.fc_layer["in_channels"])

        for i in range(len(self.dec_layers)-1):
            layer = self.dec_layers[i]
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            kernel_size = layer["kernel_size"]
            stride = layer["stride"]
            padding = layer["padding"]
            output_padding = layer["output_padding"]
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = padding,
                                    output_padding =output_padding),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*decoder)
        
        last_layer = self.dec_layers[-1]
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels= last_layer["in_channels"],
                    out_channels= last_layer["out_channels"],
                    kernel_size = last_layer["kernel_size"],
                    stride = last_layer["stride"],
                    padding = last_layer["padding"]),
            nn.Tanh()
        )


    def encode(self, input):
        batch_size = input.size()[0]
        out = self.encoder(input)
        out = out.view(batch_size, -1)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)

        return [mu, log_var]
   
    def decode(self, input):
        C = self.dec_layers[0]["in_channels"]
        H = int(math.sqrt(self.fc_layer["in_channels"] // C ))
        W = int(math.sqrt(self.fc_layer["in_channels"] // C ))
        out = self.z_to_dec(input)
        out = out.view(-1, C, H, W)
        out = self.decoder(out)
        out = self.last_layer(out)
        return out
        

    def reparameterize(self, mu, log_var):
        #samplings
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (std * eps)
    
    def forward(self, input):
        mu, log_var = self.encode(input)
        z =  self.reparameterize(mu, log_var)
        out = self.decode(z)
    
        return [out, input, mu, log_var]
    
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim)  #size : num_samples X latent_dim
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def generate(self, data, device, dim):
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        
        z_list0 = []
        # z_list1 = []
        # z_list2 = []

        z_list0.append(self.decode(z))
        # z_list1.append(self.decode(z))
        # z_list2.append(self.decode(z))

        for i in range(11):
            z_i = z.clone()
            z_i[0, dim] = (i-5)/1.5
            z_list0.append(self.decode(z_i))
            
        # for j in range(11):
        #     z_j = z.clone()
        #     z_j[0, 1] = (j-5)/1.5
        #     z_list1.append(self.decode(z_j))

        # for k in range(11):
        #     z_k = z.clone()
        #     z_k[0, 2] = (k-5)/1.5
        #     z_list2.append(self.decode(z_k))
        
        zi_concat = torch.cat(z_list0, dim=3)
        # zj_concat = torch.cat(z_list1, dim=3)
        # zk_concat = torch.cat(z_list2, dim=3)
        # print(zi_concat.size())
        return zi_concat
