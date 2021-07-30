import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.activation import LeakyReLU
from models import BaseVAE
import math
import numpy as np


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
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init()

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

    def generate(self, z, device, dim):
        
        batch_size, _  = z.size()
        if len(dim) == 1:
            # batch_size, _ , _ , _ = data.size()
            # mu, log_var = self.encode(data)
            # z = self.reparameterize(mu, log_var)
            dim = dim[0]
            z_list0 = []
            z_list0.append(self.decode(z))        
            for i in range(11):
                z_i = z.clone()
                z_i[ : , dim] += (i-5)/2
                z_list0.append(self.decode(z_i))
                        
            zi_concat = torch.cat(z_list0, dim=3)
            return zi_concat
        elif len(dim) == 2:
            dim1 = dim[0]
            dim2 = dim[1]

            if batch_size != 1:
                return(RuntimeError("Batch Size should be 1 when you test 2 dimensions."))
            # mu, log_var = self.encode(data)
            # z = self.reparameterize(mu, log_var)

            z_list0 = []
            for i in range(11):
                z_list1 = []
                z_i = z.clone()
                z_i[0,dim1] += (i-5)/3
                for j in range(11):
                    z_j = z_i.clone()
                    z_j[0,dim2] += (j-5)/3
                    z_list1.append(self.decode(z_j)) 
                    zj_concat = torch.cat(z_list1, dim=3)
                z_list0.append(zj_concat)
        
            zi_concat = torch.cat(z_list0, dim=2)
            return zi_concat
        else:
            raise NotImplementedError            
    
    def traverse_latents(self, device, latents, dim, start=-3.0, end=3.0, steps= 10):
        latent_num, _ = latents.size()
        
        interpolation = torch.linspace(start= start, end =end, steps = steps)
        traversal_vectors = torch.zeros(latent_num*interpolation.size()[0], self.latent_dim)
        for i in range(latent_num):
            z_base = latents[i].clone()
            traversal_vectors[i*steps:(i+1)*steps, :] = z_base
            traversal_vectors[i*steps:(i+1)*steps, dim] = interpolation
        
        traversal_vectors = traversal_vectors.to(device)
        out = self.decode(traversal_vectors)
        # out_concat = torch.cat(out, dim=3)
        return out


        # random_seeds= [42, 62, 1024, 72, 92]

        # z_list = []
        # for seed in random_seeds:
        #     np.random.seed(seed)
        #     z = np.random.normal(size = self.latent_dim)
        #     z = np.float32(z)
        #     z = torch.tensor(z)
        #     z_list.append(z)
        
        # latents = torch.stack(z_list, dim=0)
        # print(latents.size())



    
###
# Weight Initialization
# 
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

###