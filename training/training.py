import json
import os
import numpy as np
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from training.logger import Logger


class Trainer():

    def __init__(self, device, model, distribution, name, data_loader, batch_size, num_train_imgs, kld_weight, directory , epochs, test_epochs, resume_epochs, restored_model_path, test_model_path, lr, weight_decay, beta1, beta2, milestones, scheduler_gamma, print_freq, sample_freq, model_save_freq, test_dim):
        
        self.device = device
        self.model = model
        self.name = name
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.num_train_imgs = num_train_imgs
        self.kld_weight = batch_size / num_train_imgs
        self.distribution = distribution
        self.num_batchs = num_train_imgs // batch_size

        self.directory = directory
        log_dir = os.path.join(directory, "logs")
        sample_dir = os.path.join(directory, "samples")
        result_dir = os.path.join(directory, "results")
        model_save_dir = os.path.join(directory, "models")

        if not os.path.exists(os.path.join(directory, "logs")):
            os.makedirs(log_dir)
        self.log_dir = log_dir

        if not os.path.exists(os.path.join(directory, "samples")):
            os.makedirs(sample_dir)
        self.sample_dir = sample_dir

        if not os.path.exists(os.path.join(directory, "results")):
            os.makedirs(result_dir)
        self.result_dir = result_dir

        if not os.path.exists(os.path.join(directory, "models")):
            os.makedirs(model_save_dir)
        self.model_save_dir = model_save_dir

        self.epochs = epochs
        self.test_epochs = test_epochs
        self.resume_epochs = resume_epochs

        self.restored_model_path = restored_model_path
        self.test_model_path = test_model_path

        self.lr = lr 
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2

        self.milestones = milestones
        self.scheduler_gamma = scheduler_gamma

        self.print_freq = print_freq
        self.sample_freq = sample_freq
        self.model_save_freq = model_save_freq

        self.test_dim = test_dim

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr = self.lr,
                                        betas = [self.beta1, self.beta2],
                                        weight_decay  = self.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = self.scheduler_gamma)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.milestones, gamma = self.scheduler_gamma)
        
        self.build_tensorboard()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
    
    def load_model(self, path, resume_epochs):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from epoch {}...'.format(resume_epochs))
        path = os.path.join( path , '{}-VAE.pt'.format(resume_epochs))
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        # self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def reconstruction_loss(self, recons, input):
        
        if self.distribution == 'bernoulli':
            rec_loss = nn.BCEWithLogitsLoss()
            return rec_loss(recons, input)
            
        elif self.distribution == 'gaussian':
            rec_loss = nn.MSELoss()
            return rec_loss(recons, input)

        else:
            rec_loss = None
            return rec_loss
        
    
    def KLD_loss(self, mu, log_var):
        # how to get kldweight?
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return self.kld_weight * kld_loss


    def loss_function(self, recons, input, mu, log_var):
        
        if self.name == 'VanillaVAE':
            rec_loss = self.reconstruction_loss(recons, input)
            kld_loss = self.KLD_loss(mu, log_var)
            return rec_loss + kld_loss, [rec_loss.item(), kld_loss.item()]
        elif self.name == 'BetaVAE_H':
            raise NotImplementedError
        elif self.name == 'BetaVAE_B':
            raise NotImplementedError
        else:
            raise(RuntimeError("Model Name Wrong"))

    def train(self):

        # data_iter = iter(self.data_loader)
        # sample_fixed = next(data_iter)
        # sample_fixed.to(self.device)

        start_epoch = 0
        if self.resume_epochs > 0:
            start_epoch = self.resume_epochs
            self.load_model(self.restored_model_path, self.resume_epochs)
            self.model.to(self.device)
        
        print("Start Training...")
        start_time = time.time()
        for i in range(start_epoch, self.epochs):

            for batch_idx, [data, _] in enumerate(self.data_loader):
                
                self.optimizer.zero_grad()
                data = data.to(self.device)
                recons, input, mu, log_var = self.model(data)

                loss, item = self.loss_function(recons, input, mu, log_var)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                loss_item={}
                loss_item["rec_loss"] = item[0]
                loss_item["kld_loss"] = item[1]

            
                
                if batch_idx % self.print_freq == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}/{}], Iteration[{}/{}]".format(et, i+1, self.epochs, batch_idx, self.num_batchs)
                    for tag, value in loss_item.items():
                        log += ", {}: {:4f}".format(tag, value)
                    print(log)

                    for tag, value in loss_item.items():
                        self.logger.scalar_summary(tag, value, (self.num_train_imgs // self.batch_size)*(i)+ (batch_idx))
                
                if batch_idx % self.sample_freq == 0:
                    with torch.no_grad():
                        samples = self.model.sample(self.batch_size, self.device)
                        
                        sample_path = os.path.join(self.sample_dir, "{}_{}-sample.jpg".format(i+1, batch_idx))
                        save_image(self.denorm(samples.cpu()), sample_path, nrow=self.batch_size // 8, padding =0)
                        print('Saved samples into {}...'.format(sample_path))
            

            if i % self.model_save_freq == 0:
                model_path = os.path.join(self.model_save_dir, "{}-VAE.pt".format(i+1))
                torch.save({
                    'epoch': i,
                    'model': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler' : self.scheduler.state_dict()
                }, model_path)
                # torch.save(self.model.state_dict(), model_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
        
        model_path = os.path.join(self.model_save_dir, "{}-VAE.pt".format(i+1))
        torch.save(self.model.state_dict(), model_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def test(self):

        self.load_model(self.test_model_path, self.test_epochs)
        self.model.to(self.device)
        
        if not os.path.exists(os.path.join(self.result_dir, self.test_dim)):
            os.makedirs(os.path.join(self.result_dir, self.test_dim))

        with torch.no_grad():
            for batch_idx, [data, _] in enumerate(self.data_loader):
                data = data.to(self.device)
                out = self.model.generate(data, self.device, dim= self.test_dim)
                result_path = os.path.join(os.path.join(self.result_dir, self.test_dim), '{}-out.jpg'.format(batch_idx))             
                save_image(self.denorm(out), result_path, nrow=1, padding=0)
                