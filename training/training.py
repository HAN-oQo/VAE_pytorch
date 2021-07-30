import json
import os
import numpy as np
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from training.logger import Logger


class Trainer():

    def __init__(self, device, model, distribution, name, data_loader, batch_size, num_train_imgs, kld_weight, directory , max_iters, resume_iters, capacity_iters, restored_model_path, beta, gamma, max_capacity, loss_type, lr, weight_decay, beta1, beta2, milestones, scheduler_gamma, print_freq, sample_freq, model_save_freq, test_iters, test_dim, test_seed, start, end, steps):
        
        self.device = device
        self.model = model
        self.distribution = distribution
        self.name = name

        self.data_loader = data_loader
        self.batch_size = batch_size
        
        self.num_train_imgs = num_train_imgs
        self.kld_weight = batch_size / num_train_imgs
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
        self.test_model_path = model_save_dir

        self.max_iters = max_iters
        self.resume_iters = resume_iters
        if self.resume_iters > 0:
            self.global_iters = resume_iters
        self.global_iters = 0
        self.capacity_iters = capacity_iters

        self.restored_model_path = restored_model_path

        self.beta = beta
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.C_max = torch.Tensor([max_capacity])
        self.loss_type = loss_type

        self.lr = lr 
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2

        self.milestones = milestones
        self.scheduler_gamma = scheduler_gamma

        self.print_freq = print_freq
        self.sample_freq = sample_freq
        self.model_save_freq = model_save_freq

        self.test_iters = test_iters
        self.test_dim = test_dim
        self.test_seed = test_seed
        self.start = start
        self.end = end
        self.steps = steps

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
    
    def load_model(self, path, resume_iters):
        """Restore the trained generator and discriminator."""
        resume_iters = int(resume_iters)
        print('Loading the trained models from iters {}...'.format(resume_iters))
        path = os.path.join( path , '{}-VAE.pt'.format(resume_iters))
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_iters = checkpoint['iters']
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
        return kld_loss


    def loss_function(self, recons, input, mu, log_var):
       
        if self.name == 'VanillaVAE':
            rec_loss = self.reconstruction_loss(recons, input)
            kld_loss = self.KLD_loss(mu, log_var)
            return rec_loss + self.kld_weight * kld_loss, [rec_loss.item(), kld_loss.item()]
        elif self.name == 'BetaVAE':
            if self.loss_type == 'B':
                # Understanding disentangling in β-VAE
                rec_loss = self.reconstruction_loss(recons, input)
                self.C_max = self.C_max.to(self.device)
                C = torch.clamp(self.C_max/self.capacity_iters*self.global_iters, 0, self.C_max.data[0])
                kld_loss = self.KLD_loss(mu, log_var)
                return rec_loss + self.gamma * self.kld_weight * (kld_loss - C).abs(), [rec_loss.item(), kld_loss.item()]
            elif self.loss_type == 'H':
                # beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework 
                rec_loss = self.reconstruction_loss(recons, input)
                kld_loss = self.KLD_loss(mu, log_var)
                return rec_loss + self.beta * self.kld_weight * kld_loss, [rec_loss.item(), kld_loss.item()]


        else:
            raise(RuntimeError("Model Name Wrong"))

    def train(self):

        # data_iter = iter(self.data_loader)
        # sample_fixed = next(data_iter)
        # sample_fixed.to(self.device)

        if self.resume_iters > 0:
            self.load_model(self.restored_model_path, self.resume_iters)
            self.model.to(self.device)
        
        data_iter = iter(self.data_loader)

        print("Start Training...")
        start_time = time.time()
        while self.global_iters <= self.max_iters:
            try:
                data, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                data, _ = next(data_iter)

            self.global_iters += 1

            self.optimizer.zero_grad()

            data = data.to(self.device)
            recons, input, mu, log_var = self.model(data)
            loss, item = self.loss_function(recons, input, mu, log_var)
            loss.backward()

            self.optimizer.step()
            
            loss_item={}
            loss_item["rec_loss"] = item[0]
            loss_item["kld_loss"] = item[1]
            
            if self.global_iters % self.print_freq == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration[{}/{}]".format(et, self.global_iters, self.max_iters)
                for tag, value in loss_item.items():
                    log += ", {}: {:4f}".format(tag, value)
                print(log)

                for tag, value in loss_item.items():
                    self.logger.scalar_summary(tag, value, self.global_iters)
            
            if self.global_iters % self.sample_freq == 0:
                with torch.no_grad():
                    samples = self.model.sample(self.batch_size, self.device)
                    
                    sample_path = os.path.join(self.sample_dir, "{}-sample.jpg".format(self.global_iters))
                    save_image(self.denorm(samples.cpu()), sample_path, nrow=self.batch_size // 10, padding =0)
                    print('Saved samples into {}...'.format(sample_path))
        
            # self.scheduler.step()

            if self.global_iters % self.model_save_freq == 0:
                model_path = os.path.join(self.model_save_dir, "{}-VAE.pt".format(self.global_iters))
                torch.save({
                    'iters': self.global_iters,
                    'model': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler' : self.scheduler.state_dict()
                }, model_path)
                # torch.save(self.model.state_dict(), model_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
        

        model_path = os.path.join(self.model_save_dir, "{}-VAE.pt".format(self.global_iters))
        torch.save(self.model.state_dict(), model_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def test(self):

        self.load_model(self.test_model_path, self.test_iters)
        self.model.to(self.device)
        
        with torch.no_grad():
            # -----------------------------------------------------------
            # walking specific latent dimension
            
            # -----------------------------------------------------------
            # for batch_idx, [data, _] in enumerate(self.data_loader):
            #     data = data.to(self.device) 
                
            #     mu, log_var = self.model.encode(data)
            #     z = self.model.reparameterize(mu, log_var)
            #     if len(self.test_dim) == 1:
            #         if not os.path.exists(os.path.join(self.result_dir, str(self.test_dim))):
            #             os.makedirs(os.path.join(self.result_dir, str(self.test_dim)))
            #         out = self.model.generate(z,  self.device, dim= self.test_dim)
            #         result_path = os.path.join(os.path.join(self.result_dir, str(self.test_dim)), '{}-out.jpg'.format(batch_idx))             
            #         save_image(self.denorm(out.cpu()), result_path, nrow=1, padding=0)
            #         print("Saved result images {}-out.jpg into {}...".format(batch_idx, result_path))
            #     else:
            #         for ind in range(len(self.test_dim) -1):
            #             t_dim = [self.test_dim[ind], self.test_dim[ind+1]]
            #             if not os.path.exists(os.path.join(self.result_dir, str(t_dim))):
            #                 os.makedirs(os.path.join(self.result_dir, str(t_dim)))
            #             out = self.model.generate(z,  self.device, dim= t_dim)
            #             result_path = os.path.join(os.path.join(self.result_dir, str(t_dim)), '{}-out.jpg'.format(batch_idx))             
            #             save_image(self.denorm(out.cpu()), result_path, nrow=1, padding=0)
            #             print("Saved result images {}-out.jpg into {}...".format(batch_idx, result_path))
                
            #     if batch_idx == 60:
            #         break


            # -----------------------------------------------------------------------
            # latent traversal for all dmensions
            #
            # -----------------------------------------------------------------------
            walking_result_dir = os.path.join(self.result_dir, 'walking')
            if not os.path.exists(walking_result_dir):
                os.makedirs(walking_result_dir)
            
            latent_dim = self.model.latent_dim

            random_seeds= self.test_seed
            z_list = []
            for seed in random_seeds:
                np.random.seed(seed)
                z = np.random.normal(size = latent_dim)
                z = np.float32(z)
                z = torch.tensor(z)
                z_list.append(z)
                        
            test_latents = torch.stack(z_list, dim=0)
            for d in range(latent_dim):
                out = self.model.traverse_latents(self.device, test_latents, d, start = self.start, end = self.end, steps = self.steps)
                result_path = os.path.join(walking_result_dir, 'dim-{}.jpg'.format(d))
                save_image(self.denorm(out.cpu()), result_path, nrow=10, padding = 0)
                print("Saved result images dim-{}.jpg into {}...".format(d, result_path))
                

