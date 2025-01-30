import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from models import spiking_time_unet, timeseries_unet

class SpikingDiffusionModel:
    def __init__(self, sequence_length, device, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.sequence_length = sequence_length
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.model = spiking_time_unet.Spk_UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 4], attn=[8],num_res_blocks=2, dropout=0.1, timestep=4).to(device)
        # self.model.load_state_dict(torch.load("./models/spiking_unet_noattention_128"))
    
    def q_sample(self, x_0, t):
        """Forward diffusion process"""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()
        
        return (
            sqrt_alphas_cumprod_t.view(-1, 1, 1) * x_0 +
            sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1) * noise,
            noise
        )
    
    def p_sample(self, x_t, t):
        """Reverse diffusion process (single step)"""
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # No noise if t == 0
        noise = torch.randn_like(x_t) if t[0] > 0 else 0
        functional.reset_net(self.model)

        return (
            1 / (1 - betas_t).sqrt() * (
                x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise
            ) + torch.sqrt(betas_t) * noise
        )
    
    def sample(self, n_samples, batch_size, device):
        """Generate new samples in batches to avoid GPU memory issues"""
        self.model.eval()
        samples = []  # List to store all generated samples
        
        # Calculate the number of batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                # Determine the size of the current batch
                current_batch_size = min(batch_size, n_samples - i * batch_size)
                
                # Generate noise for the current batch
                x = torch.randn(current_batch_size, 1, self.sequence_length, device=device)
                
                # Reverse diffusion process for the current batch
                for t in range(self.n_steps - 1, -1, -1):
                    t_batch = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                    x = self.p_sample(x, t_batch)
                # Append the generated samples to the list
                samples.append(x.detach().cpu())  # Move to CPU to free GPU memory
                
                # Clear GPU cache
                # torch.cuda.empty_cache()
        
        # Concatenate all batches into a single tensor
        return torch.cat(samples, dim=0).to(device)  # Move back to GPU if needed
    
    def train_step(self, x_0, optimizer):
        """Single training step"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x_0.device)
        
        # Add noise
        x_t, noise = self.q_sample(x_0, t)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Calculate loss
        loss = nn.MSELoss()(predicted_noise, noise)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        optimizer.step()
        functional.reset_net(self.model)
        
        return loss.item()

class DiffusionModel:
    def __init__(self, sequence_length, device, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.sequence_length = sequence_length
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.model = timeseries_unet.UNet(T=n_steps, ch=128, ch_mult=[1, 2, 2, 2], attn=[8], num_res_blocks=2, dropout=0.1).to(device)
    
    def q_sample(self, x_0, t):
        """Forward diffusion process"""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()
        
        return (
            sqrt_alphas_cumprod_t.view(-1, 1, 1) * x_0 +
            sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1) * noise,
            noise
        )
    
    def p_sample(self, x_t, t):
        """Reverse diffusion process (single step)"""
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # No noise if t == 0
        noise = torch.randn_like(x_t) if t[0] > 0 else 0
        
        return (
            1 / (1 - betas_t).sqrt() * (
                x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise
            ) + torch.sqrt(betas_t) * noise
        )
    
    def sample(self, n_samples, batch_size, device):
        """Generate new samples in batches to avoid GPU memory issues"""
        self.model.eval()
        samples = []  # List to store all generated samples
        
        # Calculate the number of batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                # Determine the size of the current batch
                current_batch_size = min(batch_size, n_samples - i * batch_size)
                
                # Generate noise for the current batch
                x = torch.randn(current_batch_size, 1, self.sequence_length, device=device)
                
                # Reverse diffusion process for the current batch
                for t in range(self.n_steps - 1, -1, -1):
                    t_batch = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                    x = self.p_sample(x, t_batch)
                
                # Append the generated samples to the list
                samples.append(x.detach().cpu())  # Move to CPU to free GPU memory
                
                # Clear GPU cache
                # torch.cuda.empty_cache()
        # Concatenate all batches into a single tensor
        return torch.cat(samples, dim=0)  # Move back to GPU if needed
    
    def train_step(self, x_0, optimizer):
        """Single training step"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x_0.device)
        
        # Add noise
        x_t, noise = self.q_sample(x_0, t)
        # print('x_t', x_t.shape)
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Calculate loss
        loss = nn.MSELoss()(predicted_noise, noise)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()