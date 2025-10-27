# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 14:00:35 2025

@author: Xinyu Cao
"""

import torch
import gpytorch
from gpytorch.kernels import Kernel
from botorch.models.gpytorch import GPyTorchModel
import plotly.graph_objects as go
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")
        
     
        
"""
model_type="RBF_zeroprior"/"RBF_nonzeroprior"/"reconstructed"/"smooth_reconstructed"/"memory_enhanced"
return trained GP model
"""

def train_GP(input_data,output_data,model_type,print_flag=True,**kwargs):
    
    machanistic_fun=kwargs.get('machanistic_fun', None)
    alpha=kwargs.get('alpha', 0.8)
    backward_step=kwargs.get('backward_step', 4)
    initial_sampling_point=kwargs.get('initial_sampling_point', 5)
    decay_factor=kwargs.get('decay_factor', 0.9)
    experiment_num=max(0,input_data.shape[0]-initial_sampling_point)
    decay_factor=decay_factor**experiment_num
    
    
    if print_flag:
        print("Configuration Parameters:")
        print(f"  machanistic_fun: {machanistic_fun}")
        print(f"  alpha: {alpha}")
        print(f"  backward_step: {backward_step}")
        print(f"  initial_sampling_point: {initial_sampling_point}")
        print(f"  model_type: {model_type}")
        print(f"  decay_factor: {decay_factor}")
        
        
    if model_type=="RBF_zeroprior":
        model, likelihood = train_RBF_zeroprior_gp(input_data, output_data,print_flag)
        
    elif model_type=="RBF_nonzeroprior":
        model, likelihood = train_RBF_nonzeroprior_gp(input_data, output_data, print_flag,machanistic_fun)
        
    elif model_type=="reconstructed":
        model, likelihood = train_reconstructed_gp(input_data, output_data,print_flag, machanistic_fun,decay_factor=decay_factor)
        
    elif model_type=="smooth_reconstructed":
        model, likelihood = train_memory_enhanced_gp(input_data, output_data, print_flag, machanistic_fun, alpha, 1, initial_sampling_point)
        
    elif model_type=="memory_enhanced":
        model, likelihood = train_memory_enhanced_gp(input_data, output_data,print_flag, machanistic_fun, alpha, backward_step, initial_sampling_point)
    
    return_dict={"model":model,"likelihood":likelihood,"machanistic_fun":machanistic_fun,"model_type":model_type}
    return return_dict
    
    
    
def model_prediction(new_point,model_dict):
    
    trained_model=model_dict["model"]
    likelihood=model_dict["likelihood"]
    
    input_new = torch.tensor(new_point, dtype=torch.float32)
    if input_new.ndim == 1:
        input_new = input_new.unsqueeze(0)
    
    # Ensure model is in eval mode
    trained_model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(trained_model(input_new))
        pred_mean = np.array(pred_dist.mean)
        pred_var = np.array(pred_dist.variance)  
        pred_std = pred_var ** 0.5

    return pred_mean,pred_std


            
class InputDependentWhiteKernel(Kernel):
    def __init__(self, dev_fun, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.dev_fun = dev_fun  # Callable that returns input-dependent noise

    def forward(self, x1, x2, diag=False, **params):
        # Handle batch inputs: x1.shape == (..., N, D)
        # print("x1.shape:",x1.shape)
        if x1.dim() > 2:
           # Batched input: x1.shape == (B, N, D), x2.shape == (B, M, D)
            n = x1.size(-2)
            m = x2.size(-2)
            B = x1.size(0)
            dtype = x1.dtype
            device = x1.device

            noise = self.dev_fun(x1)  # shape: (B, N)
            noise = noise.view(B, n)

            K = torch.zeros(B, n, m, dtype=dtype, device=device)

            for b in range(B):
                x1b = x1[b]  # (N, D)
                x2b = x2[b]  # (M, D)
                noise_b = noise[b]  # (N,)
                eq_mask = (x1b[:, None, :] == x2b[None, :, :]).all(dim=2)  # (N, M)
                match_indices = eq_mask.nonzero(as_tuple=False)  # (K, 2)
                for i, j in match_indices:
                    K[b, i, j] = noise_b[i]
            return K
        
        else:
                noise_x1 = self.dev_fun(x1).view(-1)
                
                # eq_mask: (N, M) bool tensor where x1[i] == x2[j]
                eq_mask = (x1[:, None, :] == x2[None, :, :]).all(dim=2)  # Broadcasting comparison
                
                # find exact position
                match_indices = eq_mask.nonzero(as_tuple=False)  # shape (K, 2)
                
                K = torch.zeros(x1.size(0), x2.size(0), dtype=x1.dtype, device=x1.device)
                for i, j in match_indices:
                    K[i, j] = noise_x1[i]
                return K
        
        
@torch.no_grad()
def log_mll_value(trained_model, trained_likelihood, train_x, train_y):
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(trained_likelihood, trained_model)
    output = trained_model(train_x)
    return mll(output, train_y).item()
  

def train_RBF_zeroprior_gp(input_data, output_data,print_flag, training_iter=100, learning_rate=0.1,para_output=False):
    """
    Train a noise-free Gaussian Process model with:
        - RBF kernel
        - Zero-mean prior
        - Very small fixed observation noise (simulates noise-free data)
    
    Args:
        input_data (array-like): Training inputs of shape [n, d]
        output_data (array-like): Training targets of shape [n] or [n, 1]
        training_iter (int): Number of training iterations
        learning_rate (float): Learning rate for the optimizer
    
    Returns:
        model: Trained GP model (set to eval mode)
        likelihood: Trained likelihood (set to eval mode)
    """

    # Convert input and output to torch tensors
    if isinstance(input_data, torch.Tensor):
        train_x = input_data.clone().detach().float()
    else:
        train_x = torch.tensor(input_data, dtype=torch.float32)
    
    # Convert output_data to tensor and squeeze
    if isinstance(output_data, torch.Tensor):
        train_y = output_data.clone().detach().float().squeeze()
    else:
        train_y = torch.tensor(output_data, dtype=torch.float32).squeeze()


    # Define a Gaussian likelihood with a tiny noise constraint (effectively noise-free)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )

    # Define a GP model with zero mean and RBF kernel
    class ZeroMeanGPModel(gpytorch.models.ExactGP,GPyTorchModel):
        
        _num_outputs = 1
        
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            
            # Use a zero mean function (i.e., GP prior mean = 0)
            self.mean_module = gpytorch.means.ZeroMean()
            
            # Use a scaled RBF kernel as the covariance function
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            
            # Set the likelihood noise to a fixed small value
            self.likelihood.noise = 1e-6
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        def forward(self, x):
            # Compute mean and covariance of the GP
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            # Return multivariate normal distribution
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Instantiate the GP model
    model = ZeroMeanGPModel(train_x, train_y, likelihood)
    
    if print_flag:
        print("---------------------start training RBF with zero prior...------------------------")
        # BEFORE training loop
        print("[Before training] Hyperparameters:")
        print(f"  lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}")
        print(f"  outputscale: {model.covar_module.outputscale.item():.4f}")
        print(f"  noise:       {likelihood.noise.item():.6f}")


    # Set the model and likelihood into training mode
    model.train()
    likelihood.train()

    # Use Adam optimizer to train kernel hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Marginal log-likelihood objective
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    for i in range(training_iter):
        optimizer.zero_grad()  # Reset gradients
        output = model(train_x)  # Forward pass
        loss = -mll(output, train_y)  # Compute negative marginal log likelihood
        loss.backward()  # Backpropagate
        if i % 10 == 0 and print_flag:
            print(f"[Iter {i+1}/{training_iter}] Loss: {loss.item():.4f}")
        optimizer.step()  # Update parameters
        
    if print_flag:
        # AFTER training loop
        print("\n[After training] Hyperparameters:")
        print(f"  lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}")
        print(f"  outputscale: {model.covar_module.outputscale.item():.4f}")
        print(f"  noise:       {likelihood.noise.item():.6f}")
        print("------------------------end training---------------------------")


    # Set model and likelihood into evaluation mode before returning
    if para_output:
        return model.eval(), likelihood.eval(),model.covar_module.base_kernel.lengthscale.item(),model.covar_module.outputscale.item()
    else:
        return model.eval(), likelihood.eval()

        
        
def train_RBF_nonzeroprior_gp(input_data, output_data, print_flag, machanistic_fun, training_iter=100, learning_rate=0.1,para_output=False):
    """
    Train a Gaussian Process model with:
        - RBF kernel
        - Non-zero mean function from machanistic_fun
        - Very small fixed observation noise (simulates noise-free data)
    
    Args:
        input_data (array-like): Training inputs of shape [n, d]
        output_data (array-like): Training targets of shape [n] or [n, 1]
        machanistic_fun (callable): Function that maps torch tensor [n, d] -> [n] or [n, 1]
        training_iter (int): Number of training iterations
        learning_rate (float): Learning rate for the optimizer
    
    Returns:
        model: Trained GP model (set to eval mode)
        likelihood: Trained likelihood (set to eval mode)
    """

    # Convert input and output to torch tensors
    if isinstance(input_data, torch.Tensor):
        train_x = input_data.clone().detach().float()
    else:
        train_x = torch.tensor(input_data, dtype=torch.float32)

    
    if isinstance(output_data, torch.Tensor):
        train_y = output_data.clone().detach().float().squeeze()
    else:
        train_y = torch.tensor(output_data, dtype=torch.float32).squeeze()


    # Define Gaussian likelihood with tiny fixed noise
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )

    # Define GP model with user-provided mean function
    class NonZeroMeanGPModel(gpytorch.models.ExactGP,GPyTorchModel):
        
        _num_outputs = 1
        
        def __init__(self, train_x, train_y, likelihood, mean_function):
            super().__init__(train_x, train_y, likelihood)
            self.mean_function = mean_function
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.likelihood.noise_covar.noise = torch.tensor(1e-6)

        def forward(self, x):
            mean_x = self.mean_function(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Instantiate model
    model = NonZeroMeanGPModel(train_x, train_y, likelihood, machanistic_fun)
    
    if print_flag:

        print("---------------------start training RBF with mechanistic prior...------------------------")
        print("[Before training] Hyperparameters:")
        print(f"  lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}")
        print(f"  outputscale: {model.covar_module.outputscale.item():.4f}")
        print(f"  noise:       {likelihood.noise.item():.6f}")

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer and marginal likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if i % 100 == 0 and print_flag:
            print(f"[Iter {i+1}/{training_iter}] Loss: {loss.item():.4f}")
        optimizer.step()

    if print_flag:
        print("\n[After training] Hyperparameters:")
        print(f"  lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}")
        print(f"  outputscale: {model.covar_module.outputscale.item():.4f}")
        print(f"  noise:       {likelihood.noise.item():.6f}")
        print("------------------------end training---------------------------")
        
        
    if para_output:
        return model.eval(), likelihood.eval(),model.covar_module.base_kernel.lengthscale.item(),model.covar_module.outputscale.item()
    else:
        return model.eval(), likelihood.eval()


# Define GP model with user-provided mean function
class NonZeroMeanGPModel_reconstructed(gpytorch.models.ExactGP,GPyTorchModel):
    
    _num_outputs = 1
    
    def __init__(self, train_x, train_y, likelihood, mean_function, dev_fun, l_range=[1e-5,1e5],sigma_range=[0,5]):
        super().__init__(train_x, train_y, likelihood)
        self.mean_function = mean_function
        # set the kernel
        self.rbf_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(l_range[0],l_range[1])),
                outputscale_constraint=gpytorch.constraints.Interval(sigma_range[0],sigma_range[1])
                )
        noise_low=min(1e-4,sigma_range[0]*0.002)
        noise_upper=max(5.01e-4,sigma_range[1]*0.02)
        # print("low:take the minimum:",5e-4,sigma_range[0]*0.002,"high:take the maximum",5.01e-4,sigma_range[1]*0.02)
        self.noise_kernel = gpytorch.kernels.ScaleKernel(InputDependentWhiteKernel(dev_fun), outputscale_constraint=gpytorch.constraints.Interval(noise_low,noise_upper))
        self.noise_kernel.outputscale = torch.tensor(5e-4)
        self.covar_module = self.rbf_kernel +self.noise_kernel
        # self.covar_module = self.rbf_kernel *self.noise_kernel+self.rbf_kernel
        # set the kernel
        self.likelihood.noise_covar.noise = torch.tensor(1e-6)
        self.lengthscale_to_outputscale_ratio = 0


    def forward(self, x):
        # mean_x = self.mean_module(x)
        mean_x = self.mean_function(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
   
    
    def constraint_loss(self):
        """
        Custom loss function to enforce the constraint:
        noise_kernel.outputscale <= 0.1 * rbf_kernel.outputscale
        """
        # Get the current outputscale for RBF kernel (this is the signal variance)
        rbf_outputscale = self.rbf_kernel.outputscale.item()
        rbf_lengthscale = self.rbf_kernel.base_kernel.lengthscale.item()
        
        # Get the current outputscale for noise kernel (this is the noise variance)
        noise_outputscale = self.noise_kernel.outputscale.item()
    
        # Ensure that the noise kernel's outputscale is <= 0.1 * RBF kernel's outputscale
        device = next(self.parameters()).device
        zero_tensor = torch.tensor(0.0, device=device)  # Ensure same device
        penalty = torch.max(zero_tensor, torch.tensor(noise_outputscale - 0.01 * rbf_outputscale, device=device))
        self.lengthscale_to_outputscale_ratio = rbf_lengthscale/rbf_outputscale
     
        return penalty
    

    def objective(self):
         """
         Custom objective function that includes both the negative log marginal likelihood 
         and the penalty for violating the constraint.
         """
         # Get the negative log marginal likelihood (NLL) from the model
         nll = super().objective()  # Assuming you have a method that computes the NLL
         
         # Add the constraint loss to the NLL
         constraint_penalty = self.constraint_loss()
         
         return nll + constraint_penalty


def train_reconstructed_gp(input_data, output_data,  print_flag, machanistic_fun, training_iter=100, learning_rate=0.1,return_all_flag=False,decay_factor=0.8):
    """
    Train a Gaussian Process model with:
        - reconstructed kernel
        - Non-zero mean function from machanistic_fun
        - Very small fixed observation noise (simulates noise-free data)
    
    Args:
        input_data (array-like): Training inputs of shape [n, d]
        output_data (array-like): Training targets of shape [n] or [n, 1]
        machanistic_fun (callable): Function that maps torch tensor [n, d] -> [n] or [n, 1]
        training_iter (int): Number of training iterations
        learning_rate (float): Learning rate for the optimizer
    
    Returns:
        model: Trained GP model (set to eval mode)
        likelihood: Trained likelihood (set to eval mode)
    """

    # Convert input and output to torch tensors
    if isinstance(input_data, torch.Tensor):
        train_x = input_data.clone().detach().float()
    else:
        train_x = torch.tensor(input_data, dtype=torch.float32)
    
    if isinstance(output_data, torch.Tensor):
        train_y = output_data.clone().detach().float().squeeze()
    else:
        train_y = torch.tensor(output_data, dtype=torch.float32).squeeze()


    # Define Gaussian likelihood with tiny fixed noise
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )
        
        
    #train pure GP model
    trained_model_RBF1, trained_likelihood_RBF1,l1,sigma1_ =train_RBF_zeroprior_gp(train_x, train_y,print_flag=False,para_output=True)
    trained_model_RBF2, trained_likelihood_RBF2,l2,sigma2_ = train_RBF_nonzeroprior_gp(train_x, train_y,print_flag=False,machanistic_fun=machanistic_fun,para_output=True)


    def dev_fun(x):
        # Set the data-driven GP model to evaluation mode
        trained_model_RBF2.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Predict the mean of the GP at input x
            pred_dist = trained_model_RBF2(x)
            pred_mean = pred_dist.mean
            # Evaluate the mechanistic function at input x
            mech_mean = machanistic_fun(x)
            # Calculate and return the absolute residual between the GP prediction and mechanistic output
            residual = torch.abs(pred_mean - mech_mean)
            # print(residual)
        return residual
    
    
    scale_index=float(torch.max(dev_fun(train_x)))
    def dev_fun_scale(x):
        return dev_fun(x)/scale_index*decay_factor
    

    # Instantiate model
    model = NonZeroMeanGPModel_reconstructed(train_x, train_y, likelihood, machanistic_fun,dev_fun_scale,l_range=[0.8*l2,1.2*l2],sigma_range=[0.8*sigma2_,1.2*sigma2_])
    
    if print_flag:

        print("---------------------start training...------------------------")
        print("[Before training] Hyperparameters:")
        print(f"  RBF kernel lengthscale:    {model.rbf_kernel.base_kernel.lengthscale.item():.4f}")
        print(f"  RBF kernel outputscale:    {model.rbf_kernel.outputscale.item():.4f}")
        print(f"  Noise kernel outputscale:  {model.noise_kernel.outputscale.item():.4f}")
        print(f"  Likelihood noise:          {likelihood.noise.item():.6f}")

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer and marginal likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        constraint_penalty = model.constraint_loss()
        total_loss = loss + constraint_penalty
        total_loss.backward()
        if i % 20 == 0 and print_flag:
            print(f"...[Iter {i+1}/{training_iter}] Loss: {loss.item():.4f}, Constraint Penalty: {constraint_penalty.item():.4f}")
            print("lengthscale_to_outputscale_ratio:",model.lengthscale_to_outputscale_ratio)
        optimizer.step()

    
    if model.lengthscale_to_outputscale_ratio>100:
        model = NonZeroMeanGPModel_reconstructed(train_x, train_y, likelihood, machanistic_fun,dev_fun_scale,l_range=[0.8*l1,1.2*l1],sigma_range=[0.8*sigma1_,1.2*sigma1_])
        
        if print_flag:
            print("---------------------re-tuning hyperparameters....------------------------")
            print("[Before training] Hyperparameters:")
            print(f"  RBF kernel lengthscale:    {model.rbf_kernel.base_kernel.lengthscale.item():.4f}")
            print(f"  RBF kernel outputscale:    {model.rbf_kernel.outputscale.item():.4f}")
            print(f"  Noise kernel outputscale:  {model.noise_kernel.outputscale.item():.4f}")
            print(f"  Likelihood noise:          {likelihood.noise.item():.6f}")

        # Training mode
        model.train()
        likelihood.train()

        # Optimizer and marginal likelihood
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            constraint_penalty = model.constraint_loss()
            total_loss = loss + constraint_penalty
            total_loss.backward()
            if i % 20 == 0 and print_flag:
                print(f"...[Iter {i+1}/{training_iter}] Loss: {loss.item():.4f}, Constraint Penalty: {constraint_penalty.item():.4f}")
            optimizer.step()
        

    if print_flag:
        print("\n[After training] Hyperparameters:")
        print(f"  RBF kernel lengthscale:    {model.covar_module.kernels[0].base_kernel.lengthscale.item():.4f}")
        print(f"  RBF kernel outputscale:    {model.covar_module.kernels[0].outputscale.item():.4f}")
        print(f"  Noise kernel outputscale:  {model.covar_module.kernels[1].outputscale.item():.4f}")
        print(f"  Likelihood noise:          {likelihood.noise.item():.6f}")
        print("------------------------end training---------------------------")

    if return_all_flag:
        return model.eval(), likelihood.eval(), trained_model_RBF2.eval(), trained_likelihood_RBF2.eval()

    else:
        return model.eval(), likelihood.eval()



def train_memory_enhanced_gp(input_data, output_data,print_flag, machanistic_fun, alpha=0.8, backward_step=4, initial_sampling_point=5, training_iter=100, learning_rate=0.1):
    """
    Train a Gaussian Process model with:
        - smooth-reconstructed kernel
        - Non-zero mean function from machanistic_fun
        - Very small fixed observation noise (simulates noise-free data)
    
    Args:
        input_data (array-like): Training inputs of shape [n, d]
        output_data (array-like): Training targets of shape [n] or [n, 1]
        machanistic_fun (callable): Function that maps torch tensor [n, d] -> [n] or [n, 1]
        training_iter (int): Number of training iterations
        learning_rate (float): Learning rate for the optimizer
    
    Returns:
        model: Trained GP model (set to eval mode)
        likelihood: Trained likelihood (set to eval mode)
    """
    
    # Ensure input_data is a float32 tensor
    if isinstance(input_data, torch.Tensor):
        train_x = input_data.clone().detach().float()
    else:
        train_x = torch.tensor(input_data, dtype=torch.float32)
    # Ensure output_data is a float32 tensor and squeezed
    if isinstance(output_data, torch.Tensor):
        train_y = output_data.clone().detach().float().squeeze()
    else:
        train_y = torch.tensor(output_data, dtype=torch.float32).squeeze()
    
    GP_list_all=[]
    current_sample_number = input_data.shape[0]
    current_gp_number=current_sample_number-initial_sampling_point+1  #because the initial sampling also build GP
    
    if backward_step<current_gp_number:
        real_step=backward_step
        account_machanistic_flag=False
    else:
        real_step=current_gp_number
        account_machanistic_flag=True
    
    # Define Gaussian likelihood with tiny fixed noise
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )
        

    trained_model1, trained_likelihood1,l1,sigma1_ =train_RBF_zeroprior_gp(train_x, train_y,print_flag=False,para_output=True)
    trained_model, trained_likelihood,l2,sigma2_ = train_RBF_nonzeroprior_gp(train_x, train_y,print_flag,machanistic_fun,para_output=True)
    GP_list_all.append([trained_model,trained_likelihood])

    
    for i in range(1,real_step):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        train_x_prev = train_x[:-i]    # shape: [N-1, D]
        train_y_prev = train_y[:-i]    # shape: [N-1] or [N-1, 1]
        trained_model, trained_likelihood = train_RBF_nonzeroprior_gp(train_x_prev, train_y_prev,print_flag,machanistic_fun)
        GP_list_all.append([trained_model,trained_likelihood])



    def predict_with_model(x, model_num):
        """
        Predict using the specified GP model in GP_list_all.
    
        Args:
            GP_list_all (list): List of [model, likelihood] pairs.
            x (Tensor): Input tensor for prediction.
            model_num (int): Index of the model to use (0-based).
    
        Returns:
            pred_mean (Tensor): Predicted mean.
            pred_std (Tensor): Predicted standard deviation.
        """
        # Check bounds
        if model_num < 0 or model_num >= len(GP_list_all):
            raise IndexError(f"model_num {model_num} is out of bounds (0 to {len(GP_list_all) - 1})")
    
        # Extract the model and likelihood
        model, likelihood = GP_list_all[model_num]
    
        # Set to eval mode
        model.eval()
        likelihood.eval()
    
        # Make prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(x))
            pred_mean = pred_dist.mean
            pred_std = pred_dist.stddev
    
        return pred_mean, pred_std


    def dev_fun(x):
        # Flatten x to shape (-1, d) so it can be processed in one go
        original_shape = x.shape[:-1]  # e.g., (200, 6)
        x_flat = x.view(-1, x.shape[-1])  # shape: (200*6, d)
    
        residual = torch.zeros(x_flat.shape[0], dtype=torch.float32, device=x.device)
    
        for i in range(1, real_step):
            mean_pre, _ = predict_with_model(x_flat, model_num=i-1)
            mean, _ = predict_with_model(x_flat, model_num=i)
            diff = torch.abs(mean_pre - mean)
            residual += (alpha ** i) * diff.squeeze()
    
        if account_machanistic_flag:
            try:
                mech_mean = machanistic_fun(x_flat)
                diff_mech = torch.abs(mean_pre - mech_mean).squeeze()
            except:
                mean_pre, _ = predict_with_model(x_flat, model_num=0)
                mech_mean = machanistic_fun(x_flat)
                diff_mech = torch.abs(mean_pre - mech_mean).squeeze()
            residual += (alpha ** real_step) * diff_mech
    
        # Reshape back to original batch dimensions (e.g., (200, 6))
        # print(residual.view(*original_shape))
        return residual.view(*original_shape)
    
    
    scale_index=float(torch.max(dev_fun(train_x)))+1e-6
    def dev_fun_scale(x):
        return dev_fun(x)/scale_index


    # Instantiate model
    model = NonZeroMeanGPModel_reconstructed(train_x, train_y, likelihood, machanistic_fun,dev_fun_scale,l_range=[0.8*l2,1.2*l2],sigma_range=[0.8*sigma2_,1.2*sigma2_])
    

    if print_flag:
        print("---------------------start training...------------------------")
        print("[Before training] Hyperparameters:")
        print(f"  RBF kernel lengthscale:    {model.rbf_kernel.base_kernel.lengthscale.item():.4f}")
        print(f"  RBF kernel outputscale:    {model.rbf_kernel.outputscale.item():.4f}")
        print(f"  Noise kernel outputscale:  {model.noise_kernel.outputscale.item():.4f}")
        print(f"  Likelihood noise:          {likelihood.noise.item():.6f}")

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer and marginal likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        constraint_penalty = model.constraint_loss()
        total_loss = loss + constraint_penalty
        total_loss.backward()
        if i % 20 == 0 and print_flag:
            print(f"...[Iter {i+1}/{training_iter}] Loss: {loss.item():.4f}, Constraint Penalty: {constraint_penalty.item():.4f}")
        optimizer.step()

    
    if model.lengthscale_to_outputscale_ratio>100:
        model = NonZeroMeanGPModel_reconstructed(train_x, train_y, likelihood, machanistic_fun,dev_fun_scale,l_range=[0.8*l1,1.2*l1],sigma_range=[0.8*sigma1_,1.2*sigma1_])
        
        
        if print_flag:
            print("---------------------re-tuning hyperparameters...------------------------")

        # Training mode
        model.train()
        likelihood.train()

        # Optimizer and marginal likelihood
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            constraint_penalty = model.constraint_loss()
            total_loss = loss + constraint_penalty
            total_loss.backward()
            if i % 10 == 0 and print_flag:
                print(f"...[Iter {i+1}/{training_iter}] Loss: {loss.item():.4f}, Constraint Penalty: {constraint_penalty.item():.4f}")
            optimizer.step()
        

    if print_flag:
        print("\n[After training] Hyperparameters:")
        print(f"  RBF kernel lengthscale:    {model.covar_module.kernels[0].base_kernel.lengthscale.item():.4f}")
        print(f"  RBF kernel outputscale:    {model.covar_module.kernels[0].outputscale.item():.4f}")
        print(f"  Noise kernel outputscale:  {model.covar_module.kernels[1].outputscale.item():.4f}")
        print(f"  Likelihood noise:          {likelihood.noise.item():.6f}")
        print("------------------------end training---------------------------")

    return model.eval(), likelihood.eval()




"""
The next functions are for visualization
"""



def build_rbf_gp_traces_3d(rbf_model, rbf_likelihood, input_data,output_data, machanistic_fun,
                           bounds, grid_size=50, showscale=False, truth_ground_fun=None):
    """
    Return Plotly 3D traces including:
    - GP model mean
    - Confidence bounds
    - Training samples
    - Mechanistic prior
    - Deviation (|prior - mean|)
    - Truth ground surface (optional)
    """
    # Generate grid
    x1 = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    x2 = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # GP model prediction
    rbf_model.eval()
    rbf_likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = rbf_likelihood(rbf_model(X_test_tensor))
        mean = pred_dist.mean.numpy().reshape(grid_size, grid_size)
        std = pred_dist.variance.sqrt().numpy().reshape(grid_size, grid_size)

    # Mechanistic prior
    with torch.no_grad():
        prior_vals = machanistic_fun(X_test_tensor).numpy().reshape(grid_size, grid_size)

    # Deviation
    deviation = np.abs(prior_vals - mean)

    # Truth ground model
    if truth_ground_fun is not None:
        X_test_np = X_test_tensor.numpy()  # shape: (N, 2)
        truth_vals_flat = np.array([
            truth_ground_fun(float(x[0]), float(x[1])) for x in X_test_np
        ])
        truth_vals = truth_vals_flat.reshape(grid_size, grid_size)

        trace_truthground = go.Surface(
            z=truth_vals, x=X1, y=X2,
            colorscale=[[0, '#F2A65A'], [1, '#F2A65A']],
            opacity=0.9,
            name="Ground truth",
            showscale=False,
            showlegend=True
        )
    else:
        trace_truthground = None


    # GP model
    trace_model = go.Surface(
        z=mean, x=X1, y=X2,
        colorscale=[[0, '#5E94A2'], [1, '#5E94A2']],
        opacity=1.0,
        name="GP model",
        showscale=showscale,
        showlegend=True
    )

    # Confidence bounds
    trace_upper = go.Surface(
        z=mean + 1.96 * std, x=X1, y=X2,
        colorscale=[[0, '#C4C4C4'], [1, '#C4C4C4']],
        opacity=0.45,
        name="GP upper bound",
        showscale=False,
        showlegend=True
    )
    trace_lower = go.Surface(
        z=mean - 1.96 * std, x=X1, y=X2,
        colorscale=[[0, '#C4C4C4'], [1, '#C4C4C4']],
        opacity=0.45,
        name="GP lower bound",
        showscale=False,
        showlegend=True
    )

    # Mechanistic prior
    trace_prior = go.Surface(
        z=prior_vals, x=X1, y=X2,
        colorscale=[[0, '#80CFA9'], [1, '#80CFA9']],
        opacity=0.4,
        name="Mechanistic prior",
        showscale=False,
        showlegend=True
    )

    # Deviation
    trace_dev = go.Surface(
        z=deviation, x=X1, y=X2,
        colorscale=[[0, '#D52B4D'], [1, '#D52B4D']],
        opacity=0.45,
        name="Deviation",
        showscale=False,
        showlegend=True
    )

    # Training samples
    input_data = np.array(input_data)
    trace_sampling = go.Scatter3d(
        x=input_data[:, 0],
        y=input_data[:, 1],
        z=output_data,
        mode='markers',
        name='Training samples',
        marker=dict(size=3, color='black'),
        showlegend=True
    )

    return trace_truthground, trace_model, trace_lower, trace_upper, trace_sampling, trace_prior, trace_dev






   













