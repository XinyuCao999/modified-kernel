# PyTorch Project: Bayesian Optimization Implementation with Modified Kernel

## Project Overview

This project implements a modified kernel GP using **PyTorch** and integrates **Bayesian Optimization** (BO) for automatic hyperparameter tuning. 

The main modification of the method is based on the part **“GPyTorch”**. The **train_GP** function initiates the training by calling **train_reconstructed_gp** with the necessary input and output data, along with model-specific parameters. The **InputDependentWhiteKernel class** defines the kernel used in the model, with the init method for initialization and the forward method for computing the kernel’s output for given inputs. This modular structure organizes the code efficiently, with each component playing a clear role in the model's training and optimization.

<img width="1002" height="670" alt="image" src="https://github.com/user-attachments/assets/e10941da-695b-40c1-84cf-eeb759dddac9" />
