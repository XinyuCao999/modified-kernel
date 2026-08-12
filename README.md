# PyTorch Project: Bayesian Optimization Implementation with Modified Kernel

## Project Overview

This project implements a modified kernel GP using **PyTorch** and integrates **Bayesian Optimization** (BO) for automatic hyperparameter tuning. 

The main modification of the method is based on the part **“GPyTorch”**. This modular structure organizes the code efficiently, with each component playing a clear role in the model's training and optimization.

<img width="1002" height="670" alt="image" src="https://github.com/user-attachments/assets/e10941da-695b-40c1-84cf-eeb759dddac9" />


## Code Structure

The code structure of this project is designed for readability, maintainability, and modularity. Below is a brief overview of the project directory structure:

### Code Modules

1. **Gassian Process Model Training (`gp_model_base.py`)**:
   - `train_GP`: This function initiates the training process by calling `train_reconstructed_gp`, which performs the actual model training.
   - `InputDependentWhiteKernel`: Defines the kernel used in the model, with the `__init__` method for initialization and the `forward` method for computing the kernel's output for given inputs.

2. **Bayesian Optimization (`BO_base.py`)**:
   - This file provides the tools necessary for Bayesian Optimization for different models built from `gp_model_base.py`.
  
3. **Example (`example.py`)**:
   - This file demonstrates a one-dimensional case using the modules encapsulated in `gp_model_base.py`.

4. **Case Implementations (`Case1-vdv/`, `Case2-william otto/`, `Case3-schotten baumann/`)**:
   - These folders contain specific datasets and experiment configurations for each case study, showing how to apply the model in different scenarios.
   - Running the `main.py` file in the corresponding folder will train the model from scratch, and the results will be saved in the `BO/` folder under the respective folder. Running the Jupyter file in the corresponding folder will read the results from the `BO/` folder and generate plots.