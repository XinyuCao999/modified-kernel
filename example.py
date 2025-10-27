#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 21:44:45 2025

@author: Xinyu Cao
"""


import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BO_base import bo_result_statistics,run_BO
from gp_model_base import train_GP,model_prediction
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
model_type="RBF_zeroprior"/"RBF_nonzeroprior"/"reconstructed"
"""
model_type="RBF_nonzeroprior"



# Create a synthetic parameter space
x = np.linspace(0, 20, 500)
true_objective = -np.sin(x) * np.exp(0.05 * x) + 2.5  # "True" unknown objective



def truth_gound(input_all):
    if isinstance(input_all, np.ndarray):
        input_all = torch.tensor(input_all, dtype=torch.float32)
    
    #Ensure input_all is a 3D tensor (b, q, d)
    if input_all.dim() == 2:
        input_all = input_all.unsqueeze(0)  # convert (q, d) -> (1, q, d)

    b, q, d = input_all.shape
    input_flat = input_all.view(-1, d)  # shape: (b*q, d)

    results = []
    for input_single in input_flat:
        x = input_single[0].item()
        y=  -np.sin(x) * np.exp(0.05 * x) + 2.5
        results.append(torch.tensor(y, dtype=torch.float32))

    result_tensor = torch.stack(results)  # shape: (b*q,)
    return result_tensor.view(b, q)  # reshape back to (b, q)


# Simulate prior model prediction

def prior_fun(input_all):
    
    if isinstance(input_all, np.ndarray):
        input_all = torch.tensor(input_all, dtype=torch.float32)
    
    #Ensure input_all is a 3D tensor (b, q, d)
    if input_all.dim() == 2:
        input_all = input_all.unsqueeze(0)  # convert (q, d) -> (1, q, d)

    b, q, d = input_all.shape
    input_flat = input_all.view(-1, d)  # shape: (b*q, d)

    results = []
    for input_single in input_flat:
        x = input_single[0].item()
        # y= -np.sin(x*1.1) * 0.8 + 1.5*(2-0.2*x)
        y= -np.sin(x*1.1) * 0.8 + 1.5*(2-0.1*x**1.3)
        results.append(torch.tensor(y, dtype=torch.float32))

    result_tensor = torch.stack(results)  # shape: (b*q,)
    return result_tensor.view(b, q)  # reshape back to (b, q)


def plot_res(xs,ys,model1):
    prior_prediction = np.array(prior_fun(x.reshape(-1,1))).reshape(-1)
    
    # Define uncertainty
    # For GP model1
    mean1,std1=model_prediction(x.reshape(-1,1),model1) 
    mean1=mean1.reshape(-1)
    std1=std1.reshape(-1)
    
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10,5),dpi=200)
    
    
    # Plot true function and prior
    ax.plot(x, true_objective,'k-',label='True Objective (unknown)',linewidth=2)
    ax.plot(x, prior_prediction,'k--', label='Mechanistic Prior Prediction')
    ax.plot(x, mean1,linestyle='-',color="#920909",  label="GP model",linewidth=2)
    # Plot background to indicate uncertainty
    ax.fill_between(x, mean1 - 3*std1, mean1 + 3*std1,
                    color='lightgray', alpha=0.5, label='Model Uncertainty')
    
    
    # Highlight Region A and B
    ax.axvspan(0, 5, color='green', alpha=0.15, label='Region A: Prior likely valid')
    ax.axvspan(10, 20, color='red', alpha=0.15, label='Region B: Prior likely invalid')
    
    # Add BO samples
    ax.scatter(xs, ys, marker='o', color='blue', label='Samples')
    
    
    # Labels and legend
    try:
        kernel_name, prior_info = model_type.split("_")
    except:
        kernel_name="modified"
        prior_info="dashed-line prior"
    if prior_info=="zeroprior":
        prior_info="0 prior"
    else:
        prior_info="dashed-line prior"
    ax.set_title('GP model ('+kernel_name+" kernel, "+prior_info+")", fontsize=24,fontname="Times New Roman")
    ax.set_xlabel('Design Space',fontsize=24,fontname="Times New Roman")
    ax.set_ylabel('Objective Value',fontsize=24,fontname="Times New Roman")
    ax.legend(loc='lower left',fontsize=12)
    ax.grid(True)
    ax.set_ylim(-6,8)
    
    
    
    plt.tight_layout()
    plt.show()


# create dataset
xs = np.array([1.5,6.5,11.5,16.5]).reshape(-1,1)+0.6
ys = np.interp(xs, x, true_objective)
model=train_GP(xs,ys,model_type,machanistic_fun=prior_fun)
plot_res(xs,ys,model)


    






























