#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 12:05:24 2025

@author: Xinyu Cao

This file includes process model, sampling function and parameter estimation for William Otto
"""

import numpy as np
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize,fsolve
from scipy.stats import norm
from skopt.sampler import Lhs, Grid
from skopt.space import Space
import warnings
warnings.filterwarnings("ignore")


bounds=np.array([[3,6],[70,100]])
optimum=np.array([4.78765,89.70268])
true_parameter=[1.660e6,7.212e8,2.675e12,6666.7,8333.3,11111]
mismatched_system_parameter=[9.38706916e+07,1.44235855e+12,7.98710955e+03,1.15500001e+04]



def my_fun(x,theta,u,mode):

    F_B,T_R=u
    F_A=1.8275
    F_R=F_A+F_B
    V_R=2105
        
    
    if mode=="accurate":
        x_A,x_B,x_C,x_E,x_G,x_P=x
        k10,k20,k30,alpha_1,alpha_2,alpha_3=theta
        
        k1=k10*np.exp(-alpha_1/(T_R+273.15))
        k2=k20*np.exp(-alpha_2/(T_R+273.15))
        k3=k30*np.exp(-alpha_3/(T_R+273.15))

     
        remainx_A = F_A-F_R*x_A-k1*V_R*x_A*x_B
        remainx_B = F_B-F_R*x_B-k1*V_R*x_A*x_B-k2*V_R*x_B*x_C
        remainx_C = 2*k1*V_R*x_A*x_B-F_R*x_C-2*k2*V_R*x_B*x_C-k3*V_R*x_C*x_P
        remainx_E = 2*k2*V_R*x_B*x_C-F_R*x_E
        remainx_G = 1.5*k3*V_R*x_C*x_P-F_R*x_G
        remainx_P = k2*V_R*x_B*x_C-F_R*x_P-0.5*k3*V_R*x_C*x_P
        
        remainx=np.array([remainx_A,remainx_B,remainx_C,remainx_E,remainx_G,remainx_P]).reshape(-1)
    
    elif mode=="mismatched":
        x_A,x_B,x_E,x_G,x_P=x
        k10,k20,alpha_1,alpha_2=theta
        
        k1=k10*np.exp(-alpha_1/(T_R+273.15))
        k2=k20*np.exp(-alpha_2/(T_R+273.15))
     
        remainx_A = F_A-F_R*x_A-k1*V_R*x_A*x_B**2-k2*V_R*x_A*x_B*x_P
        remainx_B = F_B-F_R*x_B-2*k1*V_R*x_A*x_B**2-k2*V_R*x_A*x_B*x_P
        remainx_E = 2*k1*V_R*x_A*x_B**2-F_R*x_E
        remainx_G = 3*k2*V_R*x_A*x_B*x_P-F_R*x_G
        remainx_P = k1*V_R*x_A*x_B**2-F_R*x_P-k2*V_R*x_A*x_B*x_P
    
        remainx=np.array([remainx_A,remainx_B,remainx_E,remainx_G,remainx_P]).reshape(-1)
        
    return remainx



def Williams_Otto_process(theta,u,mode="accurate"):
    if mode=="accurate":
        x0=np.array([0.1,0.1,0.1,0.1,0.1,0.1])
    else:
        x0=np.array([0.1,0.1,0.1,0.1,0.1])
        
    y = fsolve(my_fun,x0,args=tuple([theta,u,mode]))
    return y




def Williams_Otto_profit_true(u,theta):
    y=Williams_Otto_process(theta,u,"accurate")
    x_P=y[-1]
    x_E=y[-3]
    F_A=1.8275
    F_B=u[0]
    return (F_A+F_B)*(1143.48*x_P+25.92*x_E)-76.23*F_A-114.34*F_B



def Williams_Otto_profit_mismatched(u,theta):
    y=Williams_Otto_process(theta,u,"mismatched")
    x_P=y[-1]
    x_E=y[-3]
    F_A=1.8275
    F_B=u[0]
    return (F_A+F_B)*(1143.48*x_P+25.92*x_E)-76.23*F_A-114.34*F_B



def Williams_Otto_profit(u,theta=None,mode="accurate"):
    
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy().reshape(-1)
    
    if mode=="accurate":
        if theta==None:
            theta=true_parameter
        return Williams_Otto_profit_true(u,theta)
    
    if mode=="mismatched":
        if theta==None:
            theta=mismatched_system_parameter
        return Williams_Otto_profit_mismatched(u,theta)
    


def take_sample(u_series):
    return_res=[]
    for i,ele in enumerate(u_series):
        return_res.append(experiemnt(ele))
    return return_res


def experiemnt(u,theta=true_parameter,noise_sd=0):
    u= np.round(u, 2)
    if noise_sd!=0:
        noise=norm.rvs(loc=0,scale=noise_sd,size=1)[0]
        yeild=Williams_Otto_profit(u,theta)+noise
    else:
        yeild=Williams_Otto_profit(u,theta)
    
    return np.round(yeild, 2)


def initial_condition_sampling(initial_point,method="l"):
    space = Space([(3,6),(70,100)]) 
    if method=="l":
        lhs = Lhs() 
        # u_series = lhs.generate(space.dimensions,initial_point,random_state=42)
        u_series = lhs.generate(space.dimensions,initial_point)
    elif method=="g":
        grid = Grid(border="include", use_full_layout=False)
        u_series = grid.generate(space.dimensions, initial_point)
    return u_series



def initial_sampling_fun(initial_point_number):
    u_series=initial_condition_sampling(initial_point_number)
    profit=take_sample(u_series)
    return np.array(u_series),np.array(profit)
    
    

#optimize parameters
def residual_fun(theta,sample_information):
    u_series=sample_information[0]
    output_sampled=sample_information[1]
    output_predicted=output_sampled.copy()
    obj=0
    for i,ele in enumerate(u_series):
        output_predicted[i]=Williams_Otto_profit_mismatched(ele,theta)
        obj+=(output_predicted[i]-output_sampled[i])**2
    # print(obj)
    return obj


def parameter_opt(sample_information,initial_theta=np.array([2.189e8,4.310e13,8077.6,12348])):
    opt_para=minimize(fun=residual_fun,x0=initial_theta,method='powell',options={'ftol':1e-8},
                      args=tuple([sample_information]))
    # print("optimized parameters",opt_para.x)
    opt_res=opt_para.x
    
    return opt_res



def prior_defult(input_all, theta=mismatched_system_parameter):
    
    if isinstance(input_all, np.ndarray):
        input_all = torch.tensor(input_all, dtype=torch.float32)
    
    #Ensure input_all is a 3D tensor (b, q, d)
    if input_all.dim() == 2:
        input_all = input_all.unsqueeze(0)  # convert (q, d) -> (1, q, d)

    b, q, d = input_all.shape
    input_flat = input_all.view(-1, d)  # shape: (b*q, d)

    results = []
    for input_single in input_flat:
        y = Williams_Otto_profit_mismatched(input_single.detach().numpy(),theta)
        results.append(torch.tensor(y, dtype=torch.float32))

    result_tensor = torch.stack(results)  # shape: (b*q,)
    return result_tensor.view(b, q)  # reshape back to (b, q)




if __name__=='__main__':
    
    u=[3,70]
    print(Williams_Otto_profit(u,mode="mismatched"))
    
    u_series,profit=initial_sampling_fun(5)
    sample_information=[u_series,profit]
    
    res_opt=parameter_opt(sample_information)
    print("res:",residual_fun(res_opt,sample_information))
    
    
    
    
    
    
    
    
    
    



