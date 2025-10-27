#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:33:17 2025

@author: Xinyu Cao

This file is used to simulate the process of schotten-baumann reaction
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
import torch
import scipy.stats.qmc as qmc
import plotly.graph_objects as go


bounds=np.array([[1,1.5],[0.1,6]])
x0_base=[0.3,0.3,0,0.3]


def reaction_equations(C, C_A0, C_B0, C_NaOH0, k1, k2, t):
    C_A, C_B, C_N, C_NaOH = C
    eq1 = (C_A0 - C_A) - (k1 * C_A * C_B + k2 * C_NaOH * C_A) * t
    eq2 = (C_B0 - C_B) - k1 * C_A * C_B * t
    eq3 = (0-C_N) + k1 * C_A * C_B * t
    eq4 = (C_NaOH0 - C_NaOH) - k2 * C_NaOH * C_A * t
    return [eq1, eq2, eq3, eq4]



def schotten_CSTR(theta,x0,tau,return_all_flag=False):
    k1,k2=theta
    C_A0, C_B0, C_N0, C_NaOH0=x0    #sate at tau=0s
    C_A, C_B, C_N, C_NaOH = fsolve(reaction_equations, x0, args=(C_A0, C_B0, C_NaOH0, k1, k2, tau))
    if return_all_flag:
        return C_A, C_B, C_N, C_NaOH
    else:
        return C_N


def residuals(theta, input_info, output_info):
    sample_num=len(output_info)
    output_simu=np.zeros(sample_num)
    obj=0
    for sample_index in range(sample_num):
        sample_input=input_info[sample_index]
        time=sample_input["reaction_time"]
        x0=x0_base.copy()
        x0[0]=x0[0]*sample_input["equiv"]
        x0[3]=x0[3]*sample_input["equiv"]
        output_simu[sample_index]=schotten_CSTR(theta,x0,time)
        obj+=(output_info[sample_index]-output_simu[sample_index])**2
    print(obj,output_simu,theta)
    return obj


def sample_cal(theta,input_info_single):
    time=input_info_single["reaction_time"]
    x0=x0_base.copy()
    x0[0]=x0[0]*input_info_single["equiv"]
    x0[3]=x0[3]*input_info_single["equiv"]
    output_simu=schotten_CSTR(theta,x0,time)
    return output_simu
    


def theta_cal(flowrate):
    if flowrate<1:
        k1=20*flowrate+10
        k2=4.5*flowrate+0.5
    elif 1<=flowrate<4:
        k1=10*flowrate+20
        k2=0.5*flowrate+4.5
    else:
        k1=60
        k2=6.5
    return [k1,k2]



def schotten_truth_ground(x_equal,y_flowrate):
    theta=theta_cal(y_flowrate)
    resident_time=0.5/y_flowrate*60
    x0=x0_base.copy()
    x0[0]=x0[0]*x_equal
    x0[3]=x0[3]*x_equal
    output=schotten_CSTR(theta,x0,min(300,resident_time))
    yield_=output/0.3
    return yield_


def truth_ground(X):
    x_equal,y_flowrate=X.squeeze().tolist()
    theta=theta_cal(y_flowrate)
    resident_time=0.5/y_flowrate*60
    x0=x0_base.copy()
    x0[0]=x0[0]*x_equal
    x0[3]=x0[3]*x_equal
    output=schotten_CSTR(theta,x0,min(300,resident_time))
    yield_=output/0.3
    return yield_



def schotten_reaction_rate(x_equal,y_flowrate):
    theta=theta_cal(y_flowrate)
    resident_time=5
    x0=x0_base.copy()
    x0[0]=x0[0]*x_equal
    x0[3]=x0[3]*x_equal
    C_A, C_B, C_N, C_NaOH=schotten_CSTR(theta,x0,min(300,resident_time),True)
    r1=theta[0]*C_A*C_B
    r2=theta[1]*C_A*C_NaOH
    return r1,r2
    
    

def prior_schotten(x_equal,y_flowrate,theta=[60,6.5]):
    # Ensure y_flowrate is a Python float  
    if isinstance(y_flowrate, torch.Tensor):
        y_flowrate = y_flowrate.item()
    elif isinstance(y_flowrate, np.ndarray):
        y_flowrate = float(y_flowrate)
    
    resident_time=0.5/y_flowrate*60
    x0=x0_base.copy()
    x0[0]=x0[0]*x_equal
    x0[3]=x0[3]*x_equal
    output=schotten_CSTR(theta,x0,min(300,resident_time))
    yield_=output/0.3
    return yield_


def prior_defult(input_all, theta=[60, 6.5]):
    
    if isinstance(input_all, np.ndarray):
        input_all = torch.tensor(input_all, dtype=torch.float32)
    
    #Ensure input_all is a 3D tensor (b, q, d)
    if input_all.dim() == 2:
        input_all = input_all.unsqueeze(0)  # convert (q, d) -> (1, q, d)

    b, q, d = input_all.shape
    input_flat = input_all.view(-1, d)  # shape: (b*q, d)

    results = []
    for input_single in input_flat:
        x1 = input_single[0].item()
        x2 = input_single[1].item()
        try:
            y = prior_schotten(x1, x2, theta)
        except Exception as e:
            # Optional: log error for debugging
            print(f"[Warning] prior_schotten failed on input ({x1:.4f}, {x2:.4f}): {e}")
            y = 0.0  # fallback value
        results.append(torch.tensor(y, dtype=torch.float32))

    result_tensor = torch.stack(results)  # shape: (b*q,)
    return result_tensor.view(b, q)  # reshape back to (b, q)



def sample_fun(input_data):
    sample_num=len(input_data)
    output_data=np.zeros(sample_num)
    for i in range(sample_num):
        output_data[i]=schotten_truth_ground(input_data[i][0],input_data[i][1])
    return output_data
    

def initial_sampling_fun(lhs_sampling_num):
    # lhs = qmc.LatinHypercube(d=2,seed=42)
    lhs = qmc.LatinHypercube(d=2)
    samples = lhs.random(n=lhs_sampling_num)
    input_data = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    sample_num=len(input_data)
    output_data=np.zeros(sample_num)
    for i in range(sample_num):
        output_data[i]=schotten_truth_ground(input_data[i][0],input_data[i][1])
    return input_data,output_data


def plot_schotten_CSTR(trace_truthground=None, trace_model=None, trace_lower=None,
                       trace_upper=None, trace_sampling=None, trace_prior=None,trace_dev=None,
                       save_name=None,model_name=""):
    traces = []
    
    if trace_truthground is not None:
        traces.append(trace_truthground)
    if trace_model is not None:
        traces.append(trace_model)
    if trace_lower is not None:
        traces.append(trace_lower)
    if trace_upper is not None:
        traces.append(trace_upper)
    if trace_sampling is not None:
        traces.append(trace_sampling)
    if trace_prior is not None:
        traces.append(trace_prior)
    if trace_dev is not None:
        traces.append(trace_dev)

    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=model_name,
        scene=dict(
            xaxis_title="initial concentration",
            yaxis_title="flowrate",
            zaxis_title="yield"
        ),
        autosize=False,
        width=800,
        height=800
    )

    fig.show()

    if save_name is not None:
        fig.update_layout(width=1200, height=1200)
        fig.write_html(save_name + ".html")
        
        

if __name__=='__main__':

    equiv=1.2
    flowrate=6
    k1 = 11.62 * np.log(flowrate) / np.log(2.04) + 56.30
    k2 = 1.03 * np.log(flowrate) / np.log(1.61) + 5.62 
    theta=[k1,k2]
    y_output=schotten_CSTR(theta,[0.3*equiv,0.3,0,0.3*equiv],0.5/flowrate*60)
    print(k1,k2,y_output/0.3)
    r1,r2=schotten_reaction_rate(equiv,flowrate)
    
    
    
    
    
    
    
    
    
    
    

