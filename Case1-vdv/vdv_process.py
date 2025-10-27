#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 18:43:53 2025

@author: Xinyu Cao
"""


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import norm
import os
import torch
from skopt.space import Space
from skopt.sampler import Lhs, Grid
import warnings
import datetime
warnings.filterwarnings("ignore")


#set the true value
k1 = 10.0  # 1/s
k2 = 1.0   # 1/s
k3 = 1.0   # L/(mol*s)
cA0 = 5.8  # mol/L
cB0 = 1.16 # mol/L
para_true = k1, k2, k3
c0_true = cA0, cB0 
bounds=np.array([[1e-5, 0.9999], [1e-5, 0.9999], [0.0, 1.0]])
optimal_input=np.array([0.47149541, 0.07779382,0.15798068])  #the optimum, unknown


k1_model = 8  # 1/s
k2_model = 0.8   # 1/s
k3_model = 0.2  # L/(mol*s)
cA0_model = 4.8  # mol/L
cB0_model = 2.16 # mol/L
para_defult=tuple([k1_model, k2_model, k3_model])
c0_defult=tuple([cA0_model, cB0_model])




def VDV_truth_ground(x):
    
    x_np = x.detach().cpu().numpy()[0]

    return VDV(x_np, para_true, c0_true)["b_end"]



def VDV_truth(x):
    
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        x_np = x
    else:
        raise TypeError("x must be a torch.Tensor or numpy.ndarray")

    return VDV(x_np, para_true, c0_true)
    

def VDV(x, para, c0):
    f_main, f_end, z = x
    k1, k2, k3 = para
    cA0, cB0 = c0

    # Derived dimensionless parameters
    a1 = cA0 * k3 / k1
    a2 = k2 / k1
    b0 = cB0 / cA0

    def eta(f, b):
        if f <= 0:
            return 0
        return (a2 * b - f) / (f * (1 + a1 * f))

    def dbdf(f, b):
        return eta(f, b)

    def integrate_segment(f_start, f_end, b_start, steps=100):
        if f_start <= f_end:
            return -np.inf
        try:
            f_span = [f_start, f_end]
            sol = solve_ivp(lambda f, b: dbdf(f, b), f_span, [b_start], t_eval=np.linspace(f_start, f_end, steps))
            if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                return -np.inf
            return sol.y[0][-1]
        except:
            return -np.inf

    def compute_b_main(f_main):
        numerator = f_main * (1 - f_main + b0 * (1 + a1 * f_main))
        denominator = a2 * (1 - f_main) + f_main * (1 + a1 * f_main)
        return numerator / denominator

    def mix_streams(f_main, b_main, f0, b0, z):
        f_plus = (f_main + z * f0) / (1 + z)
        b_plus = (b_main + z * b0) / (1 + z)
        return f_plus, b_plus

    # Check bounds
    if f_main <= 0 or f_main >= 1 or f_end <= 0 or f_end >= f_main or z < 0:
        return {"status": "invalid_input", "b_end": 0}

    b_main = compute_b_main(f_main)
    f_plus, b_plus = mix_streams(f_main, b_main, 1.0, b0, z)
    b_end = integrate_segment(f_plus, f_end, b_plus)

    return {
        "f_main": f_main,
        "b_main": b_main,
        "f_plus": f_plus,
        "b_plus": b_plus,
        "f_end": f_end,
        "b_end": b_end,
        "z": z,
        "status":'"ok',
        }



def optimize_vdv(para, c0):
    
    def objective(x):
        result = VDV(x, para, c0)
        return -result["b_end"]

    initial_guess = [0.4, 0.1, 0.2]

    result = minimize(objective, initial_guess, bounds=bounds)

    best_x = result.x
    # best_result = VDV(best_x, para, c0)

    return best_x



def experiment_vdv(x, theta=para_true, c0=c0_true, noise_sd=0.0, save_result=False):
    x = np.round(x, 4) 
    result = VDV(x, theta, c0)

    b_end = result["b_end"]
    
    if noise_sd != 0:
        noise = norm.rvs(loc=0, scale=noise_sd, size=1)[0]
        b_end += noise

    b_end = np.round(b_end, 4)

    if save_result:
        filename = f"experiment_data/experiment({x[0]:.2f},{x[1]:.2f},{x[2]:.2f}).npy"
        os.makedirs("experiment_data", exist_ok=True)
        if not os.path.exists(filename):
            np.save(filename, {"b_end": b_end})
        else:
            b_end = np.load(filename, allow_pickle=True).item()["b_end"]

    return b_end




def take_sample(u_series):
    return_res=[]
    for i,ele in enumerate(u_series):
        return_res.append(experiment_vdv(ele))
    return return_res



def initial_sampling_vdv(initial_point, method="l", max_attempts=1000):
    space = Space(bounds) 
    valid_samples = []

    if method == "l":
        lhs = Lhs()
        attempts = 0
        while len(valid_samples) < initial_point and attempts < max_attempts:
            batch = lhs.generate(space.dimensions, initial_point)
            for x in batch:
                f_main, f_end, z = x
                if f_main > f_end:  # 合法点要求 f_main > f_end
                    valid_samples.append(x)
                    if len(valid_samples) >= initial_point:
                        break
            attempts += 1
    elif method == "g":
        grid = Grid(border="include", use_full_layout=False)
        raw_samples = grid.generate(space.dimensions, initial_point)
        valid_samples = [x for x in raw_samples if x[0] > x[1]]
        valid_samples = valid_samples[:initial_point]

    return valid_samples





def initial_sampling_fun(initial_point_number):

    u_series = initial_sampling_vdv(initial_point_number)
    yield_ = take_sample(u_series)

    return np.array(u_series), np.array(yield_)



def prior_defult(input_all, para=para_defult, c0=c0_defult):
    
    if isinstance(input_all, np.ndarray):
        input_all = torch.tensor(input_all, dtype=torch.float32)
    
    #Ensure input_all is a 3D tensor (b, q, d)
    if input_all.dim() == 2:
        input_all = input_all.unsqueeze(0)  # convert (q, d) -> (1, q, d)

    b, q, d = input_all.shape
    input_flat = input_all.view(-1, d)  # shape: (b*q, d)

    results = []
    for input_single in input_flat:
        VDV_dict=VDV(input_single.detach().numpy(), para, c0)
        y = VDV_dict["b_end"]
        results.append(torch.tensor(y, dtype=torch.float32))


    result_tensor = torch.stack(results)  # shape: (b*q,)
    return result_tensor.view(b, q)  # reshape back to (b, q)





if __name__=='__main__':
    

    
    print("\n--- Optimization Result ---")
    optimal_x_true = optimize_vdv(para_true, c0_true)
    optimal_result_true=VDV_truth(optimal_x_true)
    print("true model:",optimal_result_true["b_end"],optimal_x_true)
    
    
    #start doing the experiment
    sampling,all_sample=initial_sampling_fun(5)

    
    all_sample_mismatched=prior_defult(np.array(sampling)).numpy().reshape(-1)
    a_mismatch=all_sample_mismatched-all_sample
    
    optimal_x=optimize_vdv(para_defult,c0_defult)
    optimal_result=VDV_truth(optimal_x)
    print("mismatched model:",optimal_result["b_end"],optimal_x)
    
    
    
    
    
    
    