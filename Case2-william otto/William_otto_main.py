#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 12:06:28 2025

@author: Xinyu Cao
"""


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from BO_base import bo_result_statistics,run_BO
from gp_model_base import train_GP,model_prediction
from William_Otto_process import *



"""
model_type="RBF_zeroprior"/"RBF_nonzeroprior"/"reconstructed"/"smooth_reconstructed"/"memory_enhanced"
return trained GP model
"""

if __name__=='__main__':
    
    initial_point_number=5
    # input_data,output_data=initial_sampling_fun(initial_point_number)
    

    """
    test 5 different kind of GP models
    """
# =============================================================================
#     # 1st model, GP with RBF kernel(zero mean function)
#     model1=train_GP(input_data,output_data,"RBF_zeroprior")
#     mean1,std1=model_prediction(input_data,model1)
# 
#     # 2nd model, GP with RBF kernel(machanictic mean function)
#     model2=train_GP(input_data,output_data,"RBF_nonzeroprior",machanistic_fun=prior_defult)
#     mean2,std2=model_prediction(input_data,model2) 
#     
#     # 3rd model, GP with reconstructed kernel
#     model3=train_GP(input_data,output_data,"reconstructed",machanistic_fun=prior_defult)
#     mean3,std3=model_prediction(input_data[0],model3)
# 
#     
#     # 4th model, GP with smooth_reconstructed kernel
#     model4=train_GP(input_data,output_data,"memory_enhanced",machanistic_fun=prior_defult,backward_step=1,initial_sampling_point=4)
#     mean4,std4=model_prediction(input_data,model4)
#     
#     #5th model, GP with memory_enhanced kernel
#     model5=train_GP(input_data,output_data,"memory_enhanced",machanistic_fun=prior_defult,print_flag=False)
#     mean5,std5=model_prediction(input_data,model5)
# =============================================================================
    


# =============================================================================
#     """
#     use different GP models to conduct BO
#     """
#     model_type="RBF_nonzeroprior"
#     opt_info_dict=run_BO(model_type,initial_point_number,max_iter=5,acquisition_function="UCB",
#                          initial_sampling_fun=initial_sampling_fun,bounds=bounds,truth_ground=Williams_Otto_profit,alpha=0.8,
#                          autostop_flag=False,machanistic_fun=prior_defult,beta=10,print_flag=True)
# 
# =============================================================================


    #track error trace
    model_type_list=["RBF_zeroprior","RBF_nonzeroprior","reconstructed"]
    BO_repete_num=50
    max_iter=10
    acquisition_function="EI"
    res_dict_all=bo_result_statistics(model_type_list,BO_repete_num,initial_point_number,max_iter,acquisition_function,
                                      initial_sampling_fun=initial_sampling_fun,bounds=bounds,truth_ground=Williams_Otto_profit,
                                      beta=2,print_flag=False,machanistic_fun=prior_defult)
    
    


















    
