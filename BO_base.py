# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:17:47 2025

@author: Xinyu Cao
"""

import numpy as np
from gp_model_base import train_GP,model_prediction
import torch
from botorch.optim import optimize_acqf
import datetime
import os
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore")



def check_convergence(iteration, max_iter, input_points=None,autostop_flag=False):
    current_input=input_points[-1]
    previous_input=input_points[-2]
    f_change=np.linalg.norm(previous_input-current_input)
    if autostop_flag:
        convergence_flag= (iteration >= max_iter) or (f_change<1e-1 and iteration>3)
    else:
        convergence_flag= (iteration >= max_iter)
    return convergence_flag




def run_BO(model_type,initial_sampling_point_num,max_iter,acquisition_function,initial_sampling_fun,bounds,truth_ground,**kwargs):
    
    n_iter=0
    read_sampling_index=kwargs.get('read_sampling_index', -1)
    machanistic_fun=kwargs.get('machanistic_fun', None)
    alpha=kwargs.get('alpha', 0.8)
    backward_step=kwargs.get('backward_step', 2)
    beta = kwargs.get('beta', 2**0.5) #hyoerparameter of UCB
    print_flag = kwargs.get('print_flag', True)
    autostop_flag = kwargs.get('autostop_flag', False)
    initial_sampling=kwargs.get('initial_sampling', None)
    decay_factor=kwargs.get('decay_factor', 0.9)
    
    all_model_list=[]
    if read_sampling_index==-1:
        if initial_sampling==None:
            X,y=initial_sampling_fun(initial_sampling_point_num)
        else:
            X,y=initial_sampling
    else:
        folder = "BO"
        file_path = os.path.join(folder, f"initial_sample_{initial_sampling_point_num}.npz")
        if os.path.exists(file_path):
            print("reading random samples ["+str(read_sampling_index)+"] ...")
            data = np.load(file_path)
            X = data['u_series'][read_sampling_index]
            y = data['yield_'][read_sampling_index]
        else:
            print("creating random samples...")
            u_series = []
            yield_ = []
            for i in range(100):
                u_series_ap,yield_ap=initial_sampling_fun(initial_sampling_point_num)
                u_series.append(u_series_ap)
                yield_.append(yield_ap)
            np.savez(file_path, u_series=u_series, yield_=yield_)
            data = np.load(file_path)
            X = data['u_series'][read_sampling_index]
            y = data['yield_'][read_sampling_index]
            
        
    # Loop to optimize and check convergence
    while not check_convergence(n_iter,max_iter,X,autostop_flag):
        
        # Fit model
        model_dict=train_GP(X,y,model_type,print_flag,machanistic_fun=machanistic_fun,alpha=alpha,backward_step=backward_step,initial_sampling_point=initial_sampling_point_num,decay_factor=decay_factor,bounds=bounds)
        gp_model=model_dict["model"]
        all_model_list.append(model_dict)

        # Acquisition function
        if acquisition_function=="EI":
            acq_func = ExpectedImprovement(model=gp_model, best_f=y.max())
        elif acquisition_function=="UCB":
            acq_func = UpperConfidenceBound(model=gp_model, beta=beta**2)


        # Optimize acquisition function.
        # float32 GPs can produce NaN gradients in BoTorch's L-BFGS-B; since the
        # random restarts are unseeded, simply retrying with fresh initial
        # conditions almost always avoids the offending point.
        new_X = None
        for _attempt in range(10):
            try:
                new_X, _ = optimize_acqf(
                    acq_func,
                    bounds=torch.tensor(bounds.T, dtype=torch.float32),
                    q=1,
                    num_restarts=10,
                    raw_samples=200,
                )
                break
            except Exception as e:
                if _attempt == 9:
                    raise
                print(f"  optimize_acqf retry {_attempt+1}/10 after: {type(e).__name__}")

        # Evaluate real function
        new_y_val = truth_ground(new_X)
        new_y = torch.tensor([[new_y_val]], dtype=torch.float32)

        # Update dataset
        if isinstance(new_X, np.ndarray):
            new_X = torch.tensor(new_X, dtype=torch.float32)  #Ensure new_X is a Tensor with correct type
        else:
            new_X = new_X.clone().detach().float()  #Ensure new_X is a clean Tensor
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)  #Ensure X is a Tensor
        else:
            X = X.clone().detach().float()  #Ensure X is a clean Tensor
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)  #Ensure y is a Tensor
        else:
            y = y.clone().detach().float()  #Ensure y is a clean Tensor
        
        if y.dim() == 1:
            y = y.unsqueeze(-1)  #Ensure y has shape [N,1]
        
        X = torch.cat([X, new_X], dim=0)  #Concatenate X
        y = torch.cat([y, new_y], dim=0)  #Concatenate y

        n_iter=n_iter+1
        # print(f"Iter {n_iter}: X = {new_X.numpy()}, y = {new_y_val:.4f}")
        print(f"Iter {n_iter}")
        
        opt_info_dict={"aquisition_fun":acquisition_function,"initial_sampling_point_num":initial_sampling_point_num,"beta":beta,
                         "machanistic_fun":machanistic_fun,"X":X.numpy(),"y":y.numpy(),"all_model_list":all_model_list}

    return opt_info_dict




def bo_result_statistics(model_type_list, BO_repete_num, initial_point, max_iter, acquisition_function,initial_sampling_fun,bounds,truth_ground, **kwargs):
    
    res_dict_all = []
    input_data_size=len(bounds)
    date_str = datetime.date.today().strftime("%m%d")   # e.g. '0726'
    base_path = "BO"
    kwargs_dict = kwargs

    # Build the acquisition tag used in the filename:
    #   EI  -> "EI"
    #   UCB -> "UCB_beta{beta}"   (beta only matters for UCB)
    if acquisition_function == "UCB":
        beta = kwargs.get("beta", 2**0.5)
        acq_tag = f"UCB_beta{beta}"
    else:
        acq_tag = acquisition_function

    # Make sure the directory exists
    os.makedirs(base_path, exist_ok=True)

    for model_index, model_type in enumerate(model_type_list):

        print("Start sampling model: " + str(model_type) + "...")

        input_data_all = np.zeros([BO_repete_num, max_iter + initial_point, input_data_size])
        output_data_all = np.zeros([BO_repete_num, max_iter + initial_point])

        for BO_repete_index in range(BO_repete_num):
            opt_info_dict = run_BO(model_type, initial_point, max_iter, acquisition_function,initial_sampling_fun,bounds,truth_ground, read_sampling_index=BO_repete_index,**kwargs)
            input_data_all[BO_repete_index] = opt_info_dict["X"]
            output_data_all[BO_repete_index] = opt_info_dict["y"].reshape(-1)

        res_dict = {
            "output_data_all": output_data_all,
            "model_type": model_type
        }
        res_dict_all.append(res_dict)

        # Save one file per model:
        #   {model_type}_{date}_{acq_tag}.npy
        #   e.g. RBF_zeroprior_0726_UCB_beta1.npy  /  RBF_zeroprior_0726_EI.npy
        model_save_dict = {
            "output_data_all": output_data_all,
            "model_type": model_type,
            "BO_repete_num": BO_repete_num,
            "initial_point": initial_point,
            "max_iter": max_iter,
            "acquisition_function": acquisition_function,
        }
        model_save_dict.update(kwargs_dict)

        filename = f"{model_type}_{date_str}_{acq_tag}.npy"
        full_path = os.path.join(base_path, filename)
        np.save(full_path, model_save_dict)   # overwrites if the same file already exists
        print(f"✅ saved: {full_path}")

    return res_dict_all



"""
The next functions are for visualization
"""

def data_prepare(opt_info_dict,true_model,index):
    
    aquisition_fun = opt_info_dict["aquisition_fun"]
    initial_sampling_point_num = opt_info_dict["initial_sampling_point_num"]
    machanistic_fun = opt_info_dict["machanistic_fun"]
    input_data = opt_info_dict["X"]
    output_data = opt_info_dict["y"]
    all_model_list = opt_info_dict["all_model_list"]
    beta=opt_info_dict["beta"]

    sample_len=index+initial_sampling_point_num
    current_model=all_model_list[index]["model"]
    current_model_dict=all_model_list[index]
    u_next=input_data[sample_len]
    yield_next=output_data[sample_len]
    y_best=output_data[:sample_len].max()
    
    #initialize the aquisition function
    if aquisition_fun=="EI":
        aquisition_fun_pre = ExpectedImprovement(model=current_model, best_f=y_best, maximize=True)
    elif aquisition_fun=="UCB":
        aquisition_fun_pre = UpperConfidenceBound(model=current_model, beta=beta)
    
    input_X = np.linspace(1,1.5,50)#x 
    input_Y = np.linspace(0.1,6,50)#y 
    input_X_mesh, input_Y_mesh = np.meshgrid(input_X, input_Y)
    points_np = np.stack([input_X_mesh.ravel(), input_Y_mesh.ravel()], axis=1)
    # points_torch = torch.tensor(points_np, dtype=torch.float32)
    points_torch = torch.tensor(points_np, dtype=torch.float32).unsqueeze(1)
    
    # 1. Compute acquisition function values in batch
    acquisition_vals = aquisition_fun_pre(points_torch).detach().cpu().numpy().reshape(input_X_mesh.shape)
    
    # 2. Predict GP mean and standard deviation in batch 
    gp_mean_vals, gp_std_vals = model_prediction(points_np, current_model_dict)
    gp_mean_vals = gp_mean_vals.reshape(input_X_mesh.shape)
    gp_std_vals = gp_std_vals.reshape(input_X_mesh.shape)
        
    # 3. Compute true values over the mesh (vectorized)
    # Make sure true_model supports numpy array inputs
    true_model_vec = np.vectorize(true_model)
    yield_true = true_model_vec(input_X_mesh, input_Y_mesh)
    
    # 4. Compute mechanistic model output in batch
    # Ensure machanistic_fun supports batch input shape (N, 2)
    yield_prior = machanistic_fun(points_np).reshape(input_X_mesh.shape)
    
    # Final assignments to result matrices
    yield_mesh_ac = acquisition_vals
    yield_mesh_GP = gp_mean_vals
    yield_mesh_sd = gp_std_vals
    yield_mesh_true = yield_true
    yield_mesh_prior = yield_prior
    

    lower_bound=yield_mesh_GP-1.96*yield_mesh_sd
    upper_bound=yield_mesh_GP+1.96*yield_mesh_sd

    trace_prior= go.Surface(x=input_X, y=input_Y, z=yield_mesh_prior, colorscale=[[0,'#ADD8E6'],[0.5,'#ADD8E6'], [1,'#ADD8E6']], 
                                 opacity=0.95,name="prior", showscale=False,showlegend=True)

    trace_predicted = go.Surface(x=input_X, y=input_Y, z=yield_mesh_GP, colorscale=[[0,'#29A3A3'],[0.5,'#29A3A3'], [1,'#29A3A3']], 
                                 opacity=0.8,name="predicted model", showscale=False,showlegend=True)

    trace_true = go.Surface(x=input_X, y=input_Y, z=yield_mesh_true, colorscale=[[0,'#333333'],[0.5,'#333333'], [1,'#333333']], 
                                 opacity=1,name="true model", showscale=False,showlegend=True)
    
    trace_upper = go.Surface(x=input_X, y=input_Y, z=upper_bound, colorscale=[[0,'#BBBBBB'],[0.5,'#BBBBBB'], [1,'#BBBBBB']], 
                                 opacity=0.5,name='upper bound(95% CI)', showscale=False,showlegend=True)
    
    trace_lower = go.Surface(x=input_X, y=input_Y, z=lower_bound, colorscale=[[0,'#BBBBBB'],[0.5,'#BBBBBB'], [1,'#BBBBBB']], 
                                 opacity=0.5,name='lower bound(95% CI)', showscale=False,showlegend=True)
    
    trace_sample=go.Scatter3d(x=input_data[:sample_len,0], y=input_data[:sample_len,1], z=output_data[:sample_len],  mode="markers", opacity=0.5,marker_color="rgba(244, 198, 147, 1)",
                            marker_size=4, marker_line_width=0,  name='samples', showlegend=True)
    
    trace_nextpoint=go.Scatter3d(x=[u_next[0]], y=[u_next[1]], z=[yield_next],  name='next point',  mode="markers", opacity=0.5,marker_color="#4F9F4F",marker_size=4, showlegend=True)

    trace_nextpoint_2D=go.Scatter(x=[u_next[0]], y=[u_next[1]],  name='next point',  mode="markers", opacity=1,marker_color="#4F9F4F",marker_size=18, showlegend=True)

    trace_sample_2D=go.Scatter(x=input_data[:sample_len,0], y=input_data[:sample_len,1], mode="markers",name="previous sampling point ", marker={
        "size": 18,"color": "#4F9F4F","opacity":0.3}, showlegend=True)
    
    trace_sample_2D_best=go.Scatter(x=[1.5], y=[0.1], mode="markers",name="optimum", marker_symbol="star", marker={
        "size": 18,"color":"#AF0C2D" }, showlegend=True)
    
    data_plot = [trace_prior,trace_lower,trace_upper,trace_predicted,trace_true,trace_sample,trace_nextpoint,
                 trace_sample_2D,trace_nextpoint_2D,trace_sample_2D_best]
    return data_plot



def BO_data_plot_figure(data_plot):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                subplot_titles=("3-D model","2-D Sample Point"),
                specs=[[{'type': 'surface','colspan': 1, 'rowspan': 1},{'colspan': 1,'rowspan': 1}]],
                horizontal_spacing=0.1)

    col_list=[1,1,1,1,1,1,1,2,2,2]
    for i,ele in enumerate(data_plot):
        fig.add_trace(ele, row=1, col=col_list[i])
        
    fig.update_layout(
        title='profit function',
        autosize=False,
        width=1000,
        height=500,
        margin=dict(r=10, l=50, b=0, t=50),
        showlegend=True,
        scene=dict(
            xaxis=dict(
                backgroundcolor="white",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
                title="initial concentration",
                autorange="reversed"
            ),
            yaxis=dict(
                backgroundcolor="white",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
                title="flowrate",
                autorange="reversed"
            ),
            zaxis=dict(
                backgroundcolor="white",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
                title="yield"
            ),
        ),
        xaxis=dict(
            title="initial concentration",  # 修改 x 轴标题
            gridcolor="lightgray",
            showgrid=True,  # 显示网格
            zeroline=True,  # 显示零线
            showticklabels=True,  # 显示刻度
             dtick=0.1
        ),
        yaxis=dict(
            title="flowrate",  # 修改 y 轴标题
            gridcolor="lightgray",
            showgrid=True,
            zeroline=True,
            showticklabels=True
        ),
        
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,highlightcolor="#130919", project_z=True))
    fig.show()
    
    return fig
    
    

def BO_data_plot_animation(fig,opt_info_dict,true_model,iterative_times):

    layout_animated=[]
    frame_list=[]
    layout=dict(title='yield function', autosize=False,width=1200,height=700,margin=dict(r=10, l=50, b=0, t=50),showlegend=True)
    
    for i in range(iterative_times):
        layout_animated.append(layout)
    
    
    for i in range(iterative_times):
        data_plot=data_prepare(opt_info_dict,true_model,i) 
        frames = dict(name = i,
                      data=data_plot,
                      layout = layout_animated[i],
                      )
        frame_list.append(frames) 
    fig.update(frames=frame_list)
    
    updatemenus = [dict(type='buttons',
                 buttons=[dict(label='Play', method='animate',
                                args=[None, dict(frame=dict(duration=2000, redraw=True),fromcurrent=True,transition=dict(duration=1000,
                                              easing='quadratic-in-out'))]),
                           dict(label='Pause',method='animate',args=[[None],
                                           dict(frame=dict(duration=0, redraw=True),
                                              transition=dict(duration=0),                                             
                                              mode='immediate')])],
                 direction= 'left', pad=dict(r= 10, t=87), 
                 showactive =False, x= 0.1, y= 0, xanchor= 'right', yanchor= 'top')]
    
    sliders = [{'active': 0, 'yanchor': 'top', 'xanchor': 'left', 
                'currentvalue': {'font': {'size': 16}, 'prefix': 'Step: ', 'visible': True, 'xanchor': 'right'},
                'transition': {'duration': 1000.0, 'easing': 'cubic-in-out'},'pad': {'b': 10, 't': 50}, 
                'len': 0.9, 'x': 0.1, 'y': 0, 
                'steps': [{'args': [[i3], {'frame': {'duration': 900.0, 'easing': 'quadratic-in-out', 'redraw': True},
                                         'transition': {'duration': 900, 'easing': 'quadratic-in-out'}}],
                                        'label': str(i3), 'method': 'animate'} for i3 in range(iterative_times)]     
               }]
    
    fig.update_layout(updatemenus=updatemenus, sliders=sliders,showlegend=True,height=700,width=800)
    fig.show()
    
    return fig



def plot_acquisition_surface(opt_info_dict, true_model, index):
    # Extract optimization settings and model components
    aquisition_fun = opt_info_dict["aquisition_fun"]
    initial_sampling_point_num = opt_info_dict["initial_sampling_point_num"]
    all_model_list = opt_info_dict["all_model_list"]
    input_data = opt_info_dict["X"]
    output_data = opt_info_dict["y"]
    beta = opt_info_dict["beta"]

    # Determine how many samples have been collected up to this index
    sample_len = index + initial_sampling_point_num
    current_model = all_model_list[index]["model"]
    y_best = output_data[:sample_len].max()

    # Initialize acquisition function using either 'best_f' or 'beta'
    try:
        aquisition_fun_pre = aquisition_fun(model=current_model, best_f=y_best, maximize=True)
    except:
        aquisition_fun_pre = aquisition_fun(model=current_model, beta=beta)

    # Create a grid of input points for acquisition function evaluation
    input_X = np.linspace(1, 1.5, 50)
    input_Y = np.linspace(0.1, 6, 50)
    input_X_mesh, input_Y_mesh = np.meshgrid(input_X, input_Y)
    points_np = np.stack([input_X_mesh.ravel(), input_Y_mesh.ravel()], axis=1)
    points_torch = torch.tensor(points_np, dtype=torch.float32).unsqueeze(1)

    # Evaluate the acquisition function over the mesh grid
    acquisition_vals = aquisition_fun_pre(points_torch).detach().cpu().numpy().reshape(input_X_mesh.shape)

    # Create 3D surface plot for the acquisition function values
    trace_acquisition = go.Surface(
        x=input_X,
        y=input_Y,
        z=acquisition_vals,
        colorscale='Viridis',
        opacity=0.9,
        name="Acquisition Function",
        showscale=True
    )

    # Set up the figure with axis titles and layout
    fig = go.Figure(data=[trace_acquisition])
    fig.update_layout(
        title=f'Acquisition Function Surface (Step {index})',
        scene=dict(
            xaxis_title='Initial Concentration',
            yaxis_title='Flowrate',
            zaxis_title='Acquisition Value',
            xaxis=dict(autorange="reversed"),
            yaxis=dict(autorange="reversed")
        ),
        width=700,
        height=600,
        margin=dict(r=10, l=10, b=10, t=50),
        paper_bgcolor="white"
    )
    
    fig.show()
    return fig



def plot_bo_comparison(info_dict, reference_max=None, withdraw_model_index=[],date=str(datetime.date.today())[5:],draw_initial_point=True,draw_iter_step=None,save_path=None, yaxis_lower_bound=1.2, band="se"):
    
    fig = go.Figure()

    data_list=info_dict["res_dict_all"]
    acquisition_function=str(info_dict["acquisition_function"])
    max_iter=str(info_dict["max_iter"])
    initial_point=str(info_dict["initial_point"])
    
    if draw_iter_step==None:
        total_steps = int(max_iter)
    else:
        total_steps=int(min(draw_iter_step,int(max_iter)))
        
    
    if draw_initial_point:
        total_steps+=int(initial_point)
        
    steps = np.arange(1, total_steps + 1)
    minimum=1
    maximum=0
    
    
    color_list = [
        '#4682B4',  # Steel Blue
        '#8B4513',  # Chocolate
        '#C9A227',  # Dark Yellow (goldenrod)
        '#0fa107',  # Green
        '#7D3C98',  # Amethyst Purple
        '#6A5ACD',  # Slate Blue
        '#008B8B',  # Dark Cyan
        '#B22222',  # Firebrick
    ]


    for model_index, model in enumerate(data_list):
        output_data_all = np.array(model["output_data_all"], dtype=float)
        model_type = model["model_type"]

        # best-so-far trajectory
        best_so_far_all = np.maximum.accumulate(output_data_all, axis=1)

        # error: reference max - current best
        if reference_max is not None:
            error_so_far_all = np.abs(reference_max - best_so_far_all)
        else:
            raise ValueError("reference_max must be provided to compute error.")

        # uncertainty band: "se" -> mean +/- standard error, "std" -> mean +/- std,
        # "iqr" -> median with 25th-75th percentile band (matches a boxplot's box)
        n_reps = error_so_far_all.shape[0]
        if band == "iqr":
            center = np.median(error_so_far_all, axis=0)
            low = np.percentile(error_so_far_all, 25, axis=0)
            high = np.percentile(error_so_far_all, 75, axis=0)
        else:
            center = np.mean(error_so_far_all, axis=0)
            spread = np.std(error_so_far_all, axis=0)
            if band == "se":
                spread = spread / np.sqrt(n_reps)
            low, high = center - spread, center + spread
        if not draw_initial_point:
            s0 = int(initial_point) - 1
            center, low, high = center[s0:], low[s0:], high[s0:]
        minimum = min(minimum, min(center))
        maximum = max(maximum, max(center))
        mean_error = center
        std_error = high - center


        color = color_list[model_index % len(color_list)]
        rgba_color = colors.to_rgba(color, alpha=1)
        rgba_color_s=colors.to_rgba(color, alpha=0.2)
    
        rgba_color_str = f'rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})'
        rgba_color_std=f'rgba({int(rgba_color_s[0]*255)}, {int(rgba_color_s[1]*255)}, {int(rgba_color_s[2]*255)}, {rgba_color_s[3]})'


        if model_index not in withdraw_model_index:

            if model.get("label") is not None:
                model_name = model["label"]
            elif model_type=="RBF_zeroprior":
                model_name="zero-prior"
            elif model_type=="RBF_nonzeroprior":
                model_name="mech-prior"
            elif model_type=="multi_fidelity":
                model_name="multi-fidelity"
            else:
                model_name="modified"
                
            
            
             # Vertical error bars (whiskers) at each point; no shaded band, no bound lines
            upper_bound = np.maximum(high, 1e-12)
            lower_bound = np.maximum(low, 1e-12)

            fig.add_trace(go.Scatter(
                x=steps,
                y=mean_error,
                mode='lines+markers',
                name=model_name,
                line=dict(color=color),
                marker=dict(size=10),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=upper_bound - mean_error,
                    arrayminus=mean_error - lower_bound,
                    visible=True,
                    color=rgba_color_str,
                )
            ))
                
            
    if draw_initial_point:
        # Add background fill for initial sampling period
        fig.add_shape(
            type="rect",
            x0=0.1, y0=0.0001, x1=initial_point, y1=maximum*5,
            fillcolor="lightgray",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
        
        
        y_place=-np.log(maximum*1.2) / np.log(0.1)-0.03
        
        y_place=y_place
        
        
    #     fig.add_annotation(
    #     x=3.5, y=y_place,
    #     text="Initial Sampling",
    #     showarrow=False,
    #     font=dict(size=16, color="blue"),
    #     bgcolor="rgba(255,255,255,0.8)",
    #     bordercolor="lightblue",
    #     borderwidth=1
    # )
        

    fig.update_layout(
        font=dict(family="Times New Roman", size=24),
        title=dict(text="<b>Distance to Optimum (acq: "+acquisition_function+")</b>",
                   x=0.5, xanchor='center', font=dict(size=30)),
        yaxis_type="log",
        yaxis_range=[y_place-yaxis_lower_bound, y_place+0.2],
        xaxis_range=[0.9, total_steps+0.5],
        plot_bgcolor='white',
        template="plotly_white",
        legend=dict(x=0.42, y=0.99, font=dict(size=30)),
        width=600,  # Set the width
        height=800,  # Set the height
        showlegend=True,
        xaxis=dict(
            title=dict(text="<b>Sampling Number</b>", font=dict(size=30)),
            dtick=1, showgrid=True, gridcolor='lightgray',
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            title=dict(text="<b>|Best Output - Reference Optimum|</b>", font=dict(size=30)),
            dtick=1, showgrid=True, gridcolor='lightgray',
            tickfont=dict(size=24),
        ),
    )
    
    if save_path==None:
        # Match the .npy naming: trackplot_{date}_EI.html or
        # trackplot_{date}_UCB_beta{beta}.html
        if acquisition_function == "UCB":
            beta = info_dict.get("beta", None)
            acq_tag = f"UCB_beta{beta}" if beta is not None else "UCB"
        else:
            acq_tag = acquisition_function
        save_path = "BO/trackplot_" + date + "_" + acq_tag + ".html"
    else:
        save_path="BO/"+save_path+"_trackplot.html"
    fig.write_html(save_path)
    # high-resolution raster (PNG) for publication
    try:
        fig.write_image(save_path.replace(".html", ".png"), scale=4)
    except Exception as e:
        print(f"[image export skipped: {e}]")
    fig.show()
    
    
    
if __name__ == "__main__":
    
    os.chdir(r"2william_otto")
    date=str(datetime.date.today())[5:]
    date="09-23"
    save_path="BO/"+date+"data_list.npy"
    info_dict=np.load(save_path,allow_pickle=True).item()
    plot_bo_comparison(info_dict,reference_max=191.02,date=date,withdraw_model_index=[],draw_initial_point=True,draw_iter_step=10,save_path=None)












