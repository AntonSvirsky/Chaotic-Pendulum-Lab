# -*- coding: utf-8 -*-
"""
Script Name: dp_lyapunov.py

Description:
    This sctipt calulates the largest Lyapunov exponent (LLE) for a set of experiments
    with similar inital conditions. Given a set, it calulates the LLE for 
    each pair and saves them as an .csv file.

Usage:
    Run the script with the input and output parameters. 
    
    
Main Parameters:
    variabels : list of strings
        The variables used to calulate distance. Can be any combination of
        ['theta1', 'theta2', 'omega1', 'omega2']. For example if we wish to 
        define the distance to only consider variations in 'theta1' we set 
        variabels = ['theta1'].
    
    apply_savgol : bool
       If true, a smoothing filter is applied to the distance data before fiting.
        
    start_time : float
        inital time (sec) when to start the exponential fit for the distance
    
    max_d: float
        the maximal distance for which to fit exponential growth. Note it refers
        to the distance value after the normalization (if normalize = True). 
    
    allow_itercept: bool
        wheter to allow an intercept in the exponential fit. If true, data is 
        dited to d(t) = exp(a*t + b), otherwise to d(t) = exp(a*t). Note
        if normalize = False, this should be true. 
        
    normalize: boole
        wheter to normalzie the distance data to the inital value. 
   

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - scipy

Author: Anton S.
Date: 2025-06-12
Version: 1.0
"""

#################################################################
#                                                               #
#                        PARAMETERS                             #
#                                                               #
#################################################################

#Input parameters
set_of_experiments = ['100deg1',
                      '100deg2',
                      '100deg3',
                      '100deg4',
                      '100deg5']    

data_file_name = 'angles.csv'
analysis_dir   = 'C:\\CP_Analysis\\movies\\analysis';

#Distance parameters
variabels    = ['theta1', 'theta2', 'omega1', 'omega2']

#Filter parameters
apply_savgol = True; #filter data with Savitsky-Golay filter before fiting
savgol_window = 125; #filter windows in sample space, must be odd
savgol_ord    = 1;   #order of filter

#Fiting parameters
start_time = 0; #inital time for fit
max_d      = 1; #linear fit will be applied for d(t) < max_d
allow_itercept = True;
normalize      = False;  #normalzie data before fitting

#Output parameters
set_name       = 'LLE_100deg'; #folder name for the output data 
output_dir     = analysis_dir; #dir of output data
analysis_plots = True;        

#################################################################
#                                                               #
#                    CODE STARTS HERE                           #
#                                                               #
#################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from datetime import datetime

#make directory if it dose not exist
os.makedirs(os.path.join(output_dir,set_name), exist_ok=True) 
#create dataframe
df_LLE = pd.DataFrame(columns=['exp_1', 'exp_2', 'LLE', 'R2'])
number_of_exp = len(set_of_experiments)

for i in range(number_of_exp):
    for j in range(i+1, number_of_exp):
        #extract data for two experimetns
        exp_name_1 = set_of_experiments[i]
        exp_name_2 = set_of_experiments[j]
        
        load_file_1 = os.path.join(analysis_dir,exp_name_1,data_file_name) 
        load_file_2 = os.path.join(analysis_dir,exp_name_2,data_file_name) 
        
        df_1=  pd.read_csv(load_file_1)
        df_2=  pd.read_csv(load_file_2)
        
        max_time = np.min([len(df_1.Time), len(df_2.Time)])
        
        X1 = df_1[variabels][:max_time]
        X2 = df_2[variabels][:max_time]
        
        #compute distance
        dist = np.sqrt(np.sum((X1 - X2)**2,axis = 1))
        dist = dist/dist[0] if normalize else dist
        time = df_1.Time[:max_time]
        
        #apply filtering
        if apply_savgol:
            dist_smooth = savgol_filter(dist, window_length=savgol_window, polyorder=savgol_ord)
            dist_smooth = dist_smooth/dist_smooth[0] if normalize else dist_smooth
            dist = dist/dist_smooth[0] if normalize else dist
            fit_distance = dist_smooth
            
        else:
            fit_distance = dist
            
        #linear fit
        
        if not allow_itercept:
            linear_fix_inter = lambda x, slope: slope * x 
        else:
            linear_fix_inter = lambda x, slope, inter: slope * x +inter
        
        
        # Fit the data
        inital_time = np.where(time>=start_time)[0][0]
        if np.max(fit_distance) > max_d:
            final_time = np.where(fit_distance > max_d)[0][0]
        else: 
            final_time = -1;
    
        fit_log_distance = np.log(fit_distance[inital_time:final_time])
        fit_log_distance = np.nan_to_num(fit_log_distance, nan=0.0)
        fit_time     = time[inital_time:final_time] - time[inital_time]
        
        params, _ = curve_fit(linear_fix_inter, fit_time, fit_log_distance)
        exponent = params[0]
        
        # Compute R^2
        y     = fit_log_distance
        y_fit = linear_fix_inter(fit_time, *params)
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
            
        
        #add to dataframe
        df_LLE.loc[len(df_LLE)] = [exp_name_1, exp_name_2, exponent,r_squared]
            
        
        if analysis_plots:
            #plot figure 
            pair_name = f'{exp_name_1}_{exp_name_2}'
            title = f'Pair: {pair_name} \n Lyapunov exponent'
            
            fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex = True) 
            ax[0].semilogy(time, dist, label  = 'raw')
            ax[0].plot(time, dist_smooth, label  = f'savgol ({savgol_window},{savgol_ord})')
            ax[0].plot(fit_time + time[inital_time], np.exp(y_fit), 
                       color = 'red',
                       label  = f'$\\lambda$ = {exponent:.3f} ($R^2 = {r_squared:.2f}$)')
            
            ax[0].set_ylabel('$d(t)/d(0)$ ')
            ax[0].legend(loc = 4)
            ax[0].grid()
            
            ax[1].plot(time, df_1['theta1'][:max_time], color = 'blue', label = exp_name_1)
            ax[1].plot(time, df_2['theta1'][:max_time], color = 'red', label = exp_name_2)
            ax[1].set_ylabel('$\\theta_1(t)$ [rad]')
            ax[1].legend(loc = 4)
            ax[1].grid()
            
            ax[2].plot(time, df_1['theta2'][:max_time], color = 'blue', label = exp_name_1)
            ax[2].plot(time, df_2['theta2'][:max_time], color = 'red', label = exp_name_2)
            ax[2].set_ylabel('$\\theta_2(t)$ [rad]')
            ax[2].legend(loc = 4)
            ax[2].grid()
            ax[2].set_xlabel('Time [sec]')
            
            fig.suptitle(title)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir,set_name,f'{pair_name}.pdf'));
            plt.show()


#finally save data frame 

#write meta data file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

with open(os.path.join(output_dir,set_name,'LLE_metadata.txt'), "w") as f:
    f.write("# LLE data for double pendulum\n")
    f.write(f"# date: {current_time}\n")
    f.write(f"# set of experiments, {set_of_experiments}\n")
    f.write(f"# experiments directory, {analysis_dir}\n")
    f.write(f"# variables for distance:, {variabels}\n")
    f.write(f"# maximal distance for fit:, {max_d}\n")
    f.write(f"# initial time for fit:, {start_time}\n")
    f.write(f"# intercept assumed:, {allow_itercept}\n")
    if apply_savgol:
        f.write(f"# Filter: Savitzky-Golay, window={savgol_window}, order={savgol_ord}\n")
    
#write dataframe
df_LLE.to_csv(os.path.join(output_dir,set_name,'LLE.csv'), index=False)





