# -*- coding: utf-8 -*-
"""
Script Name: dp_post_processing.py

Description:
    This script analyzes the motion of a double pendulum system. It reads position data
    (x, y) for both masses, computes angles and angular velocities, and saves
    them in a .csv file. To remove the constant portion of the experiment, the
    starting time t_0 is chosen as |omega_1(t_0)| >= {derivative_treshold}, 
    where omega_1 is the angular velocity of m1. 
    Finally, a trajectory plot is generated for manual verification.

Usage:
    Run the script with the input and output parameters. 

Dependencies:
    - numpy
    - pandas
    - matplotlib

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
experiment ='100deg5'
data_file_name = 'data.csv'
analysis_dir = 'C:\\CP_Analysis\\movies\\analysis';

#Output parameters
output_csv_name = 'angles.csv'
derivative_treshold = 0.75;  # [rad/s] the value of theta_1 derivative at t=0
flip_y              = False; # flip the y axis, same as in extracted data
convert_data_to_deg = False; # output theta and omega in deg and not rad
write_IC_in_deg     = True;  # write the IC in the figure in degrees

#################################################################
#                                                               #
#                    CODE STARTS HERE                           #
#                                                               #
#################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

load_file = os.path.join(analysis_dir,experiment,data_file_name) 
df = pd.read_csv(load_file)

fl = -1 if not flip_y else 1; #inicator to flip Y coordinate
theta_1 =  np.unwrap(np.atan2(fl*df.Y1,df.X1)) + np.pi/2;
theta_2 =  np.unwrap(np.atan2(fl*(df.Y2 - df.Y1),df.X2 - df.X1))+ np.pi/2;

if convert_data_to_deg:
    theta_1 = np.degrees(theta_1)
    theta_2 = np.degrees(theta_2)
    derivative_treshold = np.degrees(derivative_treshold)


omega_1 = np.gradient(theta_1,df.Time);
omega_2 = np.gradient(theta_2,df.Time);

#find starting position
idx_0 = np.where(np.abs(omega_1) > derivative_treshold)[0][0]

#save output data
dat = {'Time': df.Time[idx_0:] - df.Time[idx_0], 
       'theta1': theta_1[idx_0:], 
       'theta2': theta_2[idx_0:],
       'omega1': omega_1[idx_0:], 
       'omega2': omega_2[idx_0:]}
df_out = pd.DataFrame(dat)
df_out.to_csv(os.path.join(analysis_dir,experiment,output_csv_name), index=False)

# Produce analysis figure

if write_IC_in_deg:
    IC_text = (
    f'IC: $|\\omega_1(t = 0)| \geq$ {derivative_treshold} [rad/s] \n'
    f"$\\theta_1$ = {np.degrees(theta_1[idx_0]):.1f} [deg] , "
    f"$\\theta_2$ = {np.degrees(theta_2[idx_0]):.1f} [deg], "
    f"$\\omega_1$ = {np.degrees(omega_1[idx_0]):.1f} [deg/s], "
    f"$\\omega_2$ = {np.degrees(omega_2[idx_0]):.1f} [deg/s]"
    )
else:
    IC_text = (
    f'IC: |\\omega_1(t = 0)| \geq {derivative_treshold} [rad/s] \n'
    f"$\\theta_1$ = {(theta_1[idx_0]):.2f} [rad], "
    f"$\\theta_2$ = {(theta_2[idx_0]):.2f} [rad], "
    f"$\\omega_1$ = {(omega_1[idx_0]):.2f} [rad/s], "
    f"$\\omega_2$ = {(omega_2[idx_0]):.2f} [rad/s]"
    )
title = f'Experiment {experiment}\n {IC_text}'

time = df.Time - df.Time[idx_0]
theta_units = 'deg' if convert_data_to_deg else 'rad' 
fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex = True) 

ax[0].plot(time, theta_1, color = 'blue')
ax[0].set_ylabel('$\\theta_1$'+f' [{theta_units}]')

ax[1].plot(time, (theta_2), color = 'blue')
ax[1].set_ylabel('$\\theta_2$'+f' [{theta_units}]')

ax[2].plot(time, (omega_1), color = 'blue')
ax[2].set_ylabel('$\dot{\\theta}_1$'+f' [{theta_units}/s]')

ax[3].plot(time, (omega_2), color = 'blue')
ax[3].set_ylabel('$\dot{\\theta}_2$'+f'[{theta_units}/s]')

ax[3].set_xlabel('Time [s]')
for i in range(4):
    ax[i].axvline(0, color = 'red', linestyle = '--')
    ax[i].grid()
#ax[0].set_xlim([0,1])
fig.suptitle(title)
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir,experiment,'angels.pdf'));
plt.show()

