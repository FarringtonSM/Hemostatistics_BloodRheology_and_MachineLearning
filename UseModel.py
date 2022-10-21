"""
Code for the paper ....
Sean Farrington, Soham Jariwala, Matt Armstrong, Ethan Nigro, Antony Beris, and Norman Wagner
Author: Sean Farrington
October 20th, 2022
 
This is the script to use for using the model created from main_
"""

#%% Importing necessary packages and functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import os
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" 

from sklearn.metrics import mean_squared_error as mse

# Method for saving figures
import os
# Where to save figure
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Remove Outliers"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Figures", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="svg", resolution=1200):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# For pretty matplotlib figures
import matplotlib as mpl
font = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = font
mpl.rcParams['mathtext.it'] = font+':italic'
mpl.rcParams['mathtext.bf'] = font+':bold'

SMALL = 10
MEDIUM = 12
BIG = 14

mpl.rc('font',size=BIG)
mpl.rc('axes',labelsize=BIG)
mpl.rc('xtick',labelsize=MEDIUM)
mpl.rc('ytick',labelsize=MEDIUM)
mpl.rc('legend',fontsize=MEDIUM)
mpl.rcParams['font.family'] = font


#%% Load the models using pickle
filename1 = 'finalized_model_yieldstress.sav'
filename2 = 'finalized_model_viscosity.sav'
filename3 = 'finalized_model_datasplit.pkl'

gp_yield = pickle.load(open(filename1,'rb'))
gp_visc = pickle.load(open(filename2,'rb'))
datasplits = pickle.load(open(filename3,'rb'))

X = datasplits['X']
y = datasplits['y']
X_test = datasplits['X_test']
y_test = datasplits['y_test']


#%% Make predictions on Casson parameters
yield_pred = gp_yield.predict(X_test)
yield_train_pred = gp_yield.predict(X) # Predicted training data

visc_pred = gp_visc.predict(X_test)
visc_train_pred = gp_visc.predict(X) # Predicted training data

#%% Some Data labels
yield_train = y[:,0]
yield_test = y_test[:,0]
visc_train = y[:,1]
visc_test = y_test[:,1]

#%% Metrics

# R**2 metric
r2_yield_test = gp_yield.score(X_test,y_test[:,0])
r2_yield_train = gp_yield.score(X,y[:,0])
r2_visc_test = gp_visc.score(X_test,y_test[:,1])
r2_visc_train = gp_visc.score(X,y[:,1])

# RMSE metric
RMSE_yield_train = mse(yield_train,yield_train_pred,squared=False)
RMSE_yield_test = mse(yield_test,yield_pred,squared=False)
RMSE_visc_train = mse(visc_train,visc_train_pred,squared=False)
RMSE_visc_test = mse(visc_test,visc_pred,squared=False)

df = pd.DataFrame({
    'R**2 Casson Yield Stress Training':[r2_yield_train],
    'R**2 Casson Viscosity Training':[r2_visc_train],
    'R**2 Casson Yield Stress Testing':[r2_yield_test],
    'R**2 Casson Viscosity Testing':[r2_visc_test]})
df = df.transpose()


print("Metrics for the model you are using:")
print(df)
print()

df = pd.DataFrame({
    'RMSE Casson Yield Stress Training':[RMSE_yield_train],
    'RMSE Casson Viscosity Training':[RMSE_visc_train],
    'RMSE Casson Yield Stress Testing':[RMSE_yield_test],
    'RMSE Casson Viscosity Testing':[RMSE_visc_test]})
df = df.transpose()


print()
print(df)
print()

i = int(input('Calcuate Casson Parameters? [0]\nSurface Plot? [1]\n'))

if i == 0:
    H = float(input("Hematocrit fraction (0 --> 1)?\n"))
    cf = float(input("Fibrinogen (g/dL)?\n"))
    
    XX = np.array([H,cf]).reshape(1,-1)
    ys, ys_std = gp_yield.predict(XX,return_std=True)
    mu, mu_std = gp_visc.predict(XX,return_std=True)
    
    print()
    print(f"Casson Yield Stress Prediction: {round(ys[0],2)} +/- {round(ys_std[0],2)}")
    print(f"Casson Viscosity Prediction: {round(mu[0],2)} +/- {round(mu_std[0],2)}")
    print()
else:
    #%% Surface plots
    plt.close('all')
    print('\nInput desired range')
    print('Hematocrit Healthy Range (Female): 0.36-0.47')
    print('Hematocrit Healthy Range (Male): 0.41-0.51')
    print('Fibrinogen Healthy Range: 0.150-0.350 g/dL')
    
    H_lower = float(input("Hematocrit (frac) Lower Bound? "))
    H_upper = float(input("Hematocrit (frac) Upper Bound? "))
    cf_lower = float(input("Fibrinogen (g/dL) Lower Bound? "))
    cf_upper = float(input("Fibrinogen (g/dL) Upper Bound? "))
    
    N = 15
    Hs = np.linspace(H_lower,H_upper,N)
    cfs = np.linspace(cf_lower,cf_upper,N)
    Hs, cfs = np.meshgrid(Hs,cfs)
    
    ys = np.zeros([N,N])
    ys_std = np.zeros([N,N])
    mu = np.zeros([N,N])
    mu_std = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            XX = np.array([Hs[i,j],cfs[i,j]]).reshape(1,-1)
            ys[i,j], ys_std[i,j] = gp_yield.predict(XX,return_std=True)
            mu[i,j], mu_std[i,j] = gp_visc.predict(XX,return_std=True)
    
    Hs = Hs*100
    cfs = cfs*1000
            
    from matplotlib import cm
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(Hs,cfs,ys,cmap=cm.coolwarm,
                           linewidth=0,antialiased=False)
    
    ax.set_xlabel('Hematocrit, %')
    ax.set_ylabel('Fibrinogen, mg/dL')
    ax.set_zlabel('Casson Yield Stress, mPa')
    
    plt.show()
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(Hs,cfs,mu,cmap=cm.coolwarm,
                           linewidth=0,antialiased=False)
    
    ax.set_xlabel('Hematocrit, %')
    ax.set_ylabel('Fibrinogen, mg/dL')
    ax.set_zlabel('Casson Viscosity, mPa.s')
    
    plt.show()
