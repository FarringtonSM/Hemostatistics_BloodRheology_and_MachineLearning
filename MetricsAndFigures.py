"""
Author: Sean Farrington
Thu Jun  16

# Apostolidis Model Pseudo Data Machine Learning
 
# Gaussian Process Regression

# This script includes the necessary hyperparameter tuning


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


print()
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

#%% Plotting predicted vs actual for the Cassson yield stress
fig, ax = plt.subplots(figsize=(5,4))
p1, = ax.plot(yield_test,yield_pred,
        's',
        markersize = 5,
        color='red',
        label='Testing Data',
        zorder=10)
p2, = ax.plot(yield_train,yield_train_pred,
        'o',
        markersize = 3,
        label = 'Training Data',
        color='blue')
lims = ax.get_xlim()

ax.plot(lims,lims,color='black')

ax.set_xlabel('Actual Casson Yield Stress')
ax.set_ylabel('Predicted Casson Yield Stress')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(lims)
ax.set_ylim(lims)

ax.legend(handles=[p1,p2])

save_fig("Casson Yield Stress")
plt.show()
#%% Plotting predicted vs actual Casson viscosity
lims = [0.9*min(visc_test),1.1*max(visc_test)]

fig, ax = plt.subplots(figsize=(5,4))
p1, = ax.plot(visc_test,visc_pred,
              's',
              markersize= 5,
              color='red',
              label='Testing Data',
              zorder = 10)
p2, = ax.plot(visc_train,visc_train_pred,
              'o',
              markersize = 3,
              label='Training Data',
              color='blue')

lims = ax.get_xlim()
ax.plot(lims,lims,color='black')

ax.set_xlabel('Actual Casson Viscosity')
ax.set_ylabel('Predicted Casson Viscosity')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(lims)
ax.set_ylim(lims)

ax.legend(handles=[p1,p2])

save_fig("Casson Viscosity")
plt.show()
#%% Surface plots
N = 15
Hs = np.linspace(0.36,0.51,N)
cfs = np.linspace(0.150,0.350,N)
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
        
Hs = Hs.flatten()*100
cfs = cfs.flatten()*1000
ys = ys.flatten()
ys_std = ys_std.flatten()
mu = mu.flatten()
mu_std = mu_std.flatten()

from Apostolidis import Apostolidis
p1 = Apostolidis(Hs/100,cfs/1000)
ys_apost,mu_apost,a,b = p1.apostFunc()

ys_apost = np.array([ys_apost]).reshape(N**2,)
mu_apost = np.array([mu_apost]).reshape(N**2,)

df = pd.DataFrame({
    "Hematocrit, %":Hs,
    "Fibrinogen, mg/dL":cfs,
    "Yield Stress Machine Learning, mPa":ys,
    "Yield Stress Machine Learning STD, mPa":ys_std,
    "Casson Yield Stress Apostolidis, mPa":ys_apost})

df.to_excel("Surface Plot Data _ Yield Stress.xlsx",index=False)

df = pd.DataFrame({
    "Hematocrit, %":Hs,
    "Fibrinogen, mg/dL":cfs,
    "Casson Viscosity Machine Learning, mPa":mu,
    "Casson Viscosity Machine Learning STD, mPa":mu_std,
    "Casson Viscosity Apostolidis, mPa":mu_apost})

df.to_excel("Surface Plot Data _ Casson Viscosity.xlsx",index=False)
