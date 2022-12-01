"""
Code for the paper ....
Sean Farrington, Soham Jariwala, Matt Armstrong, Ethan Nigro, Antony Beris, and Norman Wagner

Author: Sean Farrington
October 20th, 2022
 
Gaussian Process Regression for Blood Rheology

The main purpose of this script is to make a connection between the physiology and rheology of blood

The physiology of blood is the hematocrit and fibrinogen. The rheolgoy of blood is the Casson
parameters of yield stress and viscosity.

The script is sectioned as follows:
    1. Importing necessary packages and functions
    2. Defining Gaussian process regression and seed
    3. Import data from "HornerData_CassonFitTable.xlsx"
    4. Cross-validation scheme
    5. Save the model using pickle

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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic

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

#%% Define machine learning models to use
SEED = 1743
nro = 5

#%% Import Data From "Prepare_Data.py"
"""
What does our data look like ?

X [=] physiology inputs
X = [Hematocrit, Fibrinogen] [=] [%,mg/dL]

y [=] rheology outputs
y = [Casson yield stress, Casson viscosity] [=] [mPa, mPa.s]
"""

RAW_DATA = pd.read_excel('HornerData_CassonFitInCassonCoordinates.xlsx')
DATA = pd.read_excel('HornerData_CassonFitInCassonCoordinates.xlsx')

donors_out = pd.DataFrame() 

# Finding Healthy Range Section
ind = DATA.index[(DATA['Hematocrit, %']<36)&(DATA['Sex']=='F')].tolist()
donors_out = donors_out.append(DATA.loc[ind])
DATA.drop(index=ind,inplace=True)

ind = DATA.index[(DATA['Hematocrit, %']>47)&(DATA['Sex']=='F')].tolist()
donors_out = donors_out.append(DATA.loc[ind])
DATA.drop(index=ind,inplace=True)

ind = DATA.index[(DATA['Hematocrit, %']<41)&(DATA['Sex']=='M')].tolist()
donors_out = donors_out.append(DATA.loc[ind])
DATA.drop(index=ind,inplace=True)

ind = DATA.index[(DATA['Hematocrit, %']>51)&(DATA['Sex']=='M')].tolist()
donors_out = donors_out.append(DATA.loc[ind])
DATA.drop(index=ind,inplace=True)

ind = DATA.index[DATA['Fibrinogen, mg/dL']<150].tolist()
donors_out = donors_out.append(DATA.loc[ind])
DATA.drop(index=ind,inplace=True)

ind = DATA.index[DATA['Fibrinogen, mg/dL']>350].tolist()
donors_out = donors_out.append(DATA.loc[ind])
DATA.drop(index=ind,inplace=True)

donors_out.to_excel("DonorOutliers.xlsx",index=False)

Donors = DATA['donors'].to_numpy()
X_DATA = DATA[['Hematocrit, %','Fibrinogen, mg/dL']].to_numpy()
X_DATA[:,0] = X_DATA[:,0]/100
X_DATA[:,1] = X_DATA[:,1]/1000
y_DATA = DATA[['Casson yield stress, mPa','Casson viscosity, mPa.s', \
               'Casson yield stress std','Casson viscosity std']].to_numpy()
# y_DATA_std = DATA[['Casson yield stress std','Casson viscosity std']].to_numpy()

TARGETS = ['Casson Yield Stress','Casson Viscosity']
NR_FEATURES = ['Hematocrit','Fibrinogen']   

y = y_DATA[:,0:2]
y_std = y_DATA[:,2:4]

X = X_DATA


#%% Choose best kernels for Gaussian Regression
kernel = 1.0*C() + 1.0*RBF(length_scale=np.ones(X.shape[1])) + 1.0*WhiteKernel()

#%% Gaussian Regression Fitting
"""
One Gaussian process for each output feature

Casson yield stress --> gp_yield

Casson viscosity --> gp_visc
"""    

gp_yield = GaussianProcessRegressor(kernel=kernel,
                                    n_restarts_optimizer=nro)
gp_yield.fit(X,y[:,0])
tuned_kernel1 = gp_yield.kernel_

gp_visc = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=nro)
gp_visc.fit(X,y[:,1])
tuned_kernel2 = gp_visc.kernel_

#%% Save the models using pickle

filename1 = 'finalized_model_yieldstress.sav'
filename2 = 'finalized_model_viscosity.sav'
filename3 = 'finalized_model_data.pkl'

pickle.dump(gp_yield,open(filename1, 'wb'))
pickle.dump(gp_visc,open(filename2,'wb'))
data = {
    "donors":Donors,
    "X":X,
    "y":y,
    "y_std":y_std}
pickle.dump(data,open(filename3,'wb'))


#%% Load the models using pickle
filename1 = 'finalized_model_yieldstress.sav'
filename2 = 'finalized_model_viscosity.sav'
filename3 = 'finalized_model_data.pkl'

gp_yield = pickle.load(open(filename1,'rb'))
gp_visc = pickle.load(open(filename2,'rb'))
data = pickle.load(open(filename3,'rb'))

X = data['X']
y = data['y']


#%% Make predictions on Casson parameters
yield_train_pred = gp_yield.predict(X) # Predicted training data
visc_train_pred = gp_visc.predict(X) # Predicted training data

y_train_pred = np.transpose(np.array([yield_train_pred,visc_train_pred]))

#%% Some Data labels
yield_train_pred = y_train_pred[:,0]
visc_train_pred = y_train_pred[:,1]

#%% Plotting predicted vs actual for the Cassson yield stress
yield_train = y[:,0] # Actual training data

# Plot predicted vs actual Casson yield stress
fig, ax = plt.subplots(figsize=(5,4))
p1, = ax.plot(yield_train,yield_train_pred,
        'o',
        markersize = 3,
        label = 'Training Data',
        color='blue')
lims = ax.get_xlim()

ax.plot(lims,lims,color='black')

ax.set_xlabel('Actual Casson Yield Stress')
ax.set_ylabel('Predicted Casson Yield Stress')

ax.set_xlim(lims)
ax.set_ylim(lims)

ax.legend(handles=[p1])

# save_fig("Casson Yield Stress")
plt.show()

r2_yield_train = r2_score(yield_train,yield_train_pred)

#%% Plotting predicted vs actual Casson viscosity
visc_train = y[:,1] # Actual Training data

fig, ax = plt.subplots(figsize=(5,4))
p1, = ax.plot(visc_train,visc_train_pred,
              'o',
              markersize = 3,
              label='Training Data',
              color='blue')

lims = ax.get_xlim()
ax.plot(lims,lims,color='black')

ax.set_xlabel('Actual Casson Viscosity')
ax.set_ylabel('Predicted Casson Viscosity')

ax.set_xlim(lims)
ax.set_ylim(lims)

ax.legend(handles=[p1])

# save_fig("Casson Viscosity")
plt.show()

r2_mu_train = r2_score(visc_train,visc_train_pred)


df = pd.DataFrame({
    'R**2 Casson Yield Stress Training':[r2_yield_train],
    'R**2 Casson Viscosity Training':[r2_mu_train]})
df = df.transpose()


print()
print(df)
print()
