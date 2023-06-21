"""
Author: Sean Farrington
Thu Jun  16

# Apostolidis Model Pseudo Data Machine Learning
 
# Gaussian Process Regression

# KFold testing procedure for overall model accuracy
# Inner training and validation done with leave one out and grid search

"""

#%% Importing necessary packages and functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import os
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct, RationalQuadratic

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
SEED = 1 
TRAINING_OPTION = 'full' # choose between full or partial
models = {
    'gpr':{
        'kernel':None,
        'n_restarts_optimizer':[25],
        'random_state':[SEED]
    }
        }

#%% Model instantiation
def set_model(name):
    """
    Initialization module for models to be evaluated
    """
    if name == 'gpr':
        model = GaussianProcessRegressor()
        
    return model

#%% Kernel Initialization
def restart_kernels(init_length_scale=1.0):
    """
    Function that calls kernels every time they need to be instanciated.
    """
    kernels = [1.0*RBF(length_scale=init_length_scale, \
                       length_scale_bounds=(10e-10,10e10))+1.0*WhiteKernel()]
    return kernels

#%% Choose training loop
if TRAINING_OPTION == 'full':
    best_models_names = ['gpr','gpr']

#%% Import Data From "Prepare_Data.py"
"""
What does our data look like ?

X [=] physiology inputs
X = [Hematocrit, Fibrinogen] [=] [%,mg/dL]

y [=] rheology outputs
y = [Casson yield stress, Casson viscosity] [=] [mPa, mPa.s]
"""

RAW_DATA = pd.read_excel('DATA/HornerData_CassonFitTable.xlsx')
DATA = pd.read_excel('DATA/HornerData_CassonFitTable.xlsx')

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

donors_out.to_excel("DATA/DonorOutliers.xlsx",index=False)


X_DATA = DATA[['Hematocrit, %','Fibrinogen, mg/dL']].to_numpy()
X_DATA[:,0] = X_DATA[:,0]/100
X_DATA[:,1] = X_DATA[:,1]/1000
y_DATA = DATA[['Casson yield stress, mPa','Casson viscosity, mPa.s']].to_numpy()

TARGETS = ['Casson Yield Stress','Casson Viscosity']
NR_FEATURES = ['Hematocrit','Fibrinogen']   


kf = KFold(n_splits=5)
kf.get_n_splits(X_DATA)
print(kf)

r2_1 = []
r2_2 = []
r2_3 = []
r2_4 = []

for train_index, test_index in kf.split(X_DATA):
    print("TRAIN:", train_index, "TEST:", test_index)
    X, X_test = X_DATA[train_index], X_DATA[test_index]
    y, y_test = y_DATA[train_index], y_DATA[test_index]


    test = {
            'X':X,
            'y':y,
            'X_test':X_test,
            'y_test':y_test
            }
    
    mu = X.mean(axis=0)
    sigma = X.mean(axis=0)
    
    #%% Cross-Validation Scheme
    best_trained_models = []
    for j,target in enumerate(TARGETS): 
        y_train = y[:,j]
        # type of regressor
        m_i = best_models_names[j]
        # define the model
        model = set_model(m_i)
        if m_i == 'gpr':
            models[m_i]['kernel'] = restart_kernels(np.ones(X.shape[1]))
            
        # Define search space
        space = models[m_i]
        # configure the cross-validation procedure
        # cv_inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_inner = LeaveOneOut()
        # Define search
        search = GridSearchCV(model, space, scoring='r2',cv=cv_inner, refit=True, n_jobs=-1)
        # Execute search
        result = search.fit(X, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        best_trained_models.append(best_model)
        
    
    #%% Choose best kernels for Gaussian Regression    
    kernel1 = best_trained_models[0].kernel
    kernel2 = best_trained_models[1].kernel
    
    #%% Gaussian Regression Fitting
    """
    One Gaussian process for each output feature
    
    Casson yield stress --> gp_yield
    
    Casson viscosity --> gp_visc
    """    
    
    gp_yield = GaussianProcessRegressor(kernel=kernel1)
    gp_yield.fit(X,y[:,0])
    
    gp_visc = GaussianProcessRegressor(kernel=kernel2)
    gp_visc.fit(X,y[:,1])

    
    result1 = gp_yield.score(X_test,y_test[:,0])
    result2 = gp_visc.score(X_test,y_test[:,1])
    # print(result1,result2)
    
    
    #%% Make predictions on Casson parameters
    yield_pred, std_yield = gp_yield.predict(X_test, return_std=True)
    yield_train_pred = gp_yield.predict(X) # Predicted training data
    
    visc_pred, std_visc = gp_visc.predict(X_test, return_std=True)
    visc_train_pred = gp_visc.predict(X) # Predicted training data
    
    y_pred = np.transpose(np.array([yield_pred,visc_pred]))
    y_train_pred = np.transpose(np.array([yield_train_pred,visc_train_pred]))
    
    #%% Some Data labels
    yield_pred = y_pred[:,0]
    visc_pred = y_pred[:,1]
    
    yield_train_pred = y_train_pred[:,0]
    visc_train_pred = y_train_pred[:,1]
    
    
    yield_test = y_test[:,0]
    visc_test = y_test[:,1]
    #%% Plotting predicted vs actual for the Cassson yield stress
    yield_train = y[:,0] # Actual training data
    
    
    
    # Plot predicted vs actual Casson yield stress
    fig, ax = plt.subplots(figsize=(5,4))
    p1, = ax.plot(yield_test,yield_pred,
            'o',
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
    
    # save_fig("Casson Yield Stress")
    plt.show()
    
    r2_yield_pred= r2_score(yield_test,yield_pred)
    r2_yield_train = r2_score(yield_train,yield_train_pred)
    
    #%% Plotting predicted vs actual Casson viscosity
    visc_train = y[:,1] # Actual Training data
    
    
    lims = [0.9*min(visc_test),1.1*max(visc_test)]
    
    fig, ax = plt.subplots(figsize=(5,4))
    p1, = ax.plot(visc_test,visc_pred,
                  'o',
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
    
    # save_fig("Casson Viscosity")
    plt.show()
    
    r2_mu_pred = r2_score(visc_test,visc_pred)
    r2_mu_train = r2_score(visc_train,visc_train_pred)
    
    
    df = pd.DataFrame({
        'R**2 Casson Yield Stress Training':[r2_yield_train],
        'R**2 Casson Viscosity Trainging':[r2_mu_train],
        'R**2 Casson Yield Stress Prediction':[r2_yield_pred],
        'R**2 Casson Viscosity Prediction':[r2_mu_pred]})
    df = df.transpose()
    
    r2_1.append(r2_yield_train)
    r2_2.append(r2_mu_train)
    r2_3.append(r2_yield_pred)
    r2_4.append(r2_mu_pred)
    
    print()
    print(df)
    print()
    
df = pd.DataFrame({
    'R**2 Casson Yield Stress Training':r2_1,
    'R**2 Casson Viscosity Training':r2_2,
    'R**2 Casson Yield Stress Prediction':r2_3,
    'R**2 Casson Viscosity Prediction':r2_4})

filename = 'DATA/KFold_RMSE_values.csv'

df.to_csv(filename,mode='w',index=False)   
