"""
Author: Sean Farrington
Mon May  9 11:03:15 2022
"""

import numpy as np

#%% Apostolidis model calculations
class Apostolidis:
    def __init__(self,Hs,cfs,mu_std=0,sig_y_std=0):
        self.hematocrit = Hs
        self.fibrinogen = cfs
        self.viscnoise = mu_std
        self.yieldnoise = sig_y_std
        
    def apostFunc(self):
        To = 296.16   # Reference temperature, K
        n_p = 1.67e-2 # Plasma viscosity, dyne*s/cm**
        T = 310       # Temperature, K        
        H_ms = self.hematocrit # Hematocrit, fraction
        cf_ms = self.fibrinogen # fibrinogen, g/dL
        mu_std = self.viscnoise
        sig_y_std = self.yieldnoise
        
        sig_y       = []
        mu          = []
        mu_noise    = []
        sig_y_noise = []
        
        for j in range(len(H_ms)):
            H = H_ms[j]
            cf = cf_ms[j]
            # To choose critical hematocrit
            if cf<0.75:
                Hc = 0.3126*cf**2-0.468*cf+0.1764
            else:
                Hc = 0.0012
                
            # To choose yield stress
            if H>Hc:
                sig_calc = 100*(H-Hc)**2*(0.5084*cf+0.4517)**2
                # The above is converted from dyne.cm**-2 to mPa with a factor of 100
            else:
                sig_calc = 0
            
            sig_noi = sig_calc+sig_calc*np.random.normal(0,sig_y_std)
           
            # Apostolidis equation for Casson viscosity, expeirmentally determined    
            mu_calc = 100*n_p*(1+2.0703*H+3.7222*H**2)*np.exp(-7.0276*(1-To/T)) 
            # The above is Converted from dyne.s.cm**-2 to mPa.s with a factor of 100
            mu_noi = mu_calc + mu_calc*np.random.normal(0,mu_std)
            
            
            # Make all necessary lists
            sig_y.append(sig_calc)
            sig_y_noise.append(sig_noi)
            mu.append(mu_calc)
            mu_noise.append(mu_noi)
                
        return sig_y,mu,sig_y_noise,mu_noise
