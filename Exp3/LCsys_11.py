import numpy as np

def LCsys(t, d,omega,Xcen):

    dxdt = np.zeros ( 2 )

    dxdt[0] = d*np.cos(omega*t)+Xcen
    dxdt[1] = d*np.sin(omega*t)
    
    return dxdt
