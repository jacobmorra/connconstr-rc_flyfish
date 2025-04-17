import numpy as np
import numpy.linalg as npl
from numpy.linalg import inv

def generate_Wout_Batch_Ridge_Regression(beta,Rtrainsq,Xtrain):
    Wout = ( Xtrain @ np.transpose(Rtrainsq) ) @ npl.inv( Rtrainsq @ np.transpose(Rtrainsq) + beta*np.identity( len(Rtrainsq) ) )
    return Wout
