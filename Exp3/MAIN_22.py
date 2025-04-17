import numpy as np
import numpy.linalg as npl
from numpy.linalg import inv
from numpy import random
import scipy as scipy
import scipy.sparse as sparse
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from scipy.sparse import identity
from scipy.sparse import vstack

from RK4 import rk4vec
from RK4_ext import rk4vec_ext
from LCsys_11 import LCsys
from RidgeRegression import generate_Wout_Batch_Ridge_Regression

#res_params = {'gama':10,'sigma':0.014}

def generate_M(N,dens,rho):
    '''
    
    Function to generate the internal connection matrix, M, with a sparse Erdos-Renyi topology.
    
    N: The number of neurons
    dens: The density of the matrix (probability of an element to be nonzero)
    rho: The spectral radius
    
    '''
    Minit = sparse.random(N, N, density=dens, data_rvs=lambda n: 2*np.random.random(n) - 1) #sparse matrix of size (N x N) and density, dens, with elements chosen uniformly random from (-1,1)       
    alpha = np.abs(sparse.linalg.eigs(Minit,k=1,which='LM',return_eigenvectors=False)) #returns the eigenvalue of largest magnitude
    M = (rho/abs(alpha[0]))*Minit #M is now a matrix with largest eigenvalue rho
    M = sparse.csr_matrix(M) #Converts to Compressed Sparse Row format to increase computational efficiency
    return M, Minit, alpha

def generate_Win(N,SysDim):
    '''
    Function to generate the input weight connection matrix, Win, each row has one nonzero element chosen randomly from [-1,1].
    
    N: The number of neurons
    SysDim: No. of variables considered for input
    
    '''
    Win = np.zeros((N,SysDim)) #Array of zero's of size (N,sizeof(u))
    for i in range(0,N):
        k = random.randint(0,SysDim) #pick a random number between 0 and sizeof(u)
        Win[i,k] = random.uniform(-1,1) #Each row now has a randomly chosen element whose value is unifromly random between -1 and 1  
    Win = sparse.csr_matrix(Win) #Converts to Compressed Sparse Row format to increase computational efficiency
    return Win

def Big_Sys(BigVec, t, u, gama, sigma, M, Win):
    '''
    
    Function which describes the dynamics of listening/training RC.
    
    BigVec: Combined state at a given time t
    t: Time

    gama: Damping coefficient
    sigma: Input strength parameter
    M: Internal Connection Matrix
    Win: Input Weight Connection Matrix
    
    '''
    r = BigVec
    
    dBigdt = np.zeros(len(BigVec))
    
    dBigdt = gama*( (-1)*r + np.tanh( M @ r + sigma*( Win @ u ) ) )
    return dBigdt


def Big_listen_stage(ListenEndTime, t_time, dt, BigVec, u, d, omega, Xcen, gama, sigma, M, Win):
    '''
    
    Function simulates the listening stage of the RC.
    
    ListenEndTime: Time up to which listening is completed
    t_time: List which keeps all times of evaluation
    dt: Time step
    BigVec: State of the listening/training RC
    gama: Damping coefficient
    sigma: Input strength parameter
    M: Internal Connection Matrix
    Win: Input Weight Connection Matrix
    xy: Input data
    d: circle radius
    omega: angular velocity
    Xcen: xcen location
    '''
    for j in range(0, ListenEndTime):
        args = [u, gama, sigma, M, Win]
        BigVec = rk4vec_ext(BigVec, t_time[j], dt, Big_Sys,args)
        xy=[d*np.cos(omega*t_time[j])+Xcen,d*np.sin(omega*t_time[j])]
        u=np.array(xy)
    return BigVec, u

    
def Big_train_stage(ListenEndTime,TrainEndTime, t_time, dt, BigVec, u, d, omega, Xcen, gama, sigma, M, Win):
    '''
    
    Function which simulates the training stage of the listening/training RC.
    
    ListenEndTime: Time from which training stage begins
    TrainEndTime: Time up to which training stage is completed
    t_time: List which keeps all times of evaluation
    dt: Time step
    BigVec: State of the combined Lorenz and listening/training reservoir computer
    gama: Damping coefficient
    sigma: Input strength parameter
    M: Internal Connection Matrix
    Win: Input Weight Connection Matrix
    xy: Input data
    d: circle radius
    omega: angular velocity
    Xcen: xcen location
    Xtrain: stores the training data from the driving system
    Rtrain: stores the RCs response to the driving system
    Rtrainsq: stores the (r r^2)^T of the RCs response to the driving system
    '''
    Xtrain = []
    Rtrain = []
    Rtrainsq = []
    for j in range(ListenEndTime,TrainEndTime):
        args = [u, gama, sigma, M, Win]
        BigVec = rk4vec_ext(BigVec,t_time[j], dt, Big_Sys,args)
        xy=[d*np.cos(omega*t_time[j])+Xcen,d*np.sin(omega*t_time[j])]
        u=np.array(xy)
        Xtrain.append(u)
        r = BigVec
        Rtrain.append(r)
        rsq = np.square(r)
        r_con = np.concatenate((r,rsq),axis=None)
        Rtrainsq.append(r_con)
    return Xtrain, Rtrain, Rtrainsq, u, r


def predict_res(r,t,gama,sigma,M,Win,Wout):
    '''
    
    Function which describes the dynamics of predicting RC.
    
    r: Reservoir state at a given time t
    t: Time
    gama: Damping coefficient
    sigma: Input strength parameter
    M: Internal Connection Matrix
    Win: Input Weight Connection Matrix
    Wout: Output Weight Connection Matrix
    
    '''    
    rsq = np.square(r)
    q = np.concatenate((r,rsq),axis=None)
    drdt = gama*( -r + np.tanh( M @ r + sigma*( Win @ (Wout @ q) ) ) )
    return drdt

def predict_res_no_sq(r,t,gama,sigma,M,Win,Wout):
    '''
    
    Function which represents the dynamics of predicting reservoir computer.
    
    r: Reservoir state at a given time t
    t: Time
    gama: Damping coefficient
    sigma: Input strength parameter
    M: Adjacency/Internal Connection Matrix
    Win: Input Weight Connection Matrix
    Wout: Output Weight Connection Matrix
    
    '''    
    
    drdt = gama*( -r + np.tanh( M @ r + sigma*( Win @ (Wout @ r) ) ) )
    return drdt

def predict_stage(xy,r,Wout,TrainEndTime,PredictEndTime,t_time,dt, d, omega, Xcen,gama,sigma,M,Win):
    '''
    
    Function which simulates the predicting stage of the RC.
    
    xy: circle state at a given time t
    r: Reservoir state at a given time t
    Wout: Output Weight Connection Matrix
    TrainEndTime: Time from which predictinging stage begins
    PredictEndTime: Time up to which predicting stage is completed
    t_time: List which keeps all times of evaluation
    dt: Time step
    gama: Damping coefficient
    sigma: Input strength parameter
    M: Internal Connection Matrix
    Win: Input Weight Connection Matrix
    xy: Input data
    d: circle radius
    omega: angular velocity
    Xcen: xcen location
    Xpredict: ground truth data
    Rpredictsq: RCs prediction (with sq funciton)
    
    '''
    Xpredict = []
    Rpredictsq = []
    for j in range(TrainEndTime, PredictEndTime):
        args = [gama,sigma,M,Win,Wout]

        r = rk4vec_ext(r, t_time[j], dt, predict_res, args)
        
        #Rpredict.append(r)
        rsq = np.square(r)
        r_con = np.concatenate((r,rsq),axis=None)
        Rpredictsq.append(r_con)
        
        
        xy = [d*np.cos(omega*t_time[j])+Xcen,d*np.sin(omega*t_time[j])]
        Xpredict.append(xy)
    return np.array(Xpredict).T, np.array(Rpredictsq).T

def predict_stage_no_sq(xy,r,Wout,TrainEndTime,PredictEndTime,t_time,dt,d, omega, Xcen,gama,sigma,M,Win):
    '''
    
    Function which simulates the predicting stage of the Lorenz system.
    
    xy: circle state at a given time t
    r: Reservoir state at a given time t
    Wout: Output Weight Connection Matrix
    TrainEndTime: Time from which predictinging stage begins
    PredictEndTime: Time up to which predicting stage is completed
    t_time: List which keeps all times of evaluation
    dt: Time step
    gama: Damping coefficient
    sigma: Input strength parameter
    M: Internal Connection Matrix
    Win: Input Weight Connection Matrix
    xy: Input data
    d: circle radius
    omega: angular velocity
    Xcen: xcen location
    Xpredict: ground truth data
    Rpredict: RCs prediction (without sq funciton)
    
    '''
    Xpredict = []
    Rpredict = []
    for j in range(TrainEndTime, PredictEndTime):
        args = [gama,sigma,M,Win,Wout]

        r = rk4vec_ext(r, t_time[j], dt, predict_res_no_sq, args)
        
        #Rpredict.append(r)
        
        Rpredict.append(r)
        
        xy = [d*np.cos(omega*t_time[j])+Xcen,d*np.sin(omega*t_time[j])]
        Xpredict.append(xy)
    return np.array(Xpredict).T, np.array(Rpredict).T



def generate_NetOut(Wout,Rpredictsq):
    '''
    
    Function which generate the networks prediction of the data
    
    Wout: Output Weight Connection Matrix
    Rpredictsq: The data needed to be converted to the correct format for prediction results
    
    '''
    NetOut = Wout @ Rpredictsq
    return NetOut  

def Qres_rhs(Qres, t, N,gama,sigma,M,Win,Wout,Rtrain,Rtrainsq):
    '''
    
    Function which represents the evolution of the Floquet matrix for the reservoir computer
    
    Qres: State of the Floquet Matrix at time t
    t: Time
    N: Number of neurons
    gama: Damping coefficient
    sigma: Input strength parameter
    M: Adjacency/Internal Connection Matrix
    Win: Input Weight Connection Matrix
    Wout: Output Weight Connection Matrix
    Rtrain: List of data containing information about the state of the reservoir computer during the training stage
    Rtrainsq: List of data containing information about the state of the q's during the training stage
    '''
    ID = identity(N)
    NetJacobian = gama*( (-1)*ID + diags( 1/np.cosh( M @ Rtrain + sigma*Win @ (Wout @ Rtrainsq) )**2 ) @ M + 
                           diags( 1/np.cosh( M @ Rtrain + sigma*Win @ (Wout @ Rtrainsq) )**2) @ ( sigma*Win @ ( Wout @ vstack([ID,2*diags(Rtrain)]) ) ) )
    dQresdt = NetJacobian @ Qres
    return dQresdt

def Get_FloquetMatrix(Qres, t_time, TripStartTime, TripEndTime, dt, N,gama,sigma,M,Win,Wout,Rtrain,Rtrainsq):
    '''
    
    Function which generates the Floquet Matrix after one round trip for the reservoir computer
    
    Qres: State of the Floquet Matrix at time t_time
    t_time: List which keeps all times of evaluation
    TripStartTime: Start time of one round trip
    TripEndTime: End time of one round trip
    N: Number of neurons
    gama: Damping coefficient
    sigma: Input strength parameter
    M: Adjacency/Internal Connection Matrix
    Win: Input Weight Connection Matrix
    Wout: Output Weight Connection Matrix
    Rtrain: List of data containing information about the state of the reservoir computer during the training stage
    Rtrainsq: List of data containing information about the state of the q's during the training stage
    
    '''
    for i in range(TripStartTime,TripEndTime):
        args=[N,gama,sigma,M,Win,Wout,Rtrain[i],Rtrainsq[i]]
        Qres = rk4vec_ext(Qres,t_time[i],dt,Qres_rhs,args)
        #if i%100 == 0:
        #    print(i)
    return Qres

def Lyap_calc_sys(r_del_vec,t,gama,sigma,M,Win,Wout,N):
    '''
    Function which returns the RHS of the combined RC and Jacobian system
    
    r_del_vec: [:N] elements is state of RC, [N:] is state of Jacobian system
    t: time
    gama,sigma,M,Win,Wout,N: RC parameters
    '''
    r=r_del_vec[0:N]
    delta=r_del_vec[N:,]
    dQdt = np.zeros(len(r_del_vec))
    ID = identity(N)
    rsq = np.square(r)
    q = np.concatenate((r,rsq),axis=None)
    Aa=diags( 1/(np.cosh( M @ r + sigma*Win @ (Wout @ q) )**2) )
    B= sigma*( Win @ ( Wout @ vstack([ID,2*diags(r)]) ) )
    J=-gama*ID + gama*Aa @ ( M + B )
    
    dQdt[0:N] = gama*( -r + np.tanh( M @ r + sigma*( Win @ (Wout @ q) ) ) )
    dQdt[N:,] = J @ delta
    
    return dQdt

def compute_LLE(TimeEvol1,TimeEvol2,dt,r_del_vec,gama,sigma,M,Win,Wout,N,lyap_iters):
    '''
    Function which returns the largest Lyapunov exponent (LLE), and it's convergence in an array, of the RC predicted trajectory on a given attractor trained for a particular Wout
    
    TimeEvol1: Time to let RC and perturbation vector settle on attractor
    TimeEvol2: Time to measure growth of perturbation vector on attractor
    dt: Integration time-step
    gama,sigma,M,Win,Wout,N: RC parameters
    lyap_iters: Number of times perturbation vector is let evolve by time TimeEvol2, chosen sufficiently large to allow LLE to converge
    r_del_vec: np.concatenate((Rpredictsq[-1][0:N],delta_vec_IC)) #delta_vec_IC=np.random.uniform(-0.3, 0.3, N)
    '''
    Time1=int(TimeEvol1/dt)
    Time2=int(TimeEvol2/dt)
    TimeEnd=Time1+Time2
    TotalTime=TimeEvol1+TimeEvol2
    Lyap_t_time = np.linspace(0.0,TotalTime,int(TotalTime/dt))
    for j in range(0, Time1): #Let Lyap_calc_sys system settle towards an attractor
        args = [gama,sigma,M,Win,Wout,N]
        r_del_vec = rk4vec_ext(r_del_vec, Lyap_t_time[j], dt, Lyap_calc_sys, args)
    delta_mag=np.sqrt(sum(np.square(r_del_vec[N:,]))) #copmpute magnitude of perturbation vector
    r_del_vec[N:,]=r_del_vec[N:,]/delta_mag #normalise perturbation vector
    h=0
    h_list=[]
    for g in range(0,lyap_iters): #lyap_iters: convergence of LLE
        for j in range(Time1,TimeEnd): #Step forward in time to determine growth of perturbation vector
            args = [gama,sigma,M,Win,Wout,N]
            r_del_vec = rk4vec_ext(r_del_vec, Lyap_t_time[j], dt, Lyap_calc_sys, args)
        delta_mag=np.sqrt(sum(np.square(r_del_vec[N:,])))
        a=(1/TimeEvol2)*np.log(delta_mag)
        h+=a
        h_list.append( h/(g+1) ) #store convergence towards LLE
        #print('h',h/(g+1))
        r_del_vec[N:,]=r_del_vec[N:,]/delta_mag
    LLE=h/(g+1)
            
    return LLE,np.array(h_list)

##Function to return a RC's predicted trajectory on two cycles
def Generate_predicitons(rho,xcen,alpha,dt,t_time,ListenEndTime,TrainEndTime,PredictEndTime,M,Win,largest_evalue,N,dd1,omega1,dd2,omega2,gama,sigma,beta):
    '''
    Function which generates the RC's predicted trajectory on both cicrles and returns the trained Wout matrix
    '''
    
    Xcen1=xcen
    Xcen2=-Xcen1
    ycen=0
    
    r_1 = np.zeros(N)
    BigVec_1 = r_1
    #Initial input
    u1=np.array([dd1*np.cos(omega1*t_time[0])+Xcen1,dd1*np.sin(omega1*t_time[0])])
    BigVec_1,u1 = Big_listen_stage(ListenEndTime,t_time,dt,BigVec_1, u1, dd1, omega1, Xcen1, gama, sigma, M, Win)
    #Training stage _ Circ1
    Xtrain_1, Rtrain_1, Rtrainsq_1, xy_1, r_1 = Big_train_stage(ListenEndTime,TrainEndTime, t_time, dt, BigVec_1, u1, dd1, omega1, Xcen1, gama, sigma, M, Win)
    #Wout with just Circ1 data
    #Wout_Batch_1 = generate_Wout_Batch_Ridge_Regression(beta,np.array(Rtrainsq_1).T,np.array(Xtrain_1).T)
    
    #Listening stage _ Circ2
    #IC for reservoir state
    r_2 = np.zeros(N)
    BigVec_2 = r_2
    #Initial input
    u2=np.array([dd2*np.cos(omega2*t_time[0])+Xcen2,dd1*np.sin(omega2*t_time[0])])
    BigVec_2,u2 = Big_listen_stage(ListenEndTime,t_time,dt,BigVec_2, u2, dd2, omega2, Xcen2, gama, sigma, M, Win)
    #Training stage _ Circ2
    Xtrain_2, Rtrain_2, Rtrainsq_2, xy_2, r_2 = Big_train_stage(ListenEndTime,TrainEndTime, t_time, dt, BigVec_2, u2, dd2, omega2, Xcen2, gama, sigma, M, Win)

    Wout1 = generate_Wout_Batch_Ridge_Regression(beta,np.array(Rtrainsq_1).T,np.array(Xtrain_1).T)
    Wout2 = generate_Wout_Batch_Ridge_Regression(beta,np.array(Rtrainsq_2).T,np.array(Xtrain_2).T)
    
    #Predicting Stage
    
    X_true_1_MF, Rpredictsq_1_MF = predict_stage(xy_1,r_1,Wout1,TrainEndTime,PredictEndTime,t_time,dt,dd1, omega1, Xcen1,gama,sigma,M,Win)
    X_true_2_MF, Rpredictsq_2_MF = predict_stage(xy_2,r_2,Wout2,TrainEndTime,PredictEndTime,t_time,dt,dd2,omega2, Xcen2,gama,sigma,M,Win)
    
    NetOut_1_MF = generate_NetOut(Wout1,Rpredictsq_1_MF)
    NetOut_2_MF = generate_NetOut(Wout2,Rpredictsq_2_MF)
    
    return X_true_1_MF,X_true_2_MF,Rpredictsq_1_MF,Rpredictsq_2_MF, xy_1, xy_2, r_1, r_2, Wout1,Wout2, NetOut_1_MF,NetOut_2_MF

##Function to return a MF RC's predicted trajectory on two cycles
def Generate_MF_predicitons(rho,xcen,alpha,dt,t_time,ListenEndTime,TrainEndTime,PredictEndTime,M,Win,largest_evalue,N,dd1,omega1,dd2,omega2,gama,sigma,beta):
    '''
    Function which generates the MF RC's predicted trajectory on both cicrles and returns the trained MF Wout matrix
    '''
    
    Xcen1=xcen
    Xcen2=-Xcen1
    ycen=0
    
    r_1 = np.zeros(N)
    BigVec_1 = r_1
    #Initial input
    u1=np.array([dd1*np.cos(omega1*t_time[0])+Xcen1,dd1*np.sin(omega1*t_time[0])])
    BigVec_1,u1 = Big_listen_stage(ListenEndTime,t_time,dt,BigVec_1, u1, dd1, omega1, Xcen1, gama, sigma, M, Win)
    #Training stage _ Circ1
    Xtrain_1, Rtrain_1, Rtrainsq_1, xy_1, r_1 = Big_train_stage(ListenEndTime,TrainEndTime, t_time, dt, BigVec_1, u1, dd1, omega1, Xcen1, gama, sigma, M, Win)
    #Wout with just Circ1 data
    #Wout_Batch_1 = generate_Wout_Batch_Ridge_Regression(beta,np.array(Rtrainsq_1).T,np.array(Xtrain_1).T)
    
    #Listening stage _ Circ2
    #IC for reservoir state
    r_2 = np.zeros(N)
    BigVec_2 = r_2
    #Initial input
    u2=np.array([dd2*np.cos(omega2*t_time[0])+Xcen2,dd1*np.sin(omega2*t_time[0])])
    BigVec_2,u2 = Big_listen_stage(ListenEndTime,t_time,dt,BigVec_2, u2, dd2, omega2, Xcen2, gama, sigma, M, Win)
    #Training stage _ Circ2
    Xtrain_2, Rtrain_2, Rtrainsq_2, xy_2, r_2 = Big_train_stage(ListenEndTime,TrainEndTime, t_time, dt, BigVec_2, u2, dd2, omega2, Xcen2, gama, sigma, M, Win)
        
    alpha = np.round(alpha,6)
    #Make Wout_alpha
    Att1Att2_Stack = np.vstack(([i*alpha for i in Xtrain_1],[i*(1-alpha) for i in Xtrain_2]))
    Att1Att2_ResStack = np.vstack(([i*alpha for i in Rtrainsq_1],[i*(1-alpha) for i in Rtrainsq_2]))
        
    CombineData = list(zip(Att1Att2_Stack,Att1Att2_ResStack))
    np.random.shuffle(CombineData)
    Att1Att2_RandStack,Att1Att2_RandResStack = zip(*CombineData)
    Att1Att2_RandStack = np.array(Att1Att2_RandStack)
    Att1Att2_RandResStack = np.array(Att1Att2_RandResStack)
    Wout_alpha = generate_Wout_Batch_Ridge_Regression(beta,np.array(Att1Att2_RandResStack).T,np.array(Att1Att2_RandStack).T)
    
    #Predicting Stage
    
    X_true_1_MF, Rpredictsq_1_MF = predict_stage(xy_1,r_1,Wout_alpha,TrainEndTime,PredictEndTime,t_time,dt,dd1, omega1, Xcen1,gama,sigma,M,Win)
    X_true_2_MF, Rpredictsq_2_MF = predict_stage(xy_2,r_2,Wout_alpha,TrainEndTime,PredictEndTime,t_time,dt,dd2,omega2, Xcen2,gama,sigma,M,Win)
    
    NetOut_1_MF = generate_NetOut(Wout_alpha,Rpredictsq_1_MF)
    NetOut_2_MF = generate_NetOut(Wout_alpha,Rpredictsq_2_MF)
    
    return X_true_1_MF,X_true_2_MF,Rpredictsq_1_MF,Rpredictsq_2_MF, xy_1, xy_2, r_1, r_2, Wout_alpha, NetOut_1_MF,NetOut_2_MF

def Generate_training_data(rho,xcen,dt,t_time,ListenEndTime,TrainEndTime,M,Win,N,dd1,omega1,dd2,omega2,gama,sigma):
    '''
    Function which generates only the training data for an RC driven with input from both cicrles
    '''
    
    Xcen1=xcen
    Xcen2=-Xcen1
    ycen=0
    
    r_1 = np.zeros(N)
    BigVec_1 = r_1
    #Initial input
    u1=np.array([dd1*np.cos(omega1*t_time[0])+Xcen1,dd1*np.sin(omega1*t_time[0])])
    BigVec_1,u1 = Big_listen_stage(ListenEndTime,t_time,dt,BigVec_1, u1, dd1, omega1, Xcen1, gama, sigma, M, Win)
    #Training stage _ Circ1
    Xtrain_1, Rtrain_1, Rtrainsq_1, xy_1, r_1 = Big_train_stage(ListenEndTime,TrainEndTime, t_time, dt, BigVec_1, u1, dd1, omega1, Xcen1, gama, sigma, M, Win)
    #Wout with just Circ1 data
    #Wout_Batch_1 = generate_Wout_Batch_Ridge_Regression(beta,np.array(Rtrainsq_1).T,np.array(Xtrain_1).T)
    
    #Listening stage _ Circ2
    #IC for reservoir state
    r_2 = np.zeros(N)
    BigVec_2 = r_2
    #Initial input
    u2=np.array([dd2*np.cos(omega2*t_time[0])+Xcen2,dd1*np.sin(omega2*t_time[0])])
    BigVec_2,u2 = Big_listen_stage(ListenEndTime,t_time,dt,BigVec_2, u2, dd2, omega2, Xcen2, gama, sigma, M, Win)
    #Training stage _ Circ2
    Xtrain_2, Rtrain_2, Rtrainsq_2, xy_2, r_2 = Big_train_stage(ListenEndTime,TrainEndTime, t_time, dt, BigVec_2, u2, dd2, omega2, Xcen2, gama, sigma, M, Win)
    return Xtrain_1, Rtrain_1, Rtrainsq_1,Xtrain_2, Rtrain_2, Rtrainsq_2

def BlendingTechnique(alpha,Xtrain_1,Xtrain_2,Rtrainsq_1,Rtrainsq_2,beta):
    alpha = np.round(alpha,6)
    #Make Wout_alpha
    Att1Att2_Stack = np.vstack(([i*alpha for i in Xtrain_1],[i*(1-alpha) for i in Xtrain_2]))
    Att1Att2_ResStack = np.vstack(([i*alpha for i in Rtrainsq_1],[i*(1-alpha) for i in Rtrainsq_2]))
        
    CombineData = list(zip(Att1Att2_Stack,Att1Att2_ResStack))
    np.random.shuffle(CombineData)
    Att1Att2_RandStack,Att1Att2_RandResStack = zip(*CombineData)
    Att1Att2_RandStack = np.array(Att1Att2_RandStack)
    Att1Att2_RandResStack = np.array(Att1Att2_RandResStack)
    Wout_alpha = generate_Wout_Batch_Ridge_Regression(beta,np.array(Att1Att2_RandResStack).T,np.array(Att1Att2_RandStack).T)
    return Wout_alpha

def distance_between_vectors(r1,r2,norm):
    '''
    Function to compute the norm (Euclidean/infinity) between the states of two vectors at a given time
    
    r1: States of vector r1
    r2: States of vector r2
    norm: 0: Euclidean, 1: Infinity
    '''
    if norm == 0:
        distance=np.sqrt(np.sum((r1-r2)**2,axis=1))
    if norm == 1:
        distance=np.sum((r1-r2)**(int(len(r1.T))),axis=1)**(1/(int(len(r1.T))))
    return distance

def distance_of_each_point(r1,r2):
    '''
    Function to compute the Euclidean norm between all the states of one vector with respect to each state of another vector at a given time and vice-versa
    
    r1: States of vector r1
    r2: States of vector r2
    '''
    each_distance12=[]
    min_each_distance12=[]
    for k in range(len(r2)):
        print(k)
        dist=np.sqrt(np.sum((r1-r2[k])**2,axis=1))
        min_dist=min(dist)
        each_distance12.append(dist)
        min_each_distance12.append(min_dist)
        print(min_dist)
    each_distance21=[]
    min_each_distance21=[]
    for k in range(len(r1)):
        print(k)
        dist=np.sqrt(np.sum((r2-r1[k])**2,axis=1))
        min_dist=min(dist)
        each_distance21.append(dist)
        min_each_distance21.append(min_dist)
        print(min_dist)
    return(np.array(each_distance12),np.array(each_distance21),np.array(min_each_distance12),np.array(min_each_distance21))
