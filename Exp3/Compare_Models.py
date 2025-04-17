# This code was originally created by Andrew Flynn, UCC, Cork, Ireland.
# The code has been later adapted by Jacob Morra, London, ON, Canada.

import numpy as np
import pandas as pd
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
from multiprocessing import Pool
from functools import partial
from time import sleep
from tqdm import tqdm
from fxns.MAIN_22 import generate_M,generate_Win,Big_listen_stage,Big_train_stage,predict_stage,generate_NetOut
from fxns.MAIN_22 import Generate_predicitons,Generate_MF_predicitons
from fxns.LCsys_11 import LCsys
from fxns.Circle_error_tools import Error_analysis_of_Pred_Circle,test_Error_analysis_of_Pred_Circle
from fxns.Circle_error_tools import check_errmaxminCA,check_errmaxminCB, fix_length_of_maxmins_with_nans, GetErrorBoth
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import rc
from datetime import date, datetime
from fxns.PlotTools import make_hist, make_hist_ev
from fxns.MAIN_22 import generate_M_from_raw_data, generate_M_from_degree_distn, generate_M_from_weight_distn

def getcount(max_err):
    count = 1
    for i in range(len(max_err)):
        if max_err[i] == 1.0:
            count += 1
    return count

def countgoodbad(model_class, model_data, n_loops=100,gama=5,rho=1.4,xcen=10.0, N=500.):
    """
    :param n_loops: number of tries of seeing double.. try 100
    :param gama: (includes leaking rate).. try between 5 and 100
    :param rho: spectral radius.. try between 0.5 and 2.5
    :param xcen: x-center coords of the two circles.. try 1, 5, 6, 7, 8, 9, 10
    :return: the number of times /100 that you get mfxnality
    """
    #Time constants for integration
    dt = 0.01 #time step
    Tlisten = 37.7 #Listening Time 6T
    ListenEndTime = int(Tlisten/dt) #Discretised Listen Time
    Ttrain = 94.25 + Tlisten #Training Time 15T
    TrainEndTime = int(Ttrain/dt) #Discretised Train Time
    Tpredict = 94.25 + Ttrain #Predicting Time 15T
    PredictEndTime = int(Tpredict/dt) #Discretised Predict Time
    t_time = np.linspace(0.0,Tpredict,int(Tpredict/dt)) #(Total Time)
    SysDim = 2 #Used in initialising Win and regression
    #Res params
    sigma = 0.2 # input strength
    beta = 1e-2 #Regularization Parameter
    alpha = 0.5 #Blending parameter
    #Input Data Params
    dd1=5
    dd2=-5
    omega1=1
    omega2=-1
    predtime=PredictEndTime-TrainEndTime
    FP_err_lim=1e-3
    sample_start=predtime-5000#+10000
    sample_end=predtime-1000#+10000
    stepback=20
    randrange=10
    pts1=250
    pts2=200
    pts3=150
    pts4=100
    FP_sample_start=predtime-1000#+10000
    FP_sample_end=predtime#+10000
    iter_no=1000
    LC_err_tol=0.01
    LC_err_tol_v3=0.00001
    rounding_no=2
    rho_max=2.5#2.5
    rho_min=0.5#0.8#0.2
    rho_steps=21#25
    rhospace=np.array([1.4])#np.linspace(rho_min,rho_max,rho_steps)
    N_max = 2000.
    N_min = 10.#300.#50.0
    N_steps = 2#10
    N_vals = np.array([500.])#np.array([10.,50.,100.,500.,1000.,1500., 2000.])#
    d = 0.05 #prob. of an element of M to be nonzero
    Mat_Data = []
    # print('rho', '__', 'N', '__', 'err_C1', '__', 'err_C2', '__', 'C1 rel rnd', '__', 'C2 rel rnd', '__', 'C1filt', '__',
    #       'C2filt')

    # MOST IMPORTANT PARAMS -----------------------------------------------------------------------------------------------
    n_loops=n_loops
    xcen = xcen
    gama = gama  # damping coefficient

    # ---------------------------------------------------------------------------------------------------------------------

    Xcen1 = xcen
    Xcen2 = -Xcen1
    ycen = 0.0

    for nl in tqdm(range(n_loops)):
        sleep(0.01)
        N_i = N
        N = int(N_i)
        rho = rho
        rho = np.round(rho, 4)

        if model_class == 'random':
            M, Minit, largest_evalue = generate_M(N, d, rho)
        elif model_class == 'struct':
            M, Minit, largest_evalue = generate_M_from_raw_data(rho, model_data)
        elif model_class == 'degdistn':
            M, Minit, largest_evalue = generate_M_from_degree_distn(N, rho, model_data)
        elif model_class == 'weightdistn':
            M, Minit, largest_evalue = generate_M_from_weight_distn(N, d, rho, model_data)

        Win = generate_Win(N, SysDim)

        ##Multifunctional case
        Xpredict_1_MF, Xpredict_2_MF, Rpredictsq_1_MF, Rpredictsq_2_MF, xy_1, xy_2, r_1, r_2, Wout_alpha, NetOut_1_MF, NetOut_2_MF = Generate_MF_predicitons(
            rho, xcen, alpha, dt, t_time, ListenEndTime, TrainEndTime, PredictEndTime, M, Win, largest_evalue, N, dd1,
            omega1, dd2, omega2, gama, sigma, beta)

        xp1_C1, xp2_C1 = Xpredict_1_MF  # Actual output
        xpredict1_C1_MF, xpredict2_C1_MF = NetOut_1_MF  # RC prediction

        xp1_C2, xp2_C2 = Xpredict_2_MF  # Actual output
        xpredict1_C2_MF, xpredict2_C2_MF = NetOut_2_MF  # RC prediction

        ##Error Analysis
        # C1
        err_C1, C1_vel_dir_strict, C1_roundness, C1_Rad_perr, C1_xcenter_err, C1_ycenter_err, x_C1_no_of_unique_maxima, C1_periodic_prof, xmax_localmaxima_C1, xmin_localmaxima_C1, xmax_localminima_C1, xmin_localminima_C1, ymax_localmaxima_C1, ymin_localmaxima_C1, ymax_localminima_C1, ymin_localminima_C1 = test_Error_analysis_of_Pred_Circle(
            xpredict1_C1_MF, xpredict2_C1_MF, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol, rounding_no,
            sample_start, sample_end, stepback, dd1, Xcen1, ycen, iter_no)
        C1rel_roundness = C1_roundness / dd1
        err_C1filt = check_errmaxminCA(err_C1, Xcen1, xmax_localmaxima_C1, ymax_localmaxima_C1, xmax_localminima_C1,
                                       ymax_localminima_C1, xmin_localmaxima_C1, ymin_localmaxima_C1, xmin_localminima_C1,
                                       ymin_localminima_C1)
        # C2
        err_C2, C2_vel_dir_strict, C2_roundness, C2_Rad_perr, C2_xcenter_err, C2_ycenter_err, x_C2_no_of_unique_maxima, C2_periodic_prof, xmax_localmaxima_C2, xmin_localmaxima_C2, xmax_localminima_C2, xmin_localminima_C2, ymax_localmaxima_C2, ymin_localmaxima_C2, ymax_localminima_C2, ymin_localminima_C2 = test_Error_analysis_of_Pred_Circle(
            xpredict1_C2_MF, xpredict2_C2_MF, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol, rounding_no,
            sample_start, sample_end, stepback, dd1, Xcen2, ycen, iter_no)
        C2rel_roundness = C2_roundness / dd1
        err_C2filt = check_errmaxminCB(err_C2, Xcen2, xmax_localmaxima_C2, ymax_localmaxima_C2, xmax_localminima_C2,
                                       ymax_localminima_C2, xmin_localmaxima_C2, ymin_localmaxima_C2, xmin_localminima_C2,
                                       ymin_localminima_C2)

        Mat_save = [rho, N_i, dd1, err_C1, err_C1filt, C1rel_roundness,
                    err_C2, err_C2filt, C2rel_roundness,
                    M.toarray(), Minit.toarray(), largest_evalue, Win.toarray(), Wout_alpha]
        Mat_Data.append(Mat_save)

        # print([rho, N_i, err_C1, err_C2, C1rel_roundness, C2rel_roundness, err_C1filt, err_C2filt])

    ErrorData = np.array(Mat_Data, dtype=object)[:, :9]
    MData = np.array(Mat_Data, dtype=object)[:, 9:10]
    MData = np.reshape(np.stack(MData.tolist(), axis=0), (int(N_i * n_loops), int(N_i)))
    MinitData = np.array(Mat_Data, dtype=object)[:, 10:11]
    MinitData = np.reshape(np.stack(MinitData.tolist(), axis=0), (int(N_i * n_loops), int(N_i)))
    largest_evalue_Data = np.array(Mat_Data, dtype=object)[:, 11:12]
    largest_evalue_Data = np.reshape(np.stack(largest_evalue_Data.real.tolist(), axis=0), (int(1 * n_loops), int(1)))
    WinData = np.array(Mat_Data, dtype=object)[:, 12:13]
    WinData = np.reshape(np.stack(WinData.tolist(), axis=0), (int(N_i * n_loops), int(SysDim)))
    WoutData = np.array(Mat_Data, dtype=object)[:, 13:14]
    WoutData = np.reshape(np.stack(WoutData.tolist(), axis=0), (int(SysDim * n_loops), int(2 * N_i)))
    rhos=ErrorData.reshape(9,n_loops).T[:,0]
    Ns=ErrorData.reshape(9,n_loops).T[:,1]
    dd1=ErrorData.reshape(9,n_loops).T[:,2]
    err_C1=ErrorData.reshape(9,n_loops).T[:,3]
    err_C1filt=ErrorData.reshape(9,n_loops).T[:,4]
    C1rel_roundness=ErrorData.reshape(9,n_loops).T[:,5]
    err_C2=ErrorData.reshape(9,n_loops).T[:,6]
    err_C2filt=ErrorData.reshape(9,n_loops).T[:,7]
    C2rel_roundness=ErrorData.reshape(9,n_loops).T[:,8]

    #Error criteria
    LC_error_bound = 0.1
    Aper_error_bound = 0.15
    center_upper_error_bound = 0.03

    err_vals_both1,err_both1,maxerr1,maxerr_vals1=GetErrorBoth(C1rel_roundness,C2rel_roundness, err_C1,err_C2,
                                                               err_C1filt,err_C1filt,LC_error_bound)

    return getcount(maxerr1)

# maybe switch to a random search, since we have too many model/hyperparam combinations
def runexperiment(n_loops, n_trials, gammalist, srlist, xcenlist, models_class, model_names):
    for model_name in model_names:
        for xcen in xcenlist:
            for gamma in gammalist:
                for sr in srlist:
                    run_list = []  # stores counts for a particular model and xcen. Clear this list each time, and re-initialize
                    for i in range(n_trials):
                        print(f"Model: {model_name} | gamma: {gamma} | sr: {sr} | xcen: {xcen} | iteration: {i} --------------------------------------")
                        if(models_class!='random'):
                            with open(f'models/{model_name}.npy', 'rb') as f:
                                model = np.load(f)
                                N = 500.
                                if(models_class=='struct'):
                                    N = model.shape[0]
                        if(models_class=='random'):
                            model_data = None
                            N = 500.
                        else:
                            model_data = model
                        try:
                            count = countgoodbad(models_class,model_data,n_loops,gamma,sr,xcen,N)
                            print(f"{model_name} (gamma = {gamma}, sr = {sr}, xcen = {xcen}, iteration = {i}): COUNT IS ",
                                  count)
                            run_list.append(count)  # add count to run list
                        except Exception as e:
                            print(f"{model_name} (gamma = {gamma}, sr = {sr}, xcen = {xcen}, iteration = {i}): COUNT IS ERROR!!!")
                            run_list.append('ERR')
                            print(e)
                    with open('counts.txt', 'a') as fd:
                        fd.write(f'{model_name}_gamma{gamma}_sr{sr}_xcen{xcen} counts for {n_trials} runs: \n{run_list}\n')

# initialize text file to store results...
with open('counts.txt', 'a') as fd:
    fd.write(f'\nNew set of runs on {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} ----------------------------------------------\n')

# #specify experimental parameters
# xcenlist = [0.0, 1.0, 5.0, 10.0, 50.0]
# gammalist = [5,25,50,75] # 5 to 100
# srlist = [0.5,1,1.5,2] #0.5 to 2
# n_loops = 100
# n_trials = 30

# quick run experimental parameters
xcenlist = [0.0]
gammalist = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
srlist = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
n_loops = 100
n_trials = 1

#candidate models to check
# ff_candsS = ['MBRdiag_wtol100','LHRdiag_wtol100','LHRstruct_wtol100','LHRdiag_wtol50','LHRdiag_wtol100',
#              'MBRdiag_wtol50', 'HemiRstruct_wtol100','HemiRdiag_wtol100','LHRdiag_wtol10'] #structure
# ff_candsD = ['HemiRdegdistn_wtol100_dtol10','LHRdegdistn_wtol3_dtol5','MBRdegdistn_wtol3_dtol3',
#              'SMLdegdistn'] #degree distn
# ff_candsW = ['HemiRweightdistn_wtol100','LHRweightdistn_wtol3','LHRweightdistnunique_wtol3',
#              'MBRweightdistn_wtol3','MBRweightdistnunique_wtol3'] #weight distn


ff_candsS = ['MBRdiag_wtol100','LHRdiag_wtol50'] #structure


ff_randVariations = ['rand100percentLH', 'rand75percentLH', 'rand50percentLH','rand25percentLH',
                     'LHRdiag_wtol50']
#runexperiment(n_loops,n_trials,[5.0],[1.4],xcenlist,'struct',ff_randVariations)


# try random
runexperiment(n_loops,n_trials,gammalist,srlist,xcenlist,'random',['random'])
# try struc
runexperiment(n_loops,n_trials,gammalist,srlist,xcenlist,'struct',ff_candsS)
# # try degdistn
# runexperiment(n_loops,n_trials,gammalist,srlist,xcenlist,'degdistn',ff_candsD)
# # try wdistn
# runexperiment(n_loops,n_trials,gammalist,srlist,xcenlist,'weightdistn',ff_candsW)

