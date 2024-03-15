import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

######### Functions ###############################################################################

### signal sampling (for signal with non uniform timestep) ########################################
# return two lists : t_signal and y_signal (with uniform timestep)
def _sample(t_signal_nuni, y_signal_nuni, delta_t_uni_exp, subref) :
    # t_signal_nuni (list/1D array) represents computation time
    # y_signal_nuni (list/1D array) represents the function y(t) we need to sample
    # delta_t_uni_exp (float/int) is the power wanted for the time step of the uniform signal
    # subref (bool) indicates if you want to refine the uniform signal obtained

    # Need to check the size of both y_signal_nuni and t_signal_nuni
    if len(t_signal_nuni) != len(y_signal_nuni) : raise NameError("No matching size for sampling")

    delta_t_uni = 10.0**(-delta_t_uni_exp)
    N_points_signal_nuni = len(t_signal_nuni)

    t_start = t_signal_nuni[0]
    t_end = t_signal_nuni[-1]
    TS = t_end - t_start
    N_points_signal_uni = int(TS/delta_t_uni + 1)

    t_signal_uni = np.linspace(t_start, t_end, N_points_signal_uni)
    t_signal_nuni_round = [0]*N_points_signal_nuni

    for i in range(0, N_points_signal_nuni):
        t_signal_nuni_round[i] = round(t_signal_nuni[i], delta_t_uni_exp)

    y_signal_nuni_round_fw = [0]*N_points_signal_nuni
    amp = -1e100
    for i in range(1, N_points_signal_nuni):
        if y_signal_nuni[i] >= 0:
            if t_signal_nuni_round[i] == t_signal_nuni_round[i-1]:
                if y_signal_nuni[i] > amp:
                    amp = y_signal_nuni[i]
                    y_signal_nuni_round_fw[i] = amp
            else:
                amp = y_signal_nuni[i]
                y_signal_nuni_round_fw[i] = amp
                amp = -1e100
                
    y_signal_nuni_round_bw = [0]*N_points_signal_nuni
    amp = -1e100
    for i in range(N_points_signal_nuni-2,-1, -1):
        if y_signal_nuni[i] >= 0:
            if t_signal_nuni_round[i] == t_signal_nuni_round[i+1]:
                if y_signal_nuni[i] > amp:
                    amp = y_signal_nuni[i]
                    y_signal_nuni_round_bw[i] = amp
            else:
                amp = y_signal_nuni[i]
                y_signal_nuni_round_bw[i] = amp
                amp = -1e100
            
    y_signal_nuni_round = [0]*N_points_signal_nuni
    for i in range(0, N_points_signal_nuni):
        y_signal_nuni_round[i] = max(y_signal_nuni_round_fw[i], y_signal_nuni_round_bw[i])

    buff = [y_signal_nuni_round[0]]
    for i in range(1, N_points_signal_nuni):
        if t_signal_nuni_round[i] == t_signal_nuni_round[i-1]:
            buff.append(y_signal_nuni_round[i])
        else:
            maxBuff = max(buff)
            for j in range(i-len(buff),i):
                y_signal_nuni_round[j] = maxBuff
            buff = [y_signal_nuni_round[i]]
            
    i = 1
    while i < N_points_signal_nuni:
        if i == len(t_signal_nuni_round) - 1:
            break
        if t_signal_nuni_round[i] == t_signal_nuni_round[i-1]:
            t_signal_nuni_round.pop(i)
            y_signal_nuni_round.pop(i)
            i -= 1
        i += 1

    fint = interp1d(t_signal_nuni_round, y_signal_nuni_round, kind='nearest')
    y_signal_uni = fint(t_signal_uni)

    t_signal = t_signal_uni
    y_signal = y_signal_uni

    if (subref == 'true'):
        ref_factor = 10
        N_points_signal_ref = (N_points_signal_uni - 1)*ref_factor + 1

        t_signal_uni_ref = np.linspace(0.0, TS, N_points_signal_ref)
        y_signal_uni_ref = []
        for i in range(1, N_points_signal_uni):
            t_ref = 0.0
            delta_t = t_signal_uni[i] - t_signal_uni[i-1]
            y = y_signal_uni[i]
            y_old = y_signal_uni[i-1]
            for j in range(0, ref_factor):
                t_ref = delta_t/float(ref_factor)*float(j)
                y_ref = 0.5*((y_old-y)*np.cos(np.pi/delta_t*t_ref) + y + y_old)
                y_signal_uni_ref.append(y_ref)
        y_signal_uni_ref.append(y_signal_uni[-1])
        t_signal = t_signal_uni_ref
        y_signal = y_signal_uni_ref
    
    return t_signal, y_signal

### FFT Transform #################################################################################
# return two 1D arrays : tf and yf
def _FFT(t_signal, y_signal, t_start=0, t_end=0, window_mode=1, N_windows=1, window_exp=1.0e-8) :
    # t_signal (list/1D array) represents computation time (uniform timestep)
    # y_signa (list/1D array) represents y(t) (uniform timestep)

    if t_start == t_end :
        t_start, t_end = t_signal[0], t_signal[-1]
        t, y = t_signal, y_signal
    else :
        t, y = [], []
        for i in range(0, len(t_signal)):
            if t_signal[i] >= t_start and t_signal[i] <= t_end:
                t.append(t_signal[i])
                y.append(y_signal[i])
    
    N_points = len(t)
    TS = t_end - t_start
    T_window = 2.0*TS/(N_windows+1)

    N_points_window = N_points*T_window/TS
    N_points_window = int(math.floor(N_points_window/2.0))
    N_points_window = 2*N_points_window

    t_window = np.array([[0.0] * N_points_window] * N_windows)
    y_window = np.array([[0.0] * N_points_window] * N_windows)

    if window_mode == 1:
        yf = np.array([0.0] * int(N_points_window//2))
        for i in range(0, N_windows):
            t_window[i,:] = t[int(i*N_points_window/2) : int((i+2)*N_points_window/2)]
            t_window[i,:] = t_window[i,:] - t_window[i,0]
            # re-adjust window length to exact window times
            T_window = t_window[i,-1]
            y_window[i,:] = y[int(i*N_points_window/2) : int((i+2)*N_points_window/2)]
            y_window[i,:] = y_window[i,:]*(0.5*(1.0 - np.cos(2.0*np.pi/T_window*t_window[i,:])))**window_exp
            yf_window = np.fft.fft(y_window[i,:])
            yf_window = 2.0/N_points_window*np.abs(yf_window[0:N_points_window//2])
            yf = yf + yf_window
        yf = yf/float(N_windows)
        tf = np.linspace(0.0, N_points_window/(2.0*T_window), N_points_window//2)
    else:
        yf = np.fft.fft(y)
        yf = 2.0/N_points*np.abs(yf[0:N_points//2])
        tf = np.linspace(0.0, N_points/(2.0*TS), N_points//2)
    
    return tf, yf
