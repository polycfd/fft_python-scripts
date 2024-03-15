# fft_python-scripts 

The goal of this repo is to gather functions to use efficiently FFT (Fast Fourier Transform), even for sets of data with non uniform timestep

## Main file : Signal_processing.py

This file contains the two main functions : _sample and _FFT

### _sample
The two main inputs of this function are *t_signal_nuni* and *y_signal_nuni*, representing respectively the computation time and value of y = f(t). Both are list / 1D array.

*delta_t_uni_exp* (int) represents the power wanted for the uniform timestep. For example, if *delta_t_uni_exp* = 10, the timestep will be $10^{-10}$ s.

Lastly, *subref* is a boolean that indicates if you want to refine further the uniform timetsep signal you obtain by using this function. By default it's False.

This function is used because *t_signal_nuni* is not created with an uniform timestep. The main goal of this function is to sample the signal and project it on an timegrid with uniform timestep, in order to use FFT later.

This function returns two lists : *t_signal* and *y_signal*, with uniform timestep

### _FFT
The two main inputs are lists / 1D arrays : *t_signal* and *y_signal*. The goal of this function is to determine the FFT of your signal.

*t_start* and *t_end* could be used if you only want to determine FFT for a specific part of your signal (like for instance after a transitory regime)

*window_mode*, *N_windows* and *window_exp* control the cut of the original signal into several window times in which a FFT is performed. Important to know, *N_windows* must be uneven.

This function returns two 1D arrays : *tf* (frequency) and *yf*, representing the FFT of your input signal