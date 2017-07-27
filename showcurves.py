import numpy as np
import matplotlib.pyplot as plt
import glob
from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):


    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


plt.clf()
#filestrs = glob.glob('errs*orth*__*SEED0.txt')
filestrs = glob.glob('errs*l1n*__*SEED0.txt')
colorz = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#filestrs = glob.glob('errs*SEED0.txt')
for (numtype, fs) in enumerate(filestrs):
    filez = glob.glob(fs[:-30]+'*')
    errs=[]
    minlen = 999999
    for f in filez:
        err = np.loadtxt(f)#[::10]
        if err.size < minlen:
            minlen = err.size
        errs.append(np.loadtxt(f))
        
    #meanerr= np.mean(np.vstack([x[:minlen] for x in errs]), axis=0)
    #plt.plot(meanerr, label=fs, lw=1)
    for curve in errs:
        plt.plot(savitzky_golay(curve, 51, 3), colorz[numtype], lw=1)

plt.show()



