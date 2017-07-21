import numpy as np
import matplotlib.pyplot as plt
import glob

plt.clf()
filestrs = glob.glob('errs*SEED0.txt')
for fs in filestrs:
    filez = glob.glob(fs[:-7]+'*')
    errs=[]
    minlen = 999999
    for f in filez:
        err = np.loadtxt(f)
        if err.size < minlen:
            minlen = err.size
        errs.append(np.loadtxt(f))
        
    meanerr= np.mean(np.vstack([x[:minlen] for x in errs]), axis=0)
    plt.plot(meanerr, label=fs, lw=1)

plt.show()



