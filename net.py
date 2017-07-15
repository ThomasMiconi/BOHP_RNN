import numpy as np

np.random.seed(0)

NBSTEPS =100 
NBNEUR = 100
ETA = .95
EPSILON = .001


alpha = np.abs(np.random.rand(NBNEUR, NBNEUR)) *.1
np.fill_diagonal(alpha, 0)  # No platic autapses
hebb = np.zeros((NBNEUR, NBNEUR))
w = np.random.randn(NBNEUR, NBNEUR) * 1.1 / np.sqrt(NBNEUR)

# Not used for now
inputs = np.abs(np.random.rand(NBSTEPS, NBNEUR)) *.01
generaltgt = np.ones(NBNEUR)[:,None]  # Column vector



alpha.fill(0)  # For now, no plasticity


yinit = np.random.rand(NBNEUR)
finalerrs = []
wchange = np.random.randn(NBNEUR, NBNEUR) * .0001  

for numrun in range(3):
    xs=[]; ys=[]; errs=[]; tgts = []; yprevs = []; weffs=[]
    y = yinit.copy()
    yprev = np.zeros_like(y)
    weff = w.copy()
    for numstep in range(NBSTEPS): 
        if numstep == 1:
            y[0] += numrun * EPSILON

        hebb = ETA * hebb + (1.0 - ETA) * np.outer(y, yprev) 
        
        # The following doesn't work
        #weff = w + alpha * hebb  # Need to work on this 
        # On the diagonal:
        #weff2 = w 
        #np.fill_diagonal(weff, 0)
        #weff += np.diag(np.diag(weff2) + alpha.dot ( (1.0-ETA) * yprev * y))

        weff = w.copy()

        #x = (w + alpha * hebb).dot(y)
        x = w.dot(y)
        y = np.tanh(x)
        #y = x.copy()
        ys.append(y); tgts.append(generaltgt); yprevs.append(yprev); xs.append(x); weffs.append(weff)
        #errs.append(err); 
        yprev = y.copy()
    finalerr = np.sum( (y - np.ones(NBNEUR)) * (y - np.ones(NBNEUR)) )
    finalerrs.append(finalerr)
    print "Run", numrun, ":", y

    if numrun == 1:
        derrfinaldys = []
        derrfinaldyprev = 2* ( y - np.ones(NBNEUR ))
        for numstep in reversed(range(NBSTEPS)): 
            derrfinaldyraw = (1-ys[numstep] * ys[numstep]) * derrfinaldyprev
            derrfinaldyprev = np.dot(weffs[numstep].T, derrfinaldyraw)
            derrfinaldys.append(derrfinaldyprev)
        derrfinaldys.reverse()

print "Calculated gradient:", derrfinaldys[1][0]
print "Empirical gradient:", (finalerrs[2]-finalerrs[0]) / (2 * EPSILON)


