import numpy as np

np.random.seed(0)

NBSTEPS =30 
NBNEUR = 100
ETA = .90
EPSILONY = .0
EPSILONH = .001


alpha = np.abs(np.random.rand(NBNEUR, NBNEUR)) *.1
np.fill_diagonal(alpha, 0)  # No platic autapses
hebb = np.zeros((NBNEUR, NBNEUR))
w = np.random.randn(NBNEUR, NBNEUR) * 1.1 / np.sqrt(NBNEUR)

# Not used for now
inputs = np.abs(np.random.rand(NBSTEPS, NBNEUR)) *.01
generaltgt = np.ones(NBNEUR)[:,None]  # Column vector



#alpha.fill(0)  # For now, no plasticity



estimgrad11 = 0; estimgrad10 = 0; estimgradx1=0; estimgrady1=0
yinit = np.random.rand(NBNEUR)
finalerrs = []; finalhebbs = []; finalys = []; finalxs=[]
wchange = np.random.randn(NBNEUR, NBNEUR) * .0001  

for numrun in range(3):
    xs=[]; ys=[]; errs=[]; tgts = []; yprevs = []; weffs=[]
    y = yinit.copy()
    yprev = np.zeros_like(y)
    weff = w.copy()
    hebb.fill(0)
    for numstep in range(NBSTEPS): 
        if numstep == 1:
            y[0] += numrun * EPSILONY

        hebb = ETA * hebb + (1.0 - ETA) * np.outer(y, yprev) # Note: At this stage, y is really y(t-1) and yprev is really y(t-2)
        #for nr in range(NBNEUR):
        #    for nc in range(NBNEUR):
        #        hh = hebb[nr][nc]
        #        hebb[nr][nc] = ETA * hh + (1.0-ETA) * yprev[nc] * y[nr]
        
        if numstep == 28:
            hebb[1,0] += numrun * EPSILONH
            if numrun == 1:
                # Estimate the gradient of hebb(1,1)(t+1) over hebb(1,0)(t) (so v,u,b = 1, a=0)
                estimgrad11 = (1.0 - ETA) * y[1] * alpha[1,0] * y[0]  # Again, y is really y(t-1) here
                estimgradx1 = alpha[1,0] * y[0]  
        if numstep == 29:
            finalhebbs.append(hebb.copy())

        # The following doesn't work
        #weff = w + alpha * hebb  # Need to work on this 
        # On the diagonal:
        #weff2 = w 
        #np.fill_diagonal(weff, 0)
        #weff += np.diag(np.diag(weff2) + alpha.dot ( (1.0-ETA) * yprev * y))
        weff = w.copy()

        x = (w + alpha * hebb).dot(y)
        yprev = y.copy()
        y = np.tanh(x)
        if numstep == 28:
            finalxs.append(x.copy())
            finalys.append(y.copy())
            estimgrady1 = estimgradx1 * (1.0 - y[1]*y[1])
            estimgrad10 = ETA + (1.0 - ETA) * yprev[0] * alpha[1,0] * yprev[0] * (1.0 - y[1] * y[1])
            estimgrad11 = (1.0 - ETA) * yprev[1] * alpha[1,0] * yprev[0] * (1.0 - y[1] * y[1])
        ys.append(y); tgts.append(generaltgt); yprevs.append(yprev); xs.append(x); weffs.append(weff)
        #errs.append(err); 
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

print "Predicted gradient - err:", derrfinaldys[1][0]
print "Observed gradient - err:", (finalerrs[2]-finalerrs[0]) / (1e-12 + 2 * EPSILONY)

print "Predicted gradient - x(t) over hebb(t):", estimgradx1
print "Observed gradient - x(t) over hebb(t):", (finalxs[2][1]-finalxs[0][1]) / (1e-12+2 * EPSILONH)
print "Predicted gradient - y(t) over hebb(t):", estimgrady1
print "Observed gradient - y(t) over hebb(t):", (finalys[2][1]-finalys[0][1]) / (1e-12+2 * EPSILONH)
print "Predicted gradient - hebb (auto t to t+1):", estimgrad10
print "Observed gradient - hebb (auto t to t+1):", (finalhebbs[2][1,0]-finalhebbs[0][1,0]) / (1e-12+2 * EPSILONH)
print "Predicted gradient - hebb(1,1,t+1) over hebb(1,0,t):", estimgrad11
print "Observed gradient - hebb(1,1,t+1) over hebb(1,0,t):", (finalhebbs[2][1,1]-finalhebbs[0][1,1]) / (1e-12+2 * EPSILONH)


