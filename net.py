import numpy as np

np.random.seed(0)

NBSTEPS = 100
NBNEUR = 50
ETA = .90
EPSILONW = .001
EPSILONY = .0
EPSILONH = .0


alpha = np.abs(np.random.rand(NBNEUR, NBNEUR)) *.1
np.fill_diagonal(alpha, 0)  # No platic autapses
hebb = np.zeros((NBNEUR, NBNEUR))
w = np.random.randn(NBNEUR, NBNEUR) * 1.1 / np.sqrt(NBNEUR)

# Not used for now
inputs = np.abs(np.random.rand(NBSTEPS, NBNEUR)) *.01
generaltgt = np.ones(NBNEUR)[:,None]  # Column vector



alpha.fill(0)  # For now, no plasticity



estimgrad11 = 0; estimgrad10 = 0; estimgradx1=0; estimgrady1=0; estimgrady1y0=0; estimgrady0y0=0
yinit = np.random.rand(NBNEUR)
finalerrs = []; finalhebbs = []; finalys = []; finalxs=[]
wchange = np.random.randn(NBNEUR, NBNEUR) * .0001  


for numrun in range(3):
    xs=[]; ys=[]; errs=[]; tgts = []; yprevs = []; weffs=[]
    y = yinit.copy()
    yprev = np.zeros_like(y); yprevprev = np.zeros_like(y)
    weff = w.copy()
    hebb.fill(0)
    dykdwba = np.zeros((NBNEUR, NBNEUR, NBNEUR))

    w[1,3] += EPSILONW

    for numstep in range(NBSTEPS): 

        tmpdout = np.tensordot(w, dykdwba, ([1], [0]))  # Somehow this works
        # Equivalent, slow way of doing the same thing:
        #for a in range(NBNEUR):
        #    for b in range(NBNEUR):
        #        for k in range(NBNEUR):
        #            for j in range(NBNEUR):
        #                dout[k, b, a] += w[k, j] * dykdwba[j, b,a]
        dxkdwba = tmpdout 
        # Special case when k=b: 
        for b in range(NBNEUR):
            for a in range(NBNEUR):
                dxkdwba[b,b,a] += y[a]
        
        # Now dxkdwba = dx_k(t)/dW_ba
        # To get dy_k(t)/dW_ba we just need to pass it through the tanh, which
        # we do after the y(t) are computed, below.

        
        hebb = ETA * hebb + (1.0 - ETA) * np.outer(y, yprev) # Note: At this stage, y is really y(t-1) and yprev is really y(t-2)
        weff = w.copy()

        x = (w + alpha * hebb).dot(y)
        yprevprev = yprev.copy()
        yprev = y.copy()
        y = np.tanh(x)  # Finally computing y(t)

        dykdwba = (dxkdwba.T * (1.0 - y*y)).T  # Broadcasting along the last two dimensions to account for the tanh

        if numrun == 1: #and numstep == 0:
            pred_dxdwba = dxkdwba.copy()
        if numrun == 1: #and numstep == 0:
            pred_dydwba = dykdwba.copy()


        ys.append(y); tgts.append(generaltgt); yprevs.append(yprev); xs.append(x); weffs.append(weff)
            
    finalxs.append(x)
    finalys.append(y)

    finalerr = np.sum( (y - np.ones(NBNEUR)) * (y - np.ones(NBNEUR)) )
    finalerrs.append(finalerr)
    print "Run", numrun, ":", y


print "Predicted gradient - x:", pred_dxdwba[:,1,3]
print "Observed gradient - x:", (finalxs[2]-finalxs[0]) / (1e-12 + 2 * EPSILONW)
print "Predicted gradient - y:", pred_dydwba[:,1,3]
print "Observed gradient - y:", (finalys[2]-finalys[0]) / (1e-12 + 2 * EPSILONW)
#print "Predicted gradient - err:", derrfinaldys[1][0]
#print "Observed gradient - err:", (finalerrs[2]-finalerrs[0]) / (1e-12 + 2 * EPSILONY)

