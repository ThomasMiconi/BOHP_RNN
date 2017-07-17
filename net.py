import numpy as np

np.random.seed(0)

NBSTEPS = 30
NBNEUR = 20
ETA = .90
EPSILONW = .001
EPSILONY = .0
EPSILONH = .0


alpha = np.abs(np.random.rand(NBNEUR, NBNEUR)) *.1
hebb = np.zeros((NBNEUR, NBNEUR))
w = np.random.randn(NBNEUR, NBNEUR) * 1.1 / np.sqrt(NBNEUR)

# Not used for now
inputs = np.abs(np.random.rand(NBSTEPS, NBNEUR)) *.01
generaltgt = np.ones(NBNEUR)[:,None]  # Column vector



np.fill_diagonal(alpha, 0)  # No platic autapses
#alpha.fill(0)  # For now, no plasticity



estimgrad11 = 0; estimgrad10 = 0; estimgradx1=0; estimgrady1=0; estimgrady1y0=0; estimgrady0y0=0
yinit = np.random.rand(NBNEUR)
finalerrs = []; finalhebbs = []; finalys = []; finalxs=[]
wchange = np.random.randn(NBNEUR, NBNEUR) * .0001  


for numrun in range(3):
    xs=[]; ys=[]; errs=[]; tgts = []; yprevs = [] 
    y = yinit.copy()
    yprev = np.zeros_like(y); yprevprev = np.zeros_like(y)
    hebb.fill(0)
    dykdwba = np.zeros((NBNEUR, NBNEUR, NBNEUR))  # dykdwba[k, b, a] = dy_k/dWba
    dhkjdwba = np.zeros((NBNEUR, NBNEUR, NBNEUR, NBNEUR)) # dhkjdwba[k, j, b, a] = dh_kj/dWba
    dykdwbaprev = dykdwba.copy()

    w[1,3] += EPSILONW

    for numstep in range(NBSTEPS): 

        
        # Now we compute dH_kj(t)/dWba = ETA * dH_kj(t-1)/dWba + (1-ETA) [ dy_k(t-1)/dWba * y_j(t-2) + dy_j(t-2)/dWba * y_k(t-1) ]
        # (Reminder: at this point, y is really y(t-1) and yprev is really y(t-2); y(t) is computed below in the loop body)

        tmpdout1 = dhkjdwba 
        tmpdout2 = np.transpose(dykdwba[:,:,:,np.newaxis] * yprev, [0, 3, 1, 2])   # yprev_j * dy_k/dWba  
        tmpdout3 = np.transpose(dykdwbaprev[:,:,:,np.newaxis] * y , [3, 0, 1, 2]) #  y_k * dyprev_j/dWba  

        dhkjdwba = ETA * tmpdout1 + (1.0 - ETA) * (tmpdout2 + tmpdout3)

        # We compute hebb(t) now, because we need it for dxkdwba.
        hebb = ETA * hebb + (1.0 - ETA) * np.outer(y, yprev) # Note: At this stage, y is really y(t-1) and yprev is really y(t-2)

        # Now we compute dx_k(t)/dWba (we use this to compute dy_k(t)/dWba after y(t) is computed)

        tmpdout = np.tensordot(w, dykdwba, ([1], [0]))  # Somehow this works
        # Equivalent, slow way of doing the same thing:
        #for a in range(NBNEUR):
        #    for b in range(NBNEUR):
        #        for k in range(NBNEUR):
        #            for j in range(NBNEUR):
        #                dout[k, b, a] += w[k, j] * dykdwba[j, b,a]
        # Special case when k=b: 
        for b in range(NBNEUR):
            for a in range(NBNEUR):
                tmpdout[b,b,a] += y[a]

        # Now tmpdout = dx_k(t)/dW_ba - neglecting plasticity !
        # Now to add in plasticity effects...
        tmpdout1 = np.tensordot( (alpha * hebb), dykdwba, ([1], [0]) )  # Sum_j{ alpha_kj hebb_kj dy_j(t-1)/dWba } (very similar to the computation of tmpdout with w above)
        tmpdout2 = np.tensordot( (alpha[:,:, np.newaxis, np.newaxis] * dhkjdwba), y, ([1], [0]) ) # Sum_j{ alpha_kj dhebb_kj(t)/dWba y_j }

        
        dxkdwba = tmpdout + tmpdout1 + tmpdout2

        # To get dy_k(t)/dW_ba we just need to pass dxkdwba through the tanh
        # nonlinearity, which we do after the y(t) are computed, below.
        
        x = (w + alpha * hebb).dot(y)
        yprevprev = yprev.copy()
        yprev = y.copy()
        y = np.tanh(x)  # Finally computing y(t)

        dykdwbaprev = dykdwba.copy()
        dykdwba = (dxkdwba.T * (1.0 - y*y)).T  # Broadcasting along the last two dimensions to account for the tanh



        ys.append(y); tgts.append(generaltgt); yprevs.append(yprev); xs.append(x); 

    if numrun == 1: #and numstep == 0:
        pred_dxdwba = dxkdwba.copy()
    if numrun == 1: #and numstep == 0:
        pred_dydwba = dykdwba.copy()
            
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

