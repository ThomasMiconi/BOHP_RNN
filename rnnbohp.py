import numpy as np

def runNetwork(w, alpha, ETA, NBSTEPS, inputs=[], yinit=[]):
    NBNEUR = w.shape[0]
    INPUTSIZE = 0
    if (len(inputs) > 0):
        INPUTSIZE = len(inputs[0])
    if w.shape[1] != NBNEUR or alpha.shape[0] != NBNEUR or alpha.shape[1] != NBNEUR or ETA > 1.0 or ETA < 0.0 or (len(inputs) != NBSTEPS and len(inputs) != 0):
        raise ValueError("Incorrect parameters")
    xs=[]; ys=[]; dykdwbas = []; dykdalphabas = []
    if yinit.size > 0:
        if yinit.size != NBNEUR:
            raise ValueError("Incorrect yinit")
        else:
            y = yinit.copy()
    else:
        y = np.zeros(NBNEUR)
    yprev = np.zeros_like(y); yprevprev = np.zeros_like(y)
    hebb = np.zeros((NBNEUR, NBNEUR))
    dykdwba = np.zeros((NBNEUR, NBNEUR, NBNEUR))  # dykdwba[k, b, a] = dy_k/dW_ba
    dhkjdwba = np.zeros((NBNEUR, NBNEUR, NBNEUR, NBNEUR)) # dhkjdwba[k, j, b, a] = dh_kj/dW_ba
    dykdwbaprev = dykdwba.copy()
    dykdalphaba = np.zeros((NBNEUR, NBNEUR, NBNEUR))  # dykdalphaba[k, b, a] = dy_k/dAlpha_ba
    dhkjdalphaba = np.zeros((NBNEUR, NBNEUR, NBNEUR, NBNEUR)) # dhkjdalphaba[k, j, b, a] = dh_kj/dAlpha_ba
    dykdalphabaprev = dykdalphaba.copy()

    for numstep in range(NBSTEPS): 

        # Note the time conventions: hebb(t) is an input to y(t) (so it is computed beore y(t) for the same timestep), and y(t) is an input to hebb(t+1)
        
        # Now we compute dH_kj(t)/dWba = ETA * dH_kj(t-1)/dWba + (1-ETA) [ dy_k(t-1)/dWba * y_j(t-2) + dy_j(t-2)/dWba * y_k(t-1) ]
        # (Reminder: at this point, y is really y(t-1) and yprev is really y(t-2); y(t) is computed below in the loop body)
        tmpdout1 = dhkjdwba 
        # Z = X[:,:,:,np.newaxis] * Y means Z is composed of N=len(Y) submatrices (indexed by the last, new dimension), each of which is equal to Y[n] * X
        tmpdout2 = np.transpose(dykdwba[:,:,:,np.newaxis] * yprev, [0, 3, 1, 2])   # yprev_j * dy_k/dWba  
        tmpdout3 = np.transpose(dykdwbaprev[:,:,:,np.newaxis] * y , [3, 0, 1, 2]) #  y_k * dyprev_j/dWba  
        dhkjdwba = ETA * tmpdout1 + (1.0 - ETA) * (tmpdout2 + tmpdout3)

        # Same for the alphas. This section is exactly identical to the one for w's. (The computation of dyk's, below, is slightly different between w's and alpha's)
        tmpdout1 = dhkjdalphaba 
        tmpdout2 = np.transpose(dykdalphaba[:,:,:,np.newaxis] * yprev, [0, 3, 1, 2])   # yprev_j * dy_k/dWba  
        tmpdout3 = np.transpose(dykdalphabaprev[:,:,:,np.newaxis] * y , [3, 0, 1, 2]) #  y_k * dyprev_j/dWba  
        dhkjdalphaba = ETA * tmpdout1 + (1.0 - ETA) * (tmpdout2 + tmpdout3)

        # We compute hebb(t) now, because we need it for dxkdwba.
        hebb = ETA * hebb + (1.0 - ETA) * np.outer(y, yprev) # Note: At this stage, y is really y(t-1) and yprev is really y(t-2)

        # Now we compute dx_k(t)/dWba (we use this to compute dy_k(t)/dWba later, after y(t) is computed)
        tmpdout = np.tensordot(w, dykdwba, ([1], [0]))  # Somehow this works to compute Sum_j{ w_kj * dy_j(t-1)/dW_ba }
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
        
        # Now same thing for the alph's . The computation has slight differences.
        tmpdout = np.tensordot(w, dykdalphaba, ([1], [0])) 
        # Now to add in plasticity effects...
        tmpdout1 = np.tensordot( (alpha * hebb), dykdalphaba, ([1], [0]) )  # Sum_j{ alpha_kj hebb_kj dy_j(t-1)/dWba } (very similar to the computation of tmpdout with w above)
        tmpdout2 = np.tensordot( (alpha[:,:, np.newaxis, np.newaxis] * dhkjdalphaba), y, ([1], [0]) ) # Sum_j{ alpha_kj dhebb_kj(t)/dWba y_j }
        dxkdalphaba = tmpdout + tmpdout1 + tmpdout2
        for b in range(NBNEUR):
            for a in range(NBNEUR):
                dxkdalphaba[b,b,a] += y[a] * hebb[b,a]
        # To get dy_k(t)/dW_ba we just need to pass dxkdwba through the tanh
        # nonlinearity, which we do after the y(t) are computed, below.
        
        x = (w + alpha * hebb).dot(y)
        if INPUTSIZE > 0:
            x[:INPUTSIZE] += inputs[numstep]
        yprevprev = yprev.copy()
        yprev = y.copy()
        y = np.tanh(x)  # Finally computing y(t)

        dykdwbaprev = dykdwba.copy()
        dykdwba = (dxkdwba.T * (1.0 - y*y)).T  # Broadcasting along the last two dimensions to account for the tanh
        dykdalphabaprev = dykdalphaba.copy()
        dykdalphaba = (dxkdalphaba.T * (1.0 - y*y)).T  

        ys.append(y); xs.append(x); dykdwbas.append(dykdwba); dykdalphabas.append(dykdalphaba)
    return ys, xs, dykdwbas, dykdalphabas 




#if EPSILONW > 0:
#    print "Predicted gradient - x over w:", np.sum(np.sum(pred_dxdwba, axis=2), axis=1)  # This works if you modify *all* the ws by epsilon
#    print "Observed gradient - x over w:", (finalxs[2]-finalxs[0]) / (1e-12 + 2 * EPSILONW)
#    print "Predicted gradient - x over w:", np.sum(np.sum(pred_dydwba, axis=2), axis=1)
#    print "Observed gradient - y over w:", (finalys[2]-finalys[0]) / (1e-12 + 2 * EPSILONW)
#if EPSILONALPHA > 0:
#    print "Predicted gradient - x over alpha:", np.sum(np.sum(pred_dxdalphaba, axis=2), axis=1)  # This works if you modify *all* the alphas by epsilon
#    print "Observed gradient - x over alpha:", (finalxs[2]-finalxs[0]) / (1e-12 + 2 * EPSILONALPHA)
#    print "Predicted gradient - x over alpha:", np.sum(np.sum(pred_dydalphaba, axis=2), axis=1)
#    print "Observed gradient - y over alpha:", (finalys[2]-finalys[0]) / (1e-12 + 2 * EPSILONALPHA)

