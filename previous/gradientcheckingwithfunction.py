import numpy as np

import rnnbohp
from rnnbohp import runNetwork

np.random.seed(0)


NBSTEPS = 30
NBNEUR = 20
ETA = .90
#EPSILONW = .001; EPSILONALPHA = .001
EPSILON = .001


alpha = np.abs(np.random.rand(NBNEUR, NBNEUR)) *.1
w = np.random.randn(NBNEUR, NBNEUR) * 1.1 / np.sqrt(NBNEUR)

# Not used for now
generaltgt = np.ones(NBNEUR)[:,None]  # Column vector



np.fill_diagonal(alpha, 0)  # No platic autapses
#alpha.fill(0)  # For now, no plasticity



yinit = np.random.rand(NBNEUR); yinit.fill(0)
finalys = []

inputs= [np.zeros(4) for ii in range (NBSTEPS)]
for ii in range (1):
    inputs[ii].fill(1.0)

print "Run 1..."
ys, xs, dykdwbas, dykdalphabas = runNetwork(w, alpha, ETA, NBSTEPS, inputs=inputs, yinit=yinit)
finalys.append(ys[-1].copy())

print "Run 2..."
w += EPSILON; alpha += EPSILON
ys, xs, dykdwbas, dykdalphabas = runNetwork(w, alpha, ETA, NBSTEPS,inputs=inputs,  yinit=yinit)
pred_dydw = dykdwbas[-1].copy(); pred_dydalpha = dykdalphabas[-1].copy()
finalys.append(ys[-1].copy())

print "Run 3..."
w += EPSILON; alpha += EPSILON
ys, xs, dykdwbas, dykdalphabas = runNetwork(w, alpha, ETA, NBSTEPS, inputs=inputs, yinit=yinit)
finalys.append(ys[-1].copy())

print "Predicted gradient - x over w and alpha:", np.sum(np.sum(pred_dydw, axis=2), axis=1) + np.sum(np.sum(pred_dydalpha, axis=2), axis=1)   # This works if you modify *all* the ws and alphas by epsilon
#print "Predicted gradient - x over w and alpha:", pred_dydw[:,0,0]  + pred_dydalpha[:, 0, 0]   # This works if you only modify w[0,0] and alpha[0,0] by epsilon
print "Observed gradient - x over w and alpha:", (finalys[2]-finalys[0]) / (1e-12 + 2 * EPSILON)

