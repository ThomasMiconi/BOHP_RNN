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

print "We're just trying to maximize the last y while minimizing the next-to-last y."

for numstep in range(20):
    ys, xs, dydws, dydalphas = runNetwork(w, alpha, ETA, NBSTEPS, inputs=inputs, yinit=yinit)
    print "last 2 y[0]:", ys[-2][0], ys[-1][0]
    w += .01 * (dydws[-1][0, :, :] - dydws[-2][0,:,:])
    alpha += .01 * (dydalphas[-1][0, :, :] - dydalphas[-2][0, :, :])


