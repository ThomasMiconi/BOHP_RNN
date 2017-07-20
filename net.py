import numpy as np

import rnnbohp
from rnnbohp import runNetwork



NBSTEPS = 10
NBNEUR = 30
ETA = .90
EPSILON = .001
RNGSEED = 1
np.random.seed(RNGSEED)

alpha = np.abs(np.random.rand(NBNEUR, NBNEUR)) *.01
w = np.random.randn(NBNEUR, NBNEUR) * 1.1 / np.sqrt(NBNEUR)

# Not used for now
generaltgt = np.ones(NBNEUR)[:,None]  # Column vector



np.fill_diagonal(alpha, 0)  # No platic autapses
#alpha.fill(0)  # For now, no plasticity



yinit = np.random.rand(NBNEUR); yinit.fill(0)
finalys = []


np.set_printoptions(precision=3)
FILENAME = "errs_withoutplast_RNGSEED"+str(RNGSEED)+".txt"
myfile = open(FILENAME, "w") 



INPUTSIZE = 6
INPUTLENGTH = 6
for numstep in range(5000):
    print numstep
    inputs= [np.zeros(INPUTSIZE) for ii in range (NBSTEPS)]
    for ii in range (INPUTLENGTH):
        inputs[ii].fill(-1)
        inputs[ii][np.random.randint(INPUTSIZE)] = 1
    
    
    #alpha.fill(0)
    
    
    ys, xs, dydws, dydalphas = runNetwork(w, alpha, ETA, NBSTEPS, inputs=inputs, yinit=yinit)
    errs = [ys[-INPUTLENGTH+n][:INPUTSIZE] - inputs[n] for n in range(INPUTLENGTH)]  # reproduce the input sequences. Inputs are also outputs!
    #errs = [0, 0, 0, -1, 1]
    dws = [np.sum((errs[n] * dydws[-INPUTLENGTH + n][:INPUTSIZE,:,:].T).T, axis=0) for n in range(INPUTLENGTH)]
    w -= .003 * sum(dws)
    dalphas = [np.sum((errs[n] * dydalphas[-INPUTLENGTH + n][:INPUTSIZE,:,:].T).T, axis=0) for n in range(INPUTLENGTH)]
    alpha -= .003 * sum(dalphas)
    print "last INPUTLENGTH ys[:INPUTSIZE]:", [x[:INPUTSIZE] for x in ys[-INPUTLENGTH:]]
    print "Inputs: ", inputs[:INPUTLENGTH]
    totalerr = np.sum(np.abs(errs))
    print "Err:" , 
    myfile.write(str(totalerr)+"\n")

myfile.close()


