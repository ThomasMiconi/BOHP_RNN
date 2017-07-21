import numpy as np
import sys

import rnnbohp
from rnnbohp import runNetwork



NBNEUR = 25
ETA = .90
EPSILON = .001
RNGSEED = 0
PLASTICITY = 1
LEARNINGRATE= .003
arguments = sys.argv[1:]
numarg = 0
while numarg < len(arguments):
    if arguments[numarg] == 'PLASTICITY':
        PLASTICITY = int(arguments[numarg+1])
    if arguments[numarg] == 'LEARNINGRATE':
        LEARNINGRATE = float(arguments[numarg+1])
    if arguments[numarg] == 'RNGSEED':
        RNGSEED = int(arguments[numarg+1])
    numarg += 2
np.random.seed(RNGSEED)


alpha = np.abs(np.random.rand(NBNEUR, NBNEUR)) *.01
w = np.random.randn(NBNEUR, NBNEUR) * 1.1 / np.sqrt(NBNEUR)
m1w = np.zeros_like(w)
m2w = np.zeros_like(w)
m1alpha = np.zeros_like(alpha)
m2alpha = np.zeros_like(alpha)

# Not used for now
generaltgt = np.ones(NBNEUR)[:,None]  # Column vector



np.fill_diagonal(alpha, 0)  # No platic autapses
#alpha.fill(0)  # For now, no plasticity



yinit = np.random.rand(NBNEUR); yinit.fill(0)
finalys = []


np.set_printoptions(precision=3)
FILENAME = "errs_adam_LEARNINGRATE"+str(LEARNINGRATE)+"PLASTICITY"+str(PLASTICITY)+"_RNGSEED"+str(RNGSEED)+".txt"
myfile = open(FILENAME, "w") 



PATTERNSIZE = NBNEUR - 4
PRESTIME = 5
INTERPRESDELAY = 5
TESTTIME = 2
NBPATTERNS = 2
#NBSTEPS = (PRESTIME + INTERPRESDELAY) * NBPATTERNS + INTERPRESDELAY + PRESTIME
NBSTEPS = (PRESTIME + INTERPRESDELAY) * NBPATTERNS +  PRESTIME
PROBADEGRADE = .5

print "Starting - "
print "Learning rate:", str(LEARNINGRATE), " RNGSEED:", str(RNGSEED), ", PLASTICITY:", str(PLASTICITY)
for numstep in range(3000):
    print numstep
    # Create the random patterns 
    #patterns = [(np.random.randint(2, size=PATTERNSIZE) *2 - 1) for ii in range(NBPATTERNS)]
    patterns = []; patterns.append(np.random.randint(2, size=PATTERNSIZE) *2 - 1); patterns.append(-patterns[0]) # Two symmetric patterns

    # Presentation of the patterns
    inputs = [np.zeros(NBNEUR) for ii in range (NBSTEPS)]
    for ii in range(NBPATTERNS):
        for nn in range(PRESTIME):
            inputs[ii * (PRESTIME+INTERPRESDELAY) + nn][:PATTERNSIZE] = patterns[ii][:]

    # Creating the test, partially zero'ed out pattern
    numtestpattern = np.random.randint(NBPATTERNS)
    testpattern = patterns[numtestpattern].copy()
    preservedbits = np.ones(PATTERNSIZE); preservedbits[:int(PROBADEGRADE * PATTERNSIZE)] = 0; np.random.shuffle(preservedbits)
    testpattern = testpattern *preservedbits

    for nn in range(PRESTIME):
        inputs[-PRESTIME+ nn][:PATTERNSIZE] = testpattern[:]


    
    if PLASTICITY == 0:
        alpha.fill(0)
    
    ys, xs, dydws, dydalphas = runNetwork(w, alpha, ETA, NBSTEPS, inputs=inputs, yinit=yinit)
    errs = [ys[-TESTTIME+n][:PATTERNSIZE] - patterns[numtestpattern] for n in range(TESTTIME)]  # reproduce the full test pattern.
    dws = [np.sum((errs[n] * dydws[-TESTTIME + n][:PATTERNSIZE,:,:].T).T, axis=0) for n in range(TESTTIME)]
    dalphas = [np.sum((errs[n] * dydalphas[-TESTTIME+ n][:PATTERNSIZE,:,:].T).T, axis=0) for n in range(TESTTIME)]

    BETA1 = .9; BETA2 = .999; ALPHA = .001
    dw = sum(dws)
    m1w = BETA1 * m1w + (1 - BETA1) * dw
    m2w = BETA1 * m2w + (1 - BETA2) * dw * dw
    m1wcorr = m1w / (1.0 - BETA1 ** (numstep+1))
    m2wcorr = m2w / (1.0 - BETA2 ** (numstep+1))
    dalpha = sum(dalphas)
    m1alpha = BETA1 * m1alpha + (1 - BETA1) * dalpha
    m2alpha = BETA1 * m2alpha + (1 - BETA2) * dalpha * dalpha
    m1alphacorr = m1alpha / (1 - BETA1 ** (numstep+1))
    m2alphacorr = m2alpha / (1 - BETA2 ** (numstep+1))
    
    w -= ALPHA * m1wcorr / (np.sqrt(m2wcorr) + 1e-8)
    alpha -= ALPHA * m1alphacorr / (np.sqrt(m2alphacorr) + 1e-8)

    print "last PRESTIME ys[:PATTERNSIZE]:", [x[:PATTERNSIZE] for x in ys[-PRESTIME:]]
    print "test pattern:", testpattern
    #print "Inputs: ", inputs[:INPUTLENGTH]
    totalerr = np.sum(np.abs(errs))
    print "Err:" , totalerr
    myfile.write(str(totalerr)+"\n"); myfile.flush()


myfile.close()


