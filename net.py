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

# Not used for now
generaltgt = np.ones(NBNEUR)[:,None]  # Column vector



np.fill_diagonal(alpha, 0)  # No platic autapses
#alpha.fill(0)  # For now, no plasticity



yinit = np.random.rand(NBNEUR); yinit.fill(0)
finalys = []


np.set_printoptions(precision=3)
FILENAME = "errs_LEARNINGRATE"+str(LEARNINGRATE)+"PLASTICITY"+str(PLASTICITY)+"_RNGSEED"+str(RNGSEED)+".txt"
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
    w -= LEARNINGRATE * sum(dws)
    dalphas = [np.sum((errs[n] * dydalphas[-TESTTIME+ n][:PATTERNSIZE,:,:].T).T, axis=0) for n in range(TESTTIME)]
    alpha -= LEARNINGRATE * sum(dalphas)
    print "last PRESTIME ys[:PATTERNSIZE]:", [x[:PATTERNSIZE] for x in ys[-PRESTIME:]]
    print "test pattern:", testpattern
    #print "Inputs: ", inputs[:INPUTLENGTH]
    totalerr = np.sum(np.abs(errs))
    print "Err:" , totalerr
    myfile.write(str(totalerr)+"\n"); myfile.flush()


myfile.close()


