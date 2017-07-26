import numpy as np
import sys

import rnnbohp
from rnnbohp import runNetwork



NBNEUR = 32  # Note: Apparently, pattern size must be a multiple of 4 for orthogonalizing to work...
ETA = .96
EPSILON = .001
RNGSEED = 0
PLASTICITY = 1
LEARNINGRATE= .003
arguments = sys.argv[1:]
numarg = 0
while numarg < len(arguments):
    if arguments[numarg] == 'PLASTICITY':
        PLASTICITY = int(arguments[numarg+1])
    if arguments[numarg] == 'ALPHA':
        ALPHA= float(arguments[numarg+1])
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





PATTERNSIZE = NBNEUR - 4
NBPRESCYCLES = 3
PRESTIME = 6
PRESTIMETEST = 4
INTERPRESDELAY = 4
TESTTIME = 2
NBPATTERNS = 3
#NBSTEPS = (PRESTIME + INTERPRESDELAY) * NBPATTERNS + INTERPRESDELAY + PRESTIME
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) +  PRESTIMETEST
PROBADEGRADE = .5

ADAM = True
BETA1 = .9; BETA2 = .999; 
#ALPHA = .001
ALPHA = .0003
#ALPHA = .0001

np.set_printoptions(precision=3)
SUFFIX = "stronginputs_multiplepres_orthogpatterns_adam_NBNEUR"+str(NBNEUR)+"_PATTERNSIZE"+str(PATTERNSIZE)+"__ALPHA"+str(ALPHA)+"_NBPATTERNS"+str(NBPATTERNS)+"_LEARNINGRATE"+str(LEARNINGRATE)+"_PLASTICITY"+str(PLASTICITY)+"_RNGSEED"+str(RNGSEED)

myerrorfile = open("errs_"+SUFFIX+".txt", "w") 

print "Starting - "
print "Learning rate:", str(LEARNINGRATE), " RNGSEED:", str(RNGSEED), ", PLASTICITY:", str(PLASTICITY)
listerrs=[]; listtestpatterns=[]; listinputs=[]
for numstep in range(10000):
    print numstep
    
    # Create the random patterns 
    seedp = np.ones(PATTERNSIZE); seedp[:PATTERNSIZE/2] = -1
    #patterns = [np.random.permutation(seedp) for ii in range(NBPATTERNS)] # zero-sum patterns
    #patterns = []; patterns.append(np.random.randint(2, size=PATTERNSIZE) *2 - 1); patterns.append(-patterns[0]) # Two symmetric patterns
    # Orthogonal zero-sum patterns, brute force
    # (Orthogonalizing, by itself, seems to have little impact..)
    print "Generating patterns.."
    patterns=[]
    for nump in range(NBPATTERNS):
        sumdotsp = -1
        while sumdotsp != 0:
            p = np.random.permutation(seedp)
            sumdotsp = sum(np.abs([p.dot(pprevious) for pprevious in patterns]))
        patterns.append(p)
    print "patterns generated!"


    # Presentation of the patterns
    inputs = [np.zeros(NBNEUR) for ii in range (NBSTEPS)]
    for ii in range(NBSTEPS):
        inputs[ii][-1] = 10.0  # bias
        #inputs[ii][-2] = -1.0 # presenting / non-presenting
    for nc in range(NBPRESCYCLES):
        np.random.shuffle(patterns)
        for ii in range(NBPATTERNS):
            for nn in range(PRESTIME):
                numi =nc * (NBPATTERNS * (PRESTIME+INTERPRESDELAY)) + ii * (PRESTIME+INTERPRESDELAY) + nn 
                inputs[numi][:PATTERNSIZE] = patterns[ii][:]
                #inputs[numi][-2] = 1.0  # Presenting!


    # Creating the test, partially zero'ed out pattern
    numtestpattern = np.random.randint(NBPATTERNS)
    testpattern = patterns[numtestpattern].copy()
    preservedbits = np.ones(PATTERNSIZE); preservedbits[:int(PROBADEGRADE * PATTERNSIZE)] = 0; np.random.shuffle(preservedbits)
    testpattern = testpattern *preservedbits

    for nn in range(PRESTIMETEST):
        inputs[-PRESTIMETEST + nn][:PATTERNSIZE] = testpattern[:]
        #inputs[-PRESTIMETEST + nn][-2] = 1.0

    for ii in inputs:
        ii  *= 20.0
    
    if PLASTICITY == 0:
        alpha.fill(0)
    
    ys, xs, dydws, dydalphas = runNetwork(w, alpha, ETA, NBSTEPS, inputs=inputs, yinit=yinit)
    errs = [ys[-TESTTIME+n][:PATTERNSIZE] - patterns[numtestpattern] for n in range(TESTTIME)]  # reproduce the full test pattern, please
    dws = [np.sum((errs[n] * dydws[-TESTTIME + n][:PATTERNSIZE,:,:].T).T, axis=0)  for n in range(TESTTIME)] 
    dalphas = [np.sum((errs[n] * dydalphas[-TESTTIME+ n][:PATTERNSIZE,:,:].T).T, axis=0)   for n in range(TESTTIME)]

    # Adam solver
    if ADAM:
        dw = sum(dws) / TESTTIME
        m1w = BETA1 * m1w + (1 - BETA1) * dw
        m2w = BETA1 * m2w + (1 - BETA2) * dw * dw
        m1wcorr = m1w / (1.0 - BETA1 ** (numstep+1))
        m2wcorr = m2w / (1.0 - BETA2 ** (numstep+1))
        dalpha = sum(dalphas) / TESTTIME
        m1alpha = BETA1 * m1alpha + (1 - BETA1) * dalpha
        m2alpha = BETA1 * m2alpha + (1 - BETA2) * dalpha * dalpha
        m1alphacorr = m1alpha / (1 - BETA1 ** (numstep+1))
        m2alphacorr = m2alpha / (1 - BETA2 ** (numstep+1))
        
        w -= ALPHA * m1wcorr / (np.sqrt(m2wcorr) + 1e-8)
        alpha -= ALPHA * m1alphacorr / (np.sqrt(m2alphacorr) + 1e-8)
    else:
        w -= np.clip(LEARNINGRATE * sum(dws) / TESTTIME, -3e-4, 3e-4)
        alpha -= np.clip(LEARNINGRATE * sum(dalphas) / TESTTIME, -3e-4, 3e-4)
        print "Clippable ws:", np.sum(np.abs(LEARNINGRATE * sum(dws) ) / TESTTIME > 3e-4), " out of ", sum(dws).size

    if (numstep+1) % 10 == 0:
        params = {}
        params{'w'} = w; params{'alpha'} = alpha; params{'errs'} = listerrs; params{'inputs'} = listinputs; params{'testpatterns'} = listtestpatterns
        pickle.dump(params, open("results_"+SUFFIX+".pkl", "wb"))
    print "last PRESTIME ys[:PATTERNSIZE]:", [x[:PATTERNSIZE] for x in ys[-PRESTIME:]]
    print "test pattern:", testpattern
    #print "Inputs: ", inputs[:INPUTLENGTH]
    totalerr = np.sum(np.abs(errs))
    listinputs.append(inputs)
    listtestpatterns.append(testpattern)
    listerrs.append(totalerr)
    print "Err:" , totalerr
    myerrorfile.write(str(totalerr)+"\n"); myerrorfile.flush()


myerrorfile.close()


