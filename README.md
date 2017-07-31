## Backpropagation of Hebbian plasticity for recurrent networks

Backpropagation of Hebbian plasticity (BOHP) is an extension of the
backpropagation algorithm to networks with plastic, Hebbian synapses, previously
described [here](https://github.com/ThomasMiconi/LearningToLearnBOHP). With BOHP,
backpropagation can train both the (baseline) weights *and* the plasticity of
each connection. As a result, one can train networks to form memories and learn
from experience, much like animal brains. Thus, BOHP is a "learning to learn"
method, like RNN-based learning-to-learn methods
([1](https://link.springer.com/chapter/10.1007/3-540-44668-0_13),
[2](https://arxiv.org/abs/1611.02779), [3](https://arxiv.org/abs/1611.05763)),
or Memory Networks and Neural Turing Machines (e.g.
[4](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)), but more
biologically inspired.

### The good news

You can now train (small) recurrent neural networks with plastic, Hebbian
connections.  As a demonstration, the code in this repository trains an
*auto-associative memory.* An auto-associative (or content-addressable) memory
automatically memorizes the patterns it encounters, and can retrieve a complete memorized pattern when
exposed to a small part of that pattern.  

### The bad news

The algorithm we use for gradient computation is similar to Real Time
Recurrent Learning (RTRL), as well as a [related
algorithm](https://link.springer.com/chapter/10.1007/978-1-4471-2063-6_110)  by
J. Schmidhuber.  This algorithm has a major drawback: its spatial complexity is
*o(N<sup>4</sup>)* in the number of neurons *N* (because it needs to maintain one
quantity for each pair of connections). This makes it impractical for any large
network, and slow even for the small ones that we use here.

This problem can be addressed by switching to an algorithm based on
backpropagation through time (BPTT); however, the mathematics are somewhat
involved due to the additional temporal dependencies introduced by the plastic
weights. We intend to explore this direction in  future work.


### The demo program

The program trains a network to become an auto-associative memory. During each
"episode" (or "lifetime"), the network is exposed to three randomly chosen
*patterns* (vectors of *N* values, each of which is either +1 ot  -1, where *N*
is the number of neurons in the network).  Pattern presentation occurs by
feeding each value in the pattern as an input to the corresponding neuron in
the network. Each presentation lasts 6 time steps, and there are 4 time steps
with zero input after each presentation.  Finally, we randomly choose one of the three patterns 
"degrade" it (half of its bits are zero'ed out), and present it to the network.
The network must reproduce the full pattern in its output, completing the
zero'ed out bits with the appropriate value.

Errors are computed only on the last time step of the episode. For each neuron
the error is the square of the difference between the neuron's output and the
value of the corresponding bit (-1 or 1) in the pattern to be retrieved. We
learn to minimize mean square error with an Adam solver, based on the gradients
provided by BOHP (see below).

After training, the final networks look much like a Hopfield network, with
twists. All neurons have positive plastic connections to each other, with
essentially zero fixed (baseline) components. This allows them to build the
necessary connectivity to store the patterns presented duringeach episode.
However, they also have *negative* fixed-weight self-connections, allowing the
neurons to relax between successive pattern presentations.


### The equations

Equations for the gradients are provided in  `latex/equations.pdf`.


### The code

The code mainly contains two elements:

- `rnnbohp.py`: the module that implements BOHP. It contains only one function, `runNetwork`, which  runs the network and computes the gradients.

- `net.py`: the actual application, which trains a network to behave as an auto-associative memory, using the `runNetwork` function in `rnnbohp.py`

`runNetwork` taks as arguments the current parameters of the network (the fixed
baseline weights and plasticity coefficients, time constant of plasticity) and
the list of input vectors to be fed to the network. It runs the network,
computes the outputs, and computes the gradient of the output of each neuron, at
each time step, over the value of each connection parameter.

We tried to comment the code as much as possible. The application code in
`net.py` is relatively straightforward. The actual BOHP code in `rnnbohp.py` is
somewhat terse, being essentially a transcription of differential
equations into Numpy functions. Hopefully comments should help. Be sure to consult the original equations in `latex/equations.pdf`.

