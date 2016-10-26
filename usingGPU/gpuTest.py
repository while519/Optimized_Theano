#!/usr/bin/env python3

# In[1]:

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import theano


# In[2]:

vlen = 10*30*768
iters = 1000


# In[3]:

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))


# In[4]:
theano.printing.pydotprint(f, var_with_name_simple=True, compact=True, 
    outfile='nn-theano-f.png', format='png')

print(f.maker.fgraph.toposort())


# In[5]:

t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Results is %s" % (r,))


# In[6]:

if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

