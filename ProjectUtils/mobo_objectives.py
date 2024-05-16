from functools import wraps
import numpy as np

def glob_dummy(loc_fun):
    @wraps(loc_fun)
    def inner(xdic):
        #x_sorted = [xdic[p_name] for p_name in xdic.keys()] #it assumes x will be given as, e.g., dictionary
        #x_sorted = np.array(x_sorted)
        res = loc_fun(xdic)
        return res

    return inner

def glob_fun(loc_fun):
    @wraps(loc_fun)
    def inner(xdic):
        #x_sorted = [xdic[p_name] for p_name in xdic.keys()] #it assumes x will be given as, e.g., dictionary
        res = list(loc_fun(xdic))
        return res
    return inner
# Define the objectives
@glob_dummy
def f1(x):
  res = np.sum(x*x)
  return res #obj1

@glob_dummy
def f2(x):
  res = 1./(np.sum(x*x)+0.01)
  return res #obj2