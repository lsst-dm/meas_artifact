
import multiprocessing

####################################################
#
# Parallelize with multiprocessing
# 
# Example usage, assume we want to run a function foo()
#   which takes an integer.  We'll run it 100 times,
#   with 10 threads
#
# threads = 10
# n = 100
# mp = MapFunc(threads, foo)
# for i in range(100):
#     mp.add(i)
# mp.run()
# del mp
#
#####################################################
        
class _Caller(object):
    def __init__(self, func):
        self.func = func
    def __call__(self, args):
        return self.func(*args[0], **args[1])
    
class MapFunc(object):
    """Class to wrap the map() function with optional threading"""
    def __init__(self, threads, func, maxtasksperchild=1):
        self.func = func
        self.args = []
        if threads > 1:
            self.pool = multiprocessing.Pool(threads, maxtasksperchild=maxtasksperchild)
            self.map = self.pool.map
        else:
            self.map = map
    def add(self, *args, **kwargs):
        self.args.append((args, kwargs))
    def run(self):
        caller = _Caller(self.func)
        return self.map(caller, self.args)

