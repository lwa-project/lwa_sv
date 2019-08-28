
from collections import OrderedDict
from threading   import RLock
import functools

# Note: Don't forget to call this decorator as a function (e.g., @lru_cache())
# Note: This version stores one cache per class, not instance
def lru_cache(maxsize=128):
    """A decorator for cacheing/memoizing calls to a function"""
    def decorator(func):
        cached_func = lru_cache_impl(func, maxsize)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cached_func(*args, **kwargs)
        return wrapper
    return decorator
# Note: This version stores one cache per class instance
def lru_cache_method(maxsize=128):
    """A decorator for cacheing/memoizing calls to a member function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(this, *args, **kwargs):
            if not hasattr(this, '_method_cache'):
                this._method_cache = {}
            if func not in this._method_cache:
                this._method_cache[func] = lru_cache_impl(func, maxsize)
            return this._method_cache[func](this, *args, **kwargs)
        return wrapper
    return decorator

# TODO: Not needed? The plain version seems to work and avoids sequential waits
def threadsafe_lru_cache(maxsize=128):
    """Thread-safe (via a lock) version of lru_cache"""
    def decorator(func):
        cached_func = threadsafe_lru_cache_impl(func, maxsize)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cached_func(*args, **kwargs)
        return wrapper
    return decorator

class lru_cache_impl(OrderedDict):
    def __init__(self, func, maxsize):
        OrderedDict.__init__(self)
        self.func       = func
        self.maxsize    = maxsize
    def __call__(self, *args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        return self[key]
    def __missing__(self, key):
        args, kwargs = key[0], dict(key[1])
        result = self[key] = self.func(*args, **kwargs)
        if (self.maxsize is not None and
            len(self) > self.maxsize):
            self.popitem(last=False)
        return result
class threadsafe_lru_cache_impl(lru_cache_impl):
    def __init__(self, *args, **kwargs):
        lru_cache_impl.__init__(self, *args, **kwargs)
        self.lock = RLock()
    def __call__(self, *args, **kwargs):
        with self.lock:
            return lru_cache_impl.__call__(self, *args, **kwargs)

if __name__ == "__main__":
    import time
    import threading
    
    @lru_cache(maxsize=4)
    def fib(n):
        return n if n in (0,1) else fib(n-1) + fib(n-2)
    print fib(10)
    print fib
    print fib(11)
    print fib
    
    class MyObj(object):
        def __init__(self):
            pass
        @lru_cache()
        def foo(self, i):
            return i
    myobj = MyObj()
    print myobj.foo
    print myobj.foo(10)
    print myobj.foo(10)
    
    #@threadsafe_lru_cache(maxsize=4)
    @lru_cache(maxsize=4)
    def slow_func(i):
        time.sleep(1)
        return i
    def print_slow_func(i):
        print slow_func(i)
    threads = []
    for i in xrange(10):
        thread = threading.Thread(target=print_slow_func, args=(i%4,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
