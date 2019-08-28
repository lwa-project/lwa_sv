
"""
A thread pool class with a few useful features

** TODO: Try to replace timeouts with 'stop events' pushed onto the queues
         This will allow simple blocking get() calls and zero shutdown delay

Features: Worker threads created on-demand and then re-used
          Task return values are retrieved and provided in original call order
          Non-daemon worker threads exit cleanly when the pool is detroyed
          Optional limit on the total number of worker threads
          Exceptions in tasks are caught and returned as result values
"""

# Note: No need to join() workers here as long as wait() is always
#       called at the end. The daemon threads will just block on
#       the empty queue and get inconsequentially killed at exit.

from Queue import Queue#, Empty
#from StoppableThread import StoppableThread
from ConsumerThread import ConsumerThread
import time

class TimeoutQueue(Queue):
    """Adds a timeout parameter to the join() method of Queue"""
    def join(self, timeout=None):
        if timeout is None:
            Queue.join(self)
            return True
        self.all_tasks_done.acquire()
        try:
            endtime = time.time() + timeout
            while self.unfinished_tasks:
                remaining = endtime - time.time()
                if remaining <= 0.0:
                    return False # Timed out
                self.all_tasks_done.wait(remaining)
            return True
        finally:
            self.all_tasks_done.release()

class Worker(ConsumerThread):
    count = 0
    def __init__(self, task_queue, result_queue=None,
                daemon=None, pool_name=None):
        ConsumerThread.__init__(self, task_queue)
        self.result_queue = result_queue
        if daemon is not None:
            self.daemon = daemon
        self.name = "%s.Worker-%i" % (pool_name, Worker.count+1)
        Worker.count += 1
        self.start()
    def process(self, task):
        idx, func, args, kwargs = task
        try:
            ret = func(*args, **kwargs)
        except Exception, e:
            ret = e
            # TODO: Any better way to report exceptions? Specify a log?
            if self.result_queue is None:
                print ("ERROR: Uncaught exception in %s: %s %s"%
                    (self.name, e, type(e)))
        finally:
            if self.result_queue is not None:
                self.result_queue.put( (idx, ret) )

class ThreadPool(object):
    """Pool of threads consuming tasks from a queue"""
    count = 0
    def __init__(self, max_workers, name=None):
        self.tasks   = TimeoutQueue(max_workers)
        self.workers = []
        self.name    = name
        if self.name is None:
            self.name = "ThreadPool-%i" % (ThreadPool.count+1)
            ThreadPool.count += 1
        self.spawn_workers(max_workers)
    def spawn_workers(self, n):
        if self.tasks.maxsize > 0:
            n = min(n, self.tasks.maxsize)
        for _ in xrange(n):
            self.workers.append( Worker(self.tasks, daemon=True,
                                        pool_name=self.name) )
    def add_task(self, func, *args, **kwargs):
        """Launch the given func as a standalone job with no return value"""
        self.tasks.put( (None, func, args, kwargs) )
        #if len(self.workers) < self.tasks.qsize():
        #    self.spawn_workers(1)
    def wait(self, timeout=None):
        """Wait for completion of all the tasks in the queue"""
        return self.tasks.join(timeout)

class FuturePool(object):
    """Pool of threads consuming tasks from a queue and saving return values"""
    count = 0
    def __init__(self, max_workers, name=None):
        self.tasks   = Queue(max_workers)
        self.results = Queue(max_workers)
        self.workers = []
        self.ntask   = 0
        self.name    = name
        if self.name is None:
            self.name = "FuturePool-%i" % (FuturePool.count+1)
            FuturePool.count += 1
        self.spawn_workers(max_workers)
    def __del__(self):
        self.join_workers()
    def spawn_workers(self, n):
        if self.tasks.maxsize > 0:
            n = min(n, self.tasks.maxsize)
        for _ in xrange(n):
            self.workers.append( Worker(self.tasks, self.results,# daemon=True,
                                        pool_name=self.name) )
    def join_workers(self):
        #print "Stopping workers"
        for worker in self.workers:
            worker.request_stop()
        #print "Joining workers"
        for worker in self.workers:
            worker.join()
        #print "JOINED"
    def add_task(self, func, *args, **kwargs):
        """Asynchronously call the given func and save the return value"""
        idx = self.ntask
        self.tasks.put( (idx, func, args, kwargs) )
        self.ntask += 1
        #if len(self.workers) < self.ntask:
        #    self.spawn_workers(1)
    def wait(self):
        """Wait for completion of all the tasks in the queue and return
            list of return values sorted by original call order."""
        self.tasks.join()
        results = [self.results.get() for _ in xrange(self.ntask)]
        self.ntask = 0
        ret = [r[1] for r in sorted(results)]
        return ret

class ObjectPool(list):
    """A specialised list that provides asynchronous parallel access to members
        E.g., objs = ObjectPool(['a', 'bb', 'ccc'])
              print objs.upper()    # --> ['AA', 'BB', 'CCC']
              print objs.count('b') # --> [0, 2, 0]
              objs2 = ObjectPool([MyObj(0), MyObj(1), MyObj(2)])
              objs2.val = [-1, -2, -3]
              print objs2.val       # --> [-1, -2, -3]
    """
    def __init__(self, objs=[], future_pool=None):
        list.__init__(self, objs)
        if future_pool is not None:
            list.__setattr__(self, 'future_pool', future_pool)
        else:
            list.__setattr__(self, 'future_pool', FuturePool(len(self)))
    def __getattr__(self, item):
        for obj in self:
            if not hasattr(obj, item):
                obj.__getattribute__(item) # Induce exception
            self.future_pool.add_task(obj.__getattribute__, item)
        return ObjectPool(self.future_pool.wait())#, self.future_pool)
    def __call__(self, *args, **kwargs):
        for obj in self:
            self.future_pool.add_task(obj.__call__, *args, **kwargs)
        return ObjectPool(self.future_pool.wait())#, self.future_pool)
    # TODO: This works, but not sure if causing subtle issues
    def __setattr__(self, item, values):
        for obj,val in zip(self,values):
            self.future_pool.add_task(obj.__setattr__, item, val)
        rets = self.future_pool.wait()
        for ret in rets:
            if isinstance(ret, Exception):
                raise ret

if __name__ == "__main__":
    import time
    import random
    import sys
    
    a = ObjectPool(['a', 'bb', 'ccc'])
    print a.upper()
    #a.join()
    
    def delayed_print(i):
        time.sleep(random.random()*3)
        if random.randint(0,2) == 0:
            raise RuntimeError#('hello')
        print i
        return i
    n = 10
    pool = FuturePool(n)
    for i in xrange(n):
        pool.add_task(delayed_print, i)
    results = pool.wait()
    print results
    for i,result in enumerate(results):
        if isinstance(result, RuntimeError):
            print "Error caught in task", i
        results[i] = i
    assert( results == range(n) )
    
    #sys.exit(0)
    
    #class C(object):
    #    def __init__(self):
    #        self.vals = [set() for _ in xrange(3)]
    #        self.pool = FuturePool(len(self.vals))
        #def foo(self):
        #    self.pool.
    class C(object):
        def __init__(self, i):
            self.i = i
        def foo(self, j):
            return j*self.i
    objs = [C(i) for i in xrange(3)]
    obj_pool = ObjectPool(objs)
    print obj_pool.foo(2)
    print obj_pool.i
    
    obj_pool.i = [7,8,9]
    print obj_pool.i
    print '***', obj_pool.i.real
    
    #print [x for x in results]
    #print [x for x in obj_pool.i]
    obj_pool.i = [-i for i in xrange(len(obj_pool))]
    print obj_pool.foo(2)
    print len(obj_pool)
    #print list(obj_pool)
    #print [x for x in obj_pool]
    #print type(obj_pool)
    #print type(list(obj_pool))
    
    #self.socks = ObjectPool([socket() for _ in xrange(n)])
    #for sock in self.socks:
    #	sock.connect(addr)
    #self.socks.send_multipart([hdr, data])
    #rets = self.socks.recv_json()
    
    a = ObjectPool(['a', 'bb', 'ccc'])
    print a.upper()
    print a.isupper()
    #a.name = ['A', 'B', 'C']
    #print a.name
    objs2 = ObjectPool([1, 2, 3])
    print objs2
    print objs2.real
    print objs2[1]
    #objs2.imag = [-1, -2, -3]
    #print objs2
    #print objs2.imag
    #objs2.imag = [-1, -2, -3]
    
    print "All tests PASSED"
