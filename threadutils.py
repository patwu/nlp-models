import threading
from Queue import Queue
from threading import Thread

import traceback

class Worker(Thread):
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()
    
    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try: func(*args, **kargs)
            except Exception, e: traceback.print_exc()
            self.tasks.task_done()

class Counter(object):
    def __init__(self):
        self.lock=threading.Lock()
        self.sum=[0.]*1000
        self.cnt=[0]*1000
    
    def add(self,result,bucket=0):
        self.lock.acquire()
        self.sum[bucket]+=result
        self.cnt[bucket]+=1
        self.lock.release()

    def get(self,bucket=0):
        self.lock.acquire()
        ret=self.sum[bucket],self.cnt[bucket]
        self.lock.release()
        return ret 

class ThreadPool(object):
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads): Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        self.tasks.join()

class ThreadWriter(Thread):
    def __init__(self, filename):
        Thread.__init__(self)
        self.daemon = True
        self.queue=Queue(10)
        self.filename=filename
        self.start()
    
    def append(self,line):
        self.queue.put(line)

    def run(self):
        with open(self.filename, 'a') as f:
            while True:
                line=self.queue.get()
                if line=='TERMINAL':
                    break
                f.write(line+'\n')
                f.flush()
            close(f)

    def wait_completion(self):
        self.join()



