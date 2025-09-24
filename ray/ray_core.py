import ray


@ray.remote
def ray_core_f(x):
    return x*x

@ray.remote
class Counter:
    def __init__(self):
        self.n=0

    def inc(self):
        self.n+=1

    def read(self):
        return self.n


def ray_core_func():
    print("=================== this ray_core_func demo====================")
    futures = [ray_core_f.remote(i) for i in range(4)] #创建四个并行的f task
    print(ray.get(futures))
    return 


def ray_core_class():
    print("=================== this ray_core_class demo====================")
    counters=[Counter.remote() for i in range(4)]
    for i in range(4):
        counters[i].inc.remote()
    futures = [c.read.remote() for c in counters[0:-2]] # c.read.remote()返回的是objectRef,只有ray.get(futures)后才会返回真正的值
    print(ray.get(futures))

if __name__ == '__main__':
    ray.init() #创建ray instance
    ray_core_func()
    ray_core_class()
    ray.shutdown() #关闭ray instance(本地集群,释放本地的GPU和CPU资源)
    '''
    上述代码可以改为:
    with ray.init() as ray_instance:
        ray_core_func()
        ray_core_class()
    #离开该with代码块后,ray_instance就会自动关闭
    '''
