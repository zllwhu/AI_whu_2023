import time


def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        with open("results.txt", "a+") as file:
            file.write('耗时：{}秒'.format(e_time - s_time))
        return res

    return inner
