import time
from functools import wraps

from tools.logger import Logger  # noqa


def timeit(f):

    @wraps(f)
    def wrap(*args, **kw):
        s = time.time()
        result = f(*args, **kw)
        e = time.time()
        print('--> %s(), cost %2.4f sec' % (f.__name__, e - s))
        return result

    return wrap
