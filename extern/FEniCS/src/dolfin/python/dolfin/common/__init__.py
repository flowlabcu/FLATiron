import dolfin.cpp as cpp


def __iadd__(self, incr):
    self._increment(incr)
    return self


cpp.log.Progress.__iadd__ = __iadd__
del __iadd__
