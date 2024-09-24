from ..info.messages import import_fenics
fe = import_fenics()

class IndicatorFieldScalar():
    pass

if fe:

    class IndicatorFieldScalar(fe.UserExpression):

        '''
        Define a heaviside-type field where
        I = 1 if indicator_domain(x):
        and I = 0 otherwise
        '''

        def __init__(self, indicator_domain, **kwargs):
            self.indicator_domain = indicator_domain
            super().__init__(**kwargs)

        def eval(self, value, x):
            if self.indicator_domain(x):
                value[0] = 1
            else:
                value[0] = 0
            return value

        def value_shape(self):
            return ()

