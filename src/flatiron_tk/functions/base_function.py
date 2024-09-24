from ..info.messages import import_fenics
fe = import_fenics()

class BaseFunction():
    pass

if fe:

    class BaseFunction(fe.UserExpression):

        def __init__(self, dim, func_eval, *func_eval_inputs, **kwargs):

            self.dim = dim
            self.func_eval = func_eval
            self.func_eval_inputs = func_eval_inputs
            super().__init__(**kwargs)

        def eval(self, value, x):
            if self.dim == 1:
                value[0] = self.func_eval(x, *self.func_eval_inputs)
            else:
                # Here, I need to assign the values list per dimension
                # because I need the value list to be the correct input size
                return_values = self.func_eval(x, *self.func_eval_inputs)
                for d in range(self.dim):
                    value[d] = return_values[d]
            return value

        def value_shape(self):
            if self.dim == 1:
                return ()
            else:
                return (self.dim, )

