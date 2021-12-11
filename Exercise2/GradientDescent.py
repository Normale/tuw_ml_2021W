import math
import copy


class GradientDescent:
    max_it = 300
    stop_grad = 0.0001

    def __init__(self, f, params, x, y, s=0.1):
        self.s = s
        self.params = params  # [(param1, param1range, )
        self.allowed_params()
        self.it = 0
        self.f = f
        self.x = x
        self.y = y

    def solve(self):
        while self.it < self.max_it:
            self.it += 1
            print("GD Step {}".format(self.it))
            print("Params: alpha={}, l1_ratio={}".format(self.params['alpha']['value'], self.params['l1_ratio']['value']))
            cost = self.f(self.x, self.y, self.params)
            print("Cost = {}".format(cost))
            gradient = self.get_gradient()
            print("Gradient = {}".format(gradient))
            if self.stop_criterium(gradient):
                return self.params
            self.allowed_params()

            self.param_subtract(gradient)
        return self.params

    def param_subtract(self, subt: dict):
        for param, val in self.params.items():
            val['value'] = val['value'] + self.s*subt[param] # THIS SHOULD BE '-' BUT DOESNT GIVE GOOD RESULTS????
            if val['type'] == 'int':
                val['value'] = math.ceil(val['value'])
            if val['value'] > val['max']:
                val['value'] = val['max']
            if val['value'] < val['min']:
                val['value'] = val['min']

    @staticmethod
    def stop_criterium(grad):
        if max(list(map(abs, grad.values()))) < GradientDescent.stop_grad:
            return True
        return False

    def get_gradient(self) -> dict:
        grad = {}

        for param, val in self.params.items():
            value = val['value']
            plus = copy.deepcopy(self.params)
            minus = copy.deepcopy(self.params)

            param_plus = value + val['e']  # 1% above current param
            if param_plus > val['max']:
                param_plus = val['max']
            param_minus = value - val['e']  # 1% below current param
            if param_minus < val['min']:
                param_minus = val['min']

            plus[param]['value'] = param_plus
            minus[param]['value'] = param_minus

            f_plus = self.f(self.x, self.y, plus)
            f_minus = self.f(self.x, self.y, minus)
            # print("cost diff {} = {}".format(param, f_plus - f_minus))

            diff = (f_plus - f_minus) / (2 * val['e'])
            grad[param] = diff
        return grad

    def allowed_params(self):
        for name, param in self.params.items():
            if param['min'] <= param['value'] <= param['max']:
                if param['type'] == 'int' and param['value'].is_integer():
                    return True
                return True

        print(self.params)
        raise Exception("These parameters are not allowed!")
