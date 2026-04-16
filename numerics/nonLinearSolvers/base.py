class BaseRootFinder:
    
    def __init__(self, f, tol = 1e-6, max_iter = 100):
        self.f = f
        self.tol = tol
        self.max_iter = max_iter
        self.history = [] # stores root approximations
        self.errors = [] # stores |f(x)|
        self.steps = [] # stores iteration information
    
    def log_step(self, **kwargs):
        self.steps.append(kwargs)

    def get_steps(self):
        return self.steps
    
    def get_history(self):
        return self.history
    
    def solve(self):
        raise NotImplementedError