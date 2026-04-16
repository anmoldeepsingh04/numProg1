from .base import BaseRootFinder

class BisectionSolver(BaseRootFinder):
    def __init__(self, f, a, b, tol = 1e-6, max_iter = 100 ):
        super().__init__(f, tol, max_iter)
        self.a = a
        self.b = b
        self.method = "Bisection Method"

    def solve(self):
        a, b = self.a, self.b

        if self.f(a) * self.f(b) >= 0:
            raise ValueError("Invalid interval: f(a), f(b) must have opposite signs.")

        for i in range(self.max_iter):
            c = (a+b)/2
            fc = self.f(c)

            self.history.append(c)
            self.errors.append(abs(fc))
            self.log_step(iter = i, a = a, b = b, c = c, fc = fc)

            if abs(fc) < self.tol:
                return c
            
            if self.f(a) * fc < 0:
                b = c
            else:
                a = c
        
        return c