from .base import BaseRootFinder

class RegulaFalsiSolver(BaseRootFinder):
    def __init__(self, f, a, b, tol = 1e-6, max_iter = 100, method = "Regula-Falsi Method"):
        super().__init__(f, tol, max_iter)
        self.a = a
        self.b = b
        self.method = method
    
    def solve(self):
        a, b = self.a, self.b

        for i in range(self.max_iter):
            fa, fb = self.f(a), self.f(b)

            c = (a * fb - b * fa) / (fb - fa)
            fc = self.f(c)

            self.history.append(c)
            self.errors.append(abs(fc))

            self.log_step(iter = i, a = a, b = b, c = c, fc = fc)

            if abs(fc) < self.tol:
                return c
            
            if fa * fb < 0 :
                b = c
            else:
                a = c
        
        return c