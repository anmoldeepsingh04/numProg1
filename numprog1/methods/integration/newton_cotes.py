import numpy as np
from numprog1.core.integrator import BaseIntegrator


class TrapezoidalRule(BaseIntegrator):
    """
    Trapezoidal Rule for numerical integration.
    
    Approximates the integral of a function using linear interpolation
    between function values at equally spaced points.
    """
    
    def __init__(self, f, a, b, n=1, tol=1e-6, max_iter=100):
        """
        Initialize the Trapezoidal Rule integrator.
        
        Args:
            f: Function to integrate
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of subintervals (default: 1)
            tol: Tolerance for convergence (default: 1e-6)
            max_iter: Maximum iterations for adaptive refinement (default: 100)
        """
        super().__init__(f, a, b, tol, max_iter)
        self.n = n
        self.method = "Trapezoidal Rule"
    
    def integrate(self):
        """
        Perform integration using the Trapezoidal Rule.
        
        Returns:
            float: Approximation of the integral
        """
        h = (self.b - self.a) / self.n
        x = np.linspace(self.a, self.b, self.n + 1)
        y = self.f(x)
        
        # Trapezoidal rule formula: h/2 * (y0 + 2*y1 + 2*y2 + ... + 2*yn-1 + yn)
        integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
        
        self._history.append(integral)
        # Simple error estimate - for demo purposes
        self._errors.append(0.0)
        self._converged = True
        
        return integral
    
    def integrate_iter(self):
        """
        Generator that yields intermediate results for adaptive refinement.
        
        Yields:
            tuple: (iteration, estimate)
        """
        for i in range(self.max_iter):
            n_current = 2 ** i  # Double intervals each iteration
            h = (self.b - self.a) / n_current
            x = np.linspace(self.a, self.b, n_current + 1)
            y = self.f(x)
            
            integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
            
            self._history.append(integral)
            
            if i > 0:
                error = abs(self._history[-2] - self._history[-1])
                self._errors.append(error)
                if error < self.tol:
                    self._converged = True
                    yield i, integral
                    break
            else:
                self._errors.append(0)
            
            yield i, integral
            
        if not self._converged:
            self._converged = False


class SimpsonsRule(BaseIntegrator):
    """
    Simpson's 1/3 Rule for numerical integration.
    
    Approximates the integral of a function using quadratic interpolation
    between function values at equally spaced points.
    Requires an even number of subintervals.
    """
    
    def __init__(self, f, a, b, n=2, tol=1e-6, max_iter=100):
        """
        Initialize Simpson's Rule integrator.
        
        Args:
            f: Function to integrate
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of subintervals (must be even, default: 2)
            tol: Tolerance for convergence (default: 1e-6)
            max_iter: Maximum iterations for adaptive refinement (default: 100)
        """
        super().__init__(f, a, b, tol, max_iter)
        # Ensure n is even
        self.n = n if n % 2 == 0 else n + 1
        self.method = "Simpson's 1/3 Rule"
    
    def integrate(self):
        """
        Perform integration using Simpson's 1/3 Rule.
        
        Returns:
            float: Approximation of the integral
        """
        h = (self.b - self.a) / self.n
        x = np.linspace(self.a, self.b, self.n + 1)
        y = self.f(x)
        
        # Simpson's 1/3 rule formula: h/3 * (y0 + 4*y1 + 2*y2 + 4*y3 + ... + 2*yn-2 + 4*yn-1 + yn)
        integral = (h / 3) * (
            y[0] + y[-1] + 
            4 * np.sum(y[1:-1:2]) +  # Odd indices
            2 * np.sum(y[2:-2:2])    # Even indices (excluding first and last)
        )
        
        self._history.append(integral)
        self._errors.append(0)  # Simplified error estimate
        self._converged = True
        
        return integral
    
    def integrate_iter(self):
        """
        Generator that yields intermediate results for adaptive refinement.
        
        Yields:
            tuple: (iteration, estimate)
        """
        for i in range(self.max_iter):
            n_current = 2 * (2 ** i)  # Ensure even number of intervals
            h = (self.b - self.a) / n_current
            x = np.linspace(self.a, self.b, n_current + 1)
            y = self.f(x)
            
            integral = (h / 3) * (
                y[0] + y[-1] + 
                4 * np.sum(y[1:-1:2]) +  # Odd indices
                2 * np.sum(y[2:-2:2])    # Even indices
            )
            
            self._history.append(integral)
            
            if i > 0:
                error = abs(self._history[-2] - self._history[-1])
                self._errors.append(error)
                if error < self.tol:
                    self._converged = True
                    yield i, integral
                    break
            else:
                self._errors.append(0)
            
            yield i, integral
            
        if not self._converged:
            self._converged = False


class SimpsonsThreeEighthsRule(BaseIntegrator):
    """
    Simpson's 3/8 Rule for numerical integration.
    
    Approximates the integral of a function using cubic interpolation
    between function values at equally spaced points.
    Requires the number of subintervals to be a multiple of 3.
    """
    
    def __init__(self, f, a, b, n=3, tol=1e-6, max_iter=100):
        """
        Initialize Simpson's 3/8 Rule integrator.
        
        Args:
            f: Function to integrate
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of subintervals (must be multiple of 3, default: 3)
            tol: Tolerance for convergence (default: 1e-6)
            max_iter: Maximum iterations for adaptive refinement (default: 100)
        """
        super().__init__(f, a, b, tol, max_iter)
        # Ensure n is multiple of 3
        remainder = n % 3
        if remainder != 0:
            self.n = n + (3 - remainder)
        else:
            self.n = n
        self.method = "Simpson's 3/8 Rule"
    
    def integrate(self):
        """
        Perform integration using Simpson's 3/8 Rule.
        
        Returns:
            float: Approximation of the integral
        """
        h = (self.b - self.a) / self.n
        x = np.linspace(self.a, self.b, self.n + 1)
        y = self.f(x)
        
        # Simpson's 3/8 rule formula: 3h/8 * (y0 + 3*y1 + 3*y2 + 2*y3 + 3*y4 + 3*y5 + 2*y6 + ... + yn)
        # Pattern: 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1
        coefficients = np.ones(self.n + 1)
        coefficients[1:-1:3] = 3  # Every 3rd point starting from index 1
        coefficients[2:-1:3] = 3  # Every 3rd point starting from index 2
        coefficients[3:-1:3] = 2  # Every 3rd point starting from index 3
        
        integral = (3 * h / 8) * np.sum(coefficients * y)
        
        self._history.append(integral)
        self._errors.append(0)  # Simplified error estimate
        self._converged = True
        
        return integral
    
    def integrate_iter(self):
        """
        Generator that yields intermediate results for adaptive refinement.
        
        Yields:
            tuple: (iteration, estimate)
        """
        for i in range(self.max_iter):
            # Ensure n_current is multiple of 3
            base_n = 3 * (2 ** i)
            n_current = base_n
            h = (self.b - self.a) / n_current
            x = np.linspace(self.a, self.b, n_current + 1)
            y = self.f(x)
            
            # Simpson's 3/8 rule coefficients
            coefficients = np.ones(n_current + 1)
            coefficients[1:-1:3] = 3
            coefficients[2:-1:3] = 3
            coefficients[3:-1:3] = 2
            
            integral = (3 * h / 8) * np.sum(coefficients * y)
            
            self._history.append(integral)
            
            if i > 0:
                error = abs(self._history[-2] - self._history[-1])
                self._errors.append(error)
                if error < self.tol:
                    self._converged = True
                    yield i, integral
                    break
            else:
                self._errors.append(0)
            
            yield i, integral
            
        if not self._converged:
            self._converged = False


class BoolesRule(BaseIntegrator):
    """
    Boole's Rule for numerical integration.
    
    Approximates the integral of a function using 4th degree polynomial
    interpolation between function values at equally spaced points.
    Requires the number of subintervals to be a multiple of 4.
    """
    
    def __init__(self, f, a, b, n=4, tol=1e-6, max_iter=100):
        """
        Initialize Boole's Rule integrator.
        
        Args:
            f: Function to integrate
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of subintervals (must be multiple of 4, default: 4)
            tol: Tolerance for convergence (default: 1e-6)
            max_iter: Maximum iterations for adaptive refinement (default: 100)
        """
        super().__init__(f, a, b, tol, max_iter)
        # Ensure n is multiple of 4
        remainder = n % 4
        if remainder != 0:
            self.n = n + (4 - remainder)
        else:
            self.n = n
        self.method = "Boole's Rule"
    
    def integrate(self):
        """
        Perform integration using Boole's Rule.
        
        Returns:
            float: Approximation of the integral
        """
        h = (self.b - self.a) / self.n
        x = np.linspace(self.a, self.b, self.n + 1)
        y = self.f(x)
        
        # Boole's rule formula: 2h/45 * (7*y0 + 32*y1 + 12*y2 + 32*y3 + 7*y4 + ...)
        # Pattern repeats every 5 points: 7, 32, 12, 32, 7
        coefficients = np.zeros(self.n + 1)
        for i in range(0, self.n + 1, 4):
            if i + 4 <= self.n:
                coefficients[i:i+5] = [7, 32, 12, 32, 7]
        
        integral = (2 * h / 45) * np.sum(coefficients * y)
        
        self._history.append(integral)
        self._errors.append(0)  # Simplified error estimate
        self._converged = True
        
        return integral
    
    def integrate_iter(self):
        """
        Generator that yields intermediate results for adaptive refinement.
        
        Yields:
            tuple: (iteration, estimate)
        """
        for i in range(self.max_iter):
            # Ensure n_current is multiple of 4
            base_n = 4 * (2 ** i)
            n_current = base_n
            h = (self.b - self.a) / n_current
            x = np.linspace(self.a, self.b, n_current + 1)
            y = self.f(x)
            
            # Boole's rule coefficients
            coefficients = np.zeros(n_current + 1)
            for j in range(0, n_current + 1, 4):
                if j + 4 <= n_current:
                    coefficients[j:j+5] = [7, 32, 12, 32, 7]
            
            integral = (2 * h / 45) * np.sum(coefficients * y)
            
            self._history.append(integral)
            
            if i > 0:
                error = abs(self._history[-2] - self._history[-1])
                self._errors.append(error)
                if error < self.tol:
                    self._converged = True
                    yield i, integral
                    break
            else:
                self._errors.append(0)
            
            yield i, integral
            
        if not self._converged:
            self._converged = False