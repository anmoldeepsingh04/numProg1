import numpy as np
from numprog1.core.integrator import BaseIntegrator


class GaussLegendreQuadrature(BaseIntegrator):
    """
    Gauss-Legendre Quadrature for numerical integration.
    
    Approximates the integral of a function using optimal sampling points
    (roots of Legendre polynomials) and weights for maximum precision.
    """
    
    def __init__(self, f, a, b, n=5, tol=1e-6, max_iter=100):
        """
        Initialize Gauss-Legendre Quadrature integrator.
        
        Args:
            f: Function to integrate
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of sample points (degree of precision = 2n-1, default: 5)
            tol: Tolerance for convergence (default: 1e-6)
            max_iter: Maximum iterations for adaptive refinement (default: 100)
        """
        super().__init__(f, a, b, tol, max_iter)
        self.n = n
        self.method = f"Gauss-Legendre Quadrature (n={n})"
        
        # Pre-compute Legendre polynomial roots and weights for common n values
        self._legendre_data = {
            1: (np.array([0.0]), np.array([2.0])),
            2: (np.array([-1/np.sqrt(3), 1/np.sqrt(3)]), np.array([1.0, 1.0])),
            3: (np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)]), 
                np.array([5/9, 8/9, 5/9])),
            4: (np.array([-np.sqrt((3 + 2*np.sqrt(6/5))/7), 
                         -np.sqrt((3 - 2*np.sqrt(6/5))/7),
                          np.sqrt((3 - 2*np.sqrt(6/5))/7),
                           np.sqrt((3 + 2*np.sqrt(6/5))/7)]),
                np.array([(18 - np.sqrt(30))/36, (18 + np.sqrt(30))/36,
                         (18 + np.sqrt(30))/36, (18 - np.sqrt(30))/36])),
            5: (np.array([-1/3*np.sqrt(5 + 2*np.sqrt(10/7)),
                         -1/3*np.sqrt(5 - 2*np.sqrt(10/7)),
                          0.0,
                           1/3*np.sqrt(5 - 2*np.sqrt(10/7)),
                            1/3*np.sqrt(5 + 2*np.sqrt(10/7))]),
                np.array([(322 - 13*np.sqrt(70))/900, (322 + 13*np.sqrt(70))/900,
                         128/225, (322 + 13*np.sqrt(70))/900,
                         (322 - 13*np.sqrt(70))/900]))
        }
    
    def _get_legendre_data(self, n):
        """Get Legendre roots and weights for n points, computing if necessary."""
        if n in self._legendre_data:
            return self._legendre_data[n]
        
        # For higher n, use numpy's legendre function approximation
        # This is a simplified approach - in practice, you'd use numerical methods
        # to find roots of Legendre polynomials
        from numpy.polynomial.legendre import leggauss
        return leggauss(n)
    
    def integrate(self):
        """
        Perform integration using Gauss-Legendre Quadrature.
        
        Returns:
            float: Approximation of the integral
        """
        # Get Legendre roots and weights for n points
        points, weights = self._get_legendre_data(self.n)
        
        # Transform from [-1, 1] to [a, b]
        # x = (b-a)/2 * t + (a+b)/2 where t in [-1, 1]
        transformed_points = (self.b - self.a) / 2 * points + (self.a + self.b) / 2
        transformed_weights = (self.b - self.a) / 2 * weights
        
        # Evaluate function at transformed points and compute weighted sum
        integral = np.sum(transformed_weights * self.f(transformed_points))
        
        self._history.append(integral)
        self._errors.append(0)  # Gauss quadrature error estimate is complex
        self._converged = True
        
        return integral
    
    def integrate_iter(self):
        """
        Generator that yields intermediate results for increasing precision.
        
        Yields:
            tuple: (iteration, estimate)
        """
        for i in range(self.max_iter):
            # Increase number of points each iteration
            n_current = min(self.n + i, 50)  # Cap at reasonable value
            
            # Temporarily update n for this iteration
            original_n = self.n
            self.n = n_current
            
            try:
                integral = self.integrate()
                
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
            finally:
                # Restore original n
                self.n = original_n
        
        if not self._converged:
            self._converged = False