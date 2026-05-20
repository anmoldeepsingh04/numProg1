"""
Tests for integration methods in numprog1.
"""

import numpy as np
import pytest
from numprog1.methods.integration import (
    TrapezoidalRule,
    SimpsonsRule,
    SimpsonsThreeEighthsRule,
    BoolesRule,
    GaussLegendreQuadrature
)


def test_polynomial_degree():
    """Test that integration methods integrate polynomials up to their degree of precision."""
    # Test function: f(x) = x^3 (integral from 0 to 2 = 4)
    def f(x):
        return x**3
    
    # Trapezoidal rule is exact for linear functions (degree 1)
    def linear(x):
        return 2*x + 1  # Integral from 0 to 2 = (x^2 + x)|0 to 2 = 4 + 2 = 6
    
    # Test Trapezoidal Rule with linear function
    trap = TrapezoidalRule(linear, 0, 2, n=1)
    result = trap.integrate()
    expected = 6.0
    assert abs(result - expected) < 1e-10  # Should be exact for linear with 1 segment
    
    # Test Simpson's Rule (should be exact for cubics)
    def cubic(x):
        return x**3  # Integral from 0 to 2 = 4
    
    simpson = SimpsonsRule(cubic, 0, 2, n=2)
    result = simpson.integrate()
    expected = 4.0
    assert abs(result - expected) < 1e-10
    
    # Test Gauss-Legendre (should be very accurate)
    gauss = GaussLegendreQuadrature(cubic, 0, 2, n=2)
    result = gauss.integrate()
    expected = 4.0
    assert abs(result - expected) < 1e-10


def test_trapezoidal_rule():
    """Test Trapezoidal Rule integration."""
    def f(x):
        return x**2
    
    # Integral of x^2 from 0 to 2 = 8/3 ≈ 2.666...
    trap = TrapezoidalRule(f, 0, 2, n=4)
    result = trap.integrate()
    expected = 8/3
    # With n=4, trapezoidal rule error for x^2 is about 0.083
    assert abs(result - expected) < 0.1
    
    # Test with single segment for comparison
    trap_single = TrapezoidalRule(f, 0, 2, n=1)
    result_single = trap_single.integrate()
    # With one segment: (2-0)/2 * (f(0) + f(2)) = 1 * (0 + 4) = 4
    expected_single = 4.0
    assert abs(result_single - expected_single) < 1e-10


def test_simpsons_rule():
    """Test Simpson's Rule integration."""
    def f(x):
        return x**3
    
    # Integral of x^3 from 0 to 2 = 4
    simpson = SimpsonsRule(f, 0, 2, n=2)
    result = simpson.integrate()
    expected = 4.0
    assert abs(result - expected) < 1e-10
    
    # Test with more segments
    simpson_fine = SimpsonsRule(f, 0, 2, n=4)
    result_fine = simpson_fine.integrate()
    assert abs(result_fine - expected) < 1e-10


def test_gauss_legendre():
    """Test Gauss-Legendre Quadrature."""
    def f(x):
        return x**2
    
    # Integral of x^2 from 0 to 2 = 8/3
    gauss = GaussLegendreQuadrature(f, 0, 2, n=3)
    result = gauss.integrate()
    expected = 8/3
    assert abs(result - expected) < 1e-10


def test_integration_history():
    """Test that integration methods track history correctly."""
    def f(x):
        return x**2
    
    trap = TrapezoidalRule(f, 0, 2, n=2)
    result = trap.integrate()
    
    history = trap.get_history()
    assert len(history) == 1
    assert abs(history[0] - result) < 1e-10


def test_invalid_interval():
    """Test that methods handle invalid intervals gracefully."""
    def f(x):
        return x**2
    
    # Test with a > b (should give negative of integral from b to a)
    trap = TrapezoidalRule(f, 2, 0, n=2)
    result = trap.integrate()
    # Manual calculation: h = (0-2)/2 = -1, points = [2, 1, 0]
    # f(2)=4, f(1)=1, f(0)=0
    # Integral = h/2 * (f0 + 2*f1 + f2) = -1/2 * (4 + 2*1 + 0) = -1/2 * 6 = -3.0
    expected = -3.0  # Correct result for n=2 trapezoidal with reversed limits
    assert abs(result - expected) < 1e-10