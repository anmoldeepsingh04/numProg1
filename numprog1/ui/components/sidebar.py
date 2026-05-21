import streamlit as st
import sympy as sp
import numpy as np


def _parse_function(expr_str):
    """Parse mathematical expression with symbolic and numerical capabilities."""
    x = sp.Symbol('x')
    expr = sp.sympify(expr_str)
    
    func = sp.lambdify(
        x, expr,
        ['numpy', {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'sqrt': np.sqrt, 'abs': np.abs,
            'floor': np.floor, 'ceil': np.ceil,
            'pi': np.pi, 'e': np.e
        }]
    )
    
    derivative = sp.lambdify(x, sp.diff(expr, x), 'numpy')
    
    return {
        'symbolic': expr,
        'function': func,
        'derivative': derivative
    }


def render_sidebar():
    """Render sidebar with all input controls."""
    st.sidebar.header("Problem Configuration")
    
    method_category = st.sidebar.selectbox(
        "Method Category",
        ["Nonlinear Solvers", "Integration", "Linear Solvers", "Eigenvalue"],
        key="category"
    )
    
    if method_category == "Nonlinear Solvers":
        method_options = ["Bisection", "Secant", "Regula Falsi"]
        if st.sidebar.checkbox("Enable Newton-Raphson (requires derivative)", False):
            method_options.append("Newton-Raphson")
    elif method_category == "Integration":
        method_options = ["Trapezoidal", "Simpson's 1/3", "Simpson's 3/8", "Boole's", "Gauss-Legendre"]
    else:
        method_options = ["Not yet implemented"]
    
    method = st.sidebar.selectbox("Method", method_options, key="method")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Function")
    func_str = st.sidebar.text_input(
        "f(x) =", 
        value="x**3 - x - 2",
        help="Use standard math notation. Available: sin, cos, tan, exp, log, sqrt, pi, e"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("Lower bound (a)", value=1.0, key="a")
    with col2:
        b = st.number_input("Upper bound (b)", value=2.0, key="b")
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        tolerance = st.number_input(
            "Tolerance", 
            value=1e-6, 
            format="%.2e",
            key="tol"
        )
    with col4:
        max_iter = st.number_input(
            "Max iterations", 
            value=100,
            min_value=1,
            key="max_iter"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced Options")
    
    animation_speed = st.sidebar.slider(
        "Animation delay (ms)",
        min_value=100,
        max_value=2000,
        value=500,
        key="speed"
    )
    
    compare_methods = st.sidebar.checkbox(
        "Compare with another method",
        key="compare"
    )
    
    if compare_methods:
        compare_method = st.sidebar.selectbox(
            "Comparison Method",
            [m for m in method_options if m != method],
            key="compare_method"
        )
    else:
        compare_method = None
    
    submit = st.sidebar.button("Solve", type="primary")
    
    try:
        func_data = _parse_function(func_str)
    except Exception as e:
        st.sidebar.error(f"Invalid function: {e}")
        func_data = None
    
    return {
        'category': method_category,
        'method': method,
        'compare_method': compare_method,
        'function_str': func_str,
        'function': func_data['function'] if func_data else None,
        'derivative': func_data['derivative'] if func_data else None,
        'a': a,
        'b': b,
        'tolerance': tolerance,
        'max_iter': max_iter,
        'animation_speed': animation_speed,
        'submitted': submit
    }