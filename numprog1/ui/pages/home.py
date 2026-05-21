import streamlit as st


def app():
    """Home page with introduction."""
    st.title("Numerical Programming 1 Tool")
    
    st.markdown("""
    ## Welcome!
    
    This tool helps you understand numerical methods through interactive visualization.
    
    ### Features
    - Root finding methods (Bisection, Secant, Regula Falsi)
    - Numerical integration (Trapezoidal, Simpson's, Gauss-Legendre)
    - Step-by-step visualization
    - Side-by-side method comparison
    
    ### Getting Started
    1. Select a method category from the sidebar
    2. Enter your function (e.g., `sin(x) - 0.5`)
    3. Set the interval and parameters
    4. Click "Solve" to see the visualization
    
    ### Available Functions
    - Trigonometric: `sin`, `cos`, `tan`
    - Exponential: `exp`
    - Logarithms: `log`, `log10`
    - Constants: `pi`, `e`
    """)
    
    st.info("Select a method from the sidebar to begin!")