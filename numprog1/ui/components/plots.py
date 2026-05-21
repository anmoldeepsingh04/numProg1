import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

try:
    from ..methods.nonlinear import BisectionSolver, SecantSolver, RegulaFalsiSolver
except ImportError:
    from numprog1.methods.nonlinear import BisectionSolver, SecantSolver, RegulaFalsiSolver


def render_visualization(config):
    """Render live visualization components."""
    st.subheader("Visualization")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Function Plot")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        
        x_vals = np.linspace(config['a'] - 0.5, config['b'] + 0.5, 200)
        try:
            y_vals = config['function'](x_vals)
            ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        except:
            ax1.text(0.5, 0.5, "Error plotting function", ha='center', va='center')
        
        st.pyplot(fig1)
    
    with c2:
        st.markdown("### Iteration History")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Root Estimate')
        ax2.set_title('Convergence')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
    
    return fig1, fig2


def render_results(config):
    """Render results table and explanation."""
    st.subheader("Results")
    
    if config['function'] is None:
        st.error("Cannot compute with invalid function")
        return
    
    from numprog1.methods.nonlinear import BisectionSolver, SecantSolver, RegulaFalsiSolver
    
    method_map = {
        "Bisection": BisectionSolver,
        "Secant": SecantSolver,
        "Regula Falsi": RegulaFalsiSolver
    }
    
    try:
        SolverClass = method_map[config['method']]
        solver = SolverClass(
            config['function'],
            config['a'],
            config['b'],
            tol=config['tolerance'],
            max_iter=config['max_iter']
        )
        result = solver.solve()
        result_data = solver.get_result()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Root", f"{result:.10f}")
        with col2:
            st.metric("Iterations", len(result_data.history))
        with col3:
            st.metric("Converged", "Yes" if result_data.converged else "No")
        
        st.markdown("### Step-by-Step Table")
        if result_data.steps:
            import pandas as pd
            df = pd.DataFrame(result_data.steps)
            st.dataframe(df, use_container_width=True)
        
        if config['compare_method'] and config['compare_method'] in method_map:
            st.markdown(f"### Comparison with {config['compare_method']}")
            SolverClass2 = method_map[config['compare_method']]
            solver2 = SolverClass2(
                config['function'],
                config['a'],
                config['b'],
                tol=config['tolerance'],
                max_iter=config['max_iter']
            )
            result2 = solver2.solve()
            st.info(f"{config['compare_method']} result: {result2:.10f}")
            
    except Exception as e:
        st.error(f"Error during computation: {e}")