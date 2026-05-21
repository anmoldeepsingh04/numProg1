import streamlit as st

try:
    from ..methods.nonlinear import BisectionSolver, SecantSolver, RegulaFalsiSolver
except ImportError:
    from numprog1.methods.nonlinear import BisectionSolver, SecantSolver, RegulaFalsiSolver


def render_results(config):
    """Render results display component."""
    st.subheader("Results")
    
    if config.get('function') is None:
        st.warning("Please enter a valid function")
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
        result_obj = solver.get_result()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Root", f"{result:.10f}")
        with col2:
            st.metric("Iterations", len(result_obj.history))
        with col3:
            st.metric("Converged", "Yes" if result_obj.converged else "No")
        with col4:
            st.metric("Error", f"{result_obj.errors[-1] if result_obj.errors else 0:.2e}")
        
        if result_obj.steps:
            st.markdown("### Step-by-Step Details")
            import pandas as pd
            df = pd.DataFrame(result_obj.steps)
            st.dataframe(df, use_container_width=True, height=300)
        
        if config.get('compare_method') and config['compare_method'] in method_map:
            st.markdown(f"### Comparison: {config['compare_method']}")
            SolverClass2 = method_map[config['compare_method']]
            solver2 = SolverClass2(
                config['function'],
                config['a'],
                config['b'],
                tol=config['tolerance'],
                max_iter=config['max_iter']
            )
            result2 = solver2.solve()
            result2_obj = solver2.get_result()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{config['method']}**: {result:.10f} ({len(result_obj.history)} iterations)")
            with col2:
                st.write(f"**{config['compare_method']}**: {result2:.10f} ({len(result2_obj.history)} iterations)")
            
    except Exception as e:
        st.error(f"Computation error: {e}")