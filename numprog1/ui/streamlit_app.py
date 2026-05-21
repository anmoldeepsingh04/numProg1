import streamlit as st
from numprog1.ui.components.sidebar import render_sidebar
from numprog1.ui.components.plots import render_visualization
from numprog1.ui.components.results import render_results


def main():
    st.set_page_config(
        page_title="Numerical Programming 1 Tool",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Numerical Programming 1 Tool")
    st.markdown("""
    Educational tool for TUM's Numerical Programming 1 course.
    Visualize and solve numerical problems step-by-step.
    """)
    
    # Get user inputs
    config = render_sidebar()
    
    # Main content area
    if config.get('submitted'):
        with st.spinner("Computing..."):
            render_visualization(config)
            render_results(config)


if __name__ == "__main__":
    main()