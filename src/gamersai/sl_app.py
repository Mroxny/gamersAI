import streamlit as st
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.project import configure_project
from pathlib import Path
import os

def load_kedro_session():
    bootstrap_project(Path.cwd().parent.parent)
    session = KedroSession.create()
    return session
    

session = load_kedro_session()

session.load_context


pipelines = ['dt_pipeline'
            ,'elasticnet_pipeline'
            ,'gradient_boosting_pipeline'
            ,'knn_pipeline'
            ,'pipeline_autogluon'
            ,'random_forest_pipeline'
            ,'xgboost_pipeline',]

st.sidebar.title("Pipeline Selector")
selected_pipeline = st.sidebar.selectbox("Choose a pipeline to run:", pipelines)

if st.sidebar.button("Run Pipeline"):
    with st.spinner(f"Running pipeline '{selected_pipeline}'..."):
        session.run(pipeline_name=selected_pipeline)
        st.success(f"Pipeline '{selected_pipeline}' completed successfully!")