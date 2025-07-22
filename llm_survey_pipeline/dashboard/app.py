"""Dashboard application for LLM Survey Pipeline"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import asyncio
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llm_survey_pipeline.config import MODEL_CONFIG, prompt_templates, all_scales, MFQ_FOUNDATIONS
from llm_survey_pipeline.main import SurveyPipeline


def create_app():
    st.set_page_config(
        page_title="LLM Survey Pipeline Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LLM Survey Pipeline Dashboard")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Models")
        available_models = list(MODEL_CONFIG.keys())
        selected_models = st.multiselect(
            "Select models to test:",
            available_models,
            default=["OpenAI", "Claude", "Grok"]
        )
        
        # Scale selection
        st.subheader("Scales")
        scale_names = list(set(q["scale_name"] for scale_list in all_scales for q in scale_list))
        selected_scales = st.multiselect(
            "Select scales to run:",
            scale_names,
            default=["RWA", "LWA"]
        )
        
        # Prompt style selection
        st.subheader("Prompt Styles")
        selected_prompts = st.multiselect(
            "Select prompt styles:",
            list(prompt_templates.keys()),
            default=["minimal", "extreme_liberal", "extreme_conservitive"]
        )
        
        # Other parameters
        st.subheader("Parameters")
        num_runs = st.number_input("Number of runs per question", min_value=1, max_value=10, value=1)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        
        # Run button
        run_survey = st.button("🚀 Run Survey", type="primary", use_container_width=True)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Analysis", "🔍 Raw Data", "⚙️ Settings"])
    
    with tab1:
        st.header("Survey Results")
        
        if run_survey:
            with st.spinner("Running survey... This may take a few minutes."):
                pipeline = SurveyPipeline(
                    scales_to_run=selected_scales,
                    prompt_styles_to_run=selected_prompts,
                    models_to_run=selected_models,
                    num_calls_test=num_runs,
                    temperature=temperature
                )
                
                # Run the survey
                df_results = asyncio.run(pipeline.run_survey())
                st.success(f"✅ Survey completed! Processed {len(df_results)} responses.")
                
                # Store results in session state
                st.session_state['df_results'] = df_results
        
        # Display results if available
        if 'df_results' in st.session_state:
            df = st.session_state['df_results']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Responses", len(df))
            with col2:
                resp_rate = df['numeric_score'].notna().sum() / len(df) * 100
                st.metric("Response Rate", f"{resp_rate:.1f}%")
            with col3:
                st.metric("Models Tested", df['model_name'].nunique())
            
            # Show average scores by model and scale
            st.subheader("Average Scores by Model and Scale")
            avg_scores = df.groupby(['model_name', 'scale_name'])['scored_value'].mean().reset_index()
            fig = px.bar(avg_scores, x='model_name', y='scored_value', color='scale_name',
                         title="Average Scores by Model and Scale",
                         labels={'scored_value': 'Average Score', 'model_name': 'Model'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Analysis")
        
        if 'df_results' in st.session_state:
            df = st.session_state['df_results']
            
            # Analysis by prompt style
            st.subheader("Scores by Prompt Style")
            prompt_analysis = df.groupby(['prompt_style', 'model_name'])['scored_value'].mean().reset_index()
            fig = px.line(prompt_analysis, x='prompt_style', y='scored_value', color='model_name',
                          title="Model Responses by Prompt Style",
                          labels={'scored_value': 'Average Score', 'prompt_style': 'Prompt Style'})
            st.plotly_chart(fig, use_container_width=True)
            
            # MFQ Foundation Analysis if available
            if 'MFQ' in df['scale_name'].unique():
                st.subheader("MFQ Foundation Scores")
                mfq_data = df[df['scale_name'] == 'MFQ']
                
                # Calculate foundation scores
                foundation_scores = []
                for model in mfq_data['model_name'].unique():
                    for prompt in mfq_data['prompt_style'].unique():
                        for foundation, questions in MFQ_FOUNDATIONS.items():
                            mask = (mfq_data['model_name'] == model) & \
                                   (mfq_data['prompt_style'] == prompt) & \
                                   (mfq_data['question_id'].isin(questions))
                            score = mfq_data[mask]['scored_value'].mean()
                            foundation_scores.append({
                                'model': model,
                                'prompt': prompt,
                                'foundation': foundation,
                                'score': score
                            })
                
                foundation_df = pd.DataFrame(foundation_scores)
                fig = px.bar(foundation_df, x='foundation', y='score', color='model',
                             facet_col='prompt', title="MFQ Foundation Scores by Model and Prompt")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a survey first to see analysis results.")
    
    with tab3:
        st.header("Raw Data")
        
        if 'df_results' in st.session_state:
            df = st.session_state['df_results']
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                filter_model = st.multiselect("Filter by Model:", df['model_name'].unique())
            with col2:
                filter_scale = st.multiselect("Filter by Scale:", df['scale_name'].unique())
            
            # Apply filters
            filtered_df = df.copy()
            if filter_model:
                filtered_df = filtered_df[filtered_df['model_name'].isin(filter_model)]
            if filter_scale:
                filtered_df = filtered_df[filtered_df['scale_name'].isin(filter_scale)]
            
            # Display data
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="survey_results.csv",
                mime="text/csv"
            )
        else:
            st.info("Run a survey first to see raw data.")
    
    with tab4:
        st.header("Settings")
        
        # Display current configuration
        st.subheader("Current Model Configuration")
        for model_name, config in MODEL_CONFIG.items():
            with st.expander(f"{model_name} Configuration"):
                st.json(config)
        
        st.subheader("Prompt Templates")
        for prompt_name, prompt_text in prompt_templates.items():
            with st.expander(f"{prompt_name}"):
                st.text(prompt_text)
        
        # Load existing results
        st.subheader("Load Existing Results")
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['df_results'] = df
            st.success("File loaded successfully!")
            st.rerun()


if __name__ == "__main__":
    create_app()