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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Results", "📈 Analysis", "🔍 Raw Data", "📉 Data Explorer", "⚙️ Settings"])
    
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
        st.header("📉 Data Explorer")
        
        if 'df_results' in st.session_state:
            df = st.session_state['df_results']
            
            st.markdown("### Interactive Data Exploration")
            st.markdown("Use the controls below to create custom visualizations from your survey data.")
            
            # Column selection for axes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Violin Plot", "Heatmap"]
                )
            
            with col2:
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                y_axis = st.selectbox("Y-Axis (Value)", numeric_cols, index=numeric_cols.index('scored_value') if 'scored_value' in numeric_cols else 0)
            
            with col3:
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                x_axis = st.selectbox("X-Axis (Category)", categorical_cols, index=categorical_cols.index('model_name') if 'model_name' in categorical_cols else 0)
            
            # Additional options
            col4, col5, col6 = st.columns(3)
            
            with col4:
                color_by = st.selectbox("Color By", ["None"] + categorical_cols)
            
            with col5:
                facet_by = st.selectbox("Facet By", ["None"] + categorical_cols)
            
            with col6:
                aggregation = st.selectbox("Aggregation", ["mean", "median", "sum", "count", "std", "min", "max"])
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    show_confidence = st.checkbox("Show confidence intervals (where applicable)")
                    show_trendline = st.checkbox("Show trendline (scatter plots)")
                
                with col_adv2:
                    custom_title = st.text_input("Custom Chart Title", "")
                    export_format = st.selectbox("Export Format", ["PNG", "HTML", "SVG"])
            
            # Filters
            st.markdown("### Filters")
            filter_cols = st.columns(4)
            
            active_filters = {}
            
            with filter_cols[0]:
                if 'model_name' in df.columns:
                    filter_models = st.multiselect("Filter Models", df['model_name'].unique())
                    if filter_models:
                        active_filters['model_name'] = filter_models
            
            with filter_cols[1]:
                if 'scale_name' in df.columns:
                    filter_scales = st.multiselect("Filter Scales", df['scale_name'].unique())
                    if filter_scales:
                        active_filters['scale_name'] = filter_scales
            
            with filter_cols[2]:
                if 'prompt_style' in df.columns:
                    filter_prompts = st.multiselect("Filter Prompt Styles", df['prompt_style'].unique())
                    if filter_prompts:
                        active_filters['prompt_style'] = filter_prompts
            
            with filter_cols[3]:
                if 'question_id' in df.columns:
                    filter_questions = st.multiselect("Filter Questions", df['question_id'].unique())
                    if filter_questions:
                        active_filters['question_id'] = filter_questions
            
            # Apply filters
            filtered_df = df.copy()
            for col, values in active_filters.items():
                filtered_df = filtered_df[filtered_df[col].isin(values)]
            
            # Create visualization
            st.markdown("### Visualization")
            
            if len(filtered_df) == 0:
                st.warning("No data matches the selected filters.")
            else:
                try:
                    # Prepare data for visualization
                    if chart_type in ["Bar Chart", "Line Chart", "Box Plot", "Violin Plot"]:
                        # Group by x-axis and calculate aggregation
                        plot_data = filtered_df.groupby(x_axis)[y_axis].agg(aggregation).reset_index()
                        
                        if color_by != "None":
                            plot_data = filtered_df.groupby([x_axis, color_by])[y_axis].agg(aggregation).reset_index()
                    else:
                        plot_data = filtered_df
                    
                    # Create the chart
                    chart_title = custom_title if custom_title else f"{aggregation.capitalize()} {y_axis} by {x_axis}"
                    
                    if chart_type == "Bar Chart":
                        fig = px.bar(plot_data, x=x_axis, y=y_axis,
                                    color=color_by if color_by != "None" else None,
                                    facet_col=facet_by if facet_by != "None" else None,
                                    title=chart_title,
                                    error_y=plot_data[y_axis].std() if show_confidence and aggregation == "mean" else None)
                    
                    elif chart_type == "Line Chart":
                        fig = px.line(plot_data, x=x_axis, y=y_axis,
                                     color=color_by if color_by != "None" else None,
                                     facet_col=facet_by if facet_by != "None" else None,
                                     title=chart_title)
                    
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(plot_data, x=x_axis, y=y_axis,
                                        color=color_by if color_by != "None" else None,
                                        facet_col=facet_by if facet_by != "None" else None,
                                        title=chart_title,
                                        trendline="ols" if show_trendline else None)
                    
                    elif chart_type == "Box Plot":
                        fig = px.box(filtered_df, x=x_axis, y=y_axis,
                                    color=color_by if color_by != "None" else None,
                                    facet_col=facet_by if facet_by != "None" else None,
                                    title=chart_title)
                    
                    elif chart_type == "Violin Plot":
                        fig = px.violin(filtered_df, x=x_axis, y=y_axis,
                                       color=color_by if color_by != "None" else None,
                                       facet_col=facet_by if facet_by != "None" else None,
                                       title=chart_title)
                    
                    elif chart_type == "Heatmap":
                        # Create pivot table for heatmap
                        pivot_data = filtered_df.pivot_table(values=y_axis, index=x_axis, 
                                                            columns=color_by if color_by != "None" else 'model_name',
                                                            aggfunc=aggregation)
                        fig = px.imshow(pivot_data, 
                                       labels=dict(x=color_by if color_by != "None" else 'model_name', 
                                                  y=x_axis, color=y_axis),
                                       title=chart_title,
                                       color_continuous_scale='RdBu_r')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export chart
                    col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
                    with col_export1:
                        if st.button("📥 Export Chart"):
                            if export_format == "HTML":
                                buffer = fig.to_html()
                                st.download_button(
                                    label="Download HTML",
                                    data=buffer,
                                    file_name="chart.html",
                                    mime="text/html"
                                )
                            else:
                                st.info(f"Use the camera icon in the chart toolbar to save as {export_format}")
                    
                    # Summary statistics
                    with st.expander("📊 Summary Statistics"):
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.dataframe(filtered_df[y_axis].describe())
                        with col_stat2:
                            if 'model_name' in filtered_df.columns:
                                model_summary = filtered_df.groupby('model_name')[y_axis].agg(['mean', 'std', 'count'])
                                st.dataframe(model_summary)
                    
                    # Raw data view
                    with st.expander("📋 View Filtered Data"):
                        st.dataframe(filtered_df)
                        
                        # Download filtered data
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Filtered Data",
                            data=csv,
                            file_name="filtered_survey_data.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    st.info("Try different combinations of axes and chart types.")
            
            # Quick comparison view
            st.markdown("### Quick Comparisons")
            comparison_type = st.selectbox(
                "Select Comparison",
                ["Model Performance by Scale", "Prompt Style Impact", "Question Response Distribution", "Score Trends"]
            )
            
            if comparison_type == "Model Performance by Scale":
                if 'scale_name' in df.columns and 'model_name' in df.columns:
                    comp_data = df.groupby(['model_name', 'scale_name'])['scored_value'].mean().reset_index()
                    fig_comp = px.bar(comp_data, x='scale_name', y='scored_value', color='model_name',
                                     barmode='group', title="Average Scores by Model and Scale")
                    st.plotly_chart(fig_comp, use_container_width=True)
            
            elif comparison_type == "Prompt Style Impact":
                if 'prompt_style' in df.columns and 'model_name' in df.columns:
                    comp_data = df.groupby(['prompt_style', 'model_name'])['scored_value'].mean().reset_index()
                    fig_comp = px.line(comp_data, x='prompt_style', y='scored_value', color='model_name',
                                      markers=True, title="Model Responses Across Prompt Styles")
                    st.plotly_chart(fig_comp, use_container_width=True)
            
            elif comparison_type == "Question Response Distribution":
                if 'question_id' in df.columns:
                    selected_questions = st.multiselect(
                        "Select Questions to Compare",
                        df['question_id'].unique(),
                        default=df['question_id'].unique()[:5] if len(df['question_id'].unique()) > 5 else df['question_id'].unique()
                    )
                    if selected_questions:
                        comp_data = df[df['question_id'].isin(selected_questions)]
                        fig_comp = px.box(comp_data, x='question_id', y='scored_value', color='model_name',
                                         title="Response Distribution by Question")
                        st.plotly_chart(fig_comp, use_container_width=True)
            
            elif comparison_type == "Score Trends":
                if 'run_number' in df.columns:
                    comp_data = df.groupby(['run_number', 'model_name'])['scored_value'].mean().reset_index()
                    fig_comp = px.line(comp_data, x='run_number', y='scored_value', color='model_name',
                                      title="Score Consistency Across Runs")
                    st.plotly_chart(fig_comp, use_container_width=True)
                
        else:
            st.info("Run a survey or load data first to use the data explorer.")
    
    with tab5:
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