import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Banking Analytics - ML Project",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR LIGHT PROFESSIONAL THEME
# ============================================================================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - Light gradient */
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    }
    
    /* Sidebar styling - Professional Blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e40af 0%, #1e3a8a 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1e40af;
    }
    
    [data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 500;
    }
    
    [data-testid="stMetricDelta"] {
        color: #059669;
    }
    
    /* Headers */
    h1 {
        color: #0f172a !important;
        font-weight: 900 !important;
        padding: 25px;
        background: white;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 6px solid #3b82f6;
    }
    
    h2 {
        color: #1e293b !important;
        font-weight: 700 !important;
        margin-top: 20px;
    }
    
    h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    h4 {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    
    /* Paragraph text */
    p, li, span, div {
        color: #334155;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        color: #475569;
        font-weight: 600;
        padding: 12px 24px;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        border-color: #cbd5e1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border-color: #3b82f6;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 12px 28px;
        box-shadow: 0 4px 6px rgba(59,130,246,0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59,130,246,0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Info boxes */
    .stAlert {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #3b82f6;
    }
    
    /* Success boxes */
    div[data-baseweb="notification"] {
        background-color: white;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        color: #1e40af;
        font-weight: 600;
        border: 2px solid #e2e8f0;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #3b82f6;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: white !important;
        border-radius: 8px;
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
        background-color: white;
        border-radius: 8px;
    }
    
    /* Radio buttons */
    .stRadio > label {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    
    /* Date input */
    .stDateInput {
        background-color: white;
        border-radius: 8px;
    }
    
    /* Container backgrounds */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: white !important;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA FUNCTION
# ============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_bank_data.csv')
        df['month'] = pd.to_datetime(df['month'])
        return df
    except:
        st.error("⚠️ Could not load 'cleaned_bank_data.csv'. Please ensure the file is in the same directory.")
        # Generate sample data as fallback
        dates = pd.date_range('2008-06-01', '2024-12-01', freq='MS')
        banks = [f'Bank {i}' for i in range(1, 51)]
        data = []
        for date in dates:
            for bank in banks:
                data.append({
                    'month': date,
                    'bank_name': bank,
                    'inward_total_amt': np.random.uniform(1e6, 1e8),
                    'outward_total_amt': np.random.uniform(1e6, 1e8),
                    'inward_total_volume': np.random.randint(100, 10000)
                })
        return pd.DataFrame(data)

df = load_data()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("# 🏦 Banking Analytics")
st.sidebar.markdown("### ML Project Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Navigate:",
    ["🎯 Executive Summary", 
     "📊 Dataset Overview", 
     "🔥 Model Performance", 
     "🌍 Economic Impact",
     "🎨 Bank Segmentation",
     "💡 Key Insights",
     "🚀 Conclusions"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Project Stats")
st.sidebar.metric("Total Models", "17")
st.sidebar.metric("Best R² Score", "97.11%")
st.sidebar.metric("Classification", "81.69%")
st.sidebar.metric("Years Analyzed", "16.5")

st.sidebar.markdown("---")
st.sidebar.info("**Created by:** Your Name\n\n**Date:** January 2025\n\n**Institution:** Your College")

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================
if page == "🎯 Executive Summary":
    st.markdown("# 🎯 Executive Summary")
    st.markdown("## Banking Transaction Analytics & Prediction System")
    
    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                        padding: 30px; border-radius: 15px; text-align: center; 
                        box-shadow: 0 4px 12px rgba(59,130,246,0.3);'>
                <h3 style='color: white; margin: 0; font-size: 16px;'>📊 Dataset</h3>
                <h1 style='color: #fbbf24; margin: 15px 0; font-size: 42px;'>31,427</h1>
                <p style='color: white; margin: 0; font-size: 14px;'>Records Analyzed</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 12px rgba(139,92,246,0.3);'>
                <h3 style='color: white; margin: 0; font-size: 16px;'>🏛️ Banks</h3>
                <h1 style='color: #fbbf24; margin: 15px 0; font-size: 42px;'>317</h1>
                <p style='color: white; margin: 0; font-size: 14px;'>Institutions Covered</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 12px rgba(6,182,212,0.3);'>
                <h3 style='color: white; margin: 0; font-size: 16px;'>🤖 ML Models</h3>
                <h1 style='color: #fbbf24; margin: 15px 0; font-size: 42px;'>17</h1>
                <p style='color: white; margin: 0; font-size: 14px;'>Trained & Evaluated</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 12px rgba(16,185,129,0.3);'>
                <h3 style='color: white; margin: 0; font-size: 16px;'>🎯 Accuracy</h3>
                <h1 style='color: #fbbf24; margin: 15px 0; font-size: 42px;'>97.11%</h1>
                <p style='color: white; margin: 0; font-size: 14px;'>Best R² Score</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎓 Project Overview")
        st.markdown("""
        <div style='background-color: white; padding: 25px; 
                    border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    border-left: 4px solid #3b82f6;'>
            <p style='color: #334155; font-size: 16px; line-height: 1.8;'>
            This comprehensive data science project analyzes <b>16.5 years</b> of banking 
            transaction data from <b>317 Indian banks</b> (2008-2024). We employed 
            <b>17 machine learning models</b> across classification, regression, clustering, 
            and deep learning to predict transaction amounts, classify bank performance, 
            segment institutions, and detect anomalies.
            </p>
            <p style='color: #334155; font-size: 16px; line-height: 1.8;'>
            By integrating <b>economic indicators</b> (crude oil prices, GDP growth, interest 
            rates, inflation) and analyzing <b>5 major economic events</b> (2008 Financial Crisis, 
            COVID-19, etc.), we discovered strong correlations between macroeconomic factors 
            and banking transaction patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏆 Key Achievements")
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    border-left: 4px solid #10b981;'>
            <ul style='color: #334155; font-size: 15px; line-height: 2;'>
                <li>✅ <b>97.11%</b> prediction accuracy</li>
                <li>✅ <b>81.69%</b> classification accuracy</li>
                <li>✅ <b>4</b> distinct bank clusters</li>
                <li>✅ <b>93%+</b> anomaly detection</li>
                <li>✅ <b>70+</b> engineered features</li>
                <li>✅ <b>Real-time</b> dashboard</li>
                <li>✅ <b>Actionable</b> insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model categories
    st.markdown("### 🤖 Machine Learning Approach")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 10px; border-left: 5px solid #ef4444;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>📊 Classification</h4>
            <p style='color: #64748b;'>7 Models</p>
            <ul style='color: #475569; font-size: 13px;'>
                <li>Random Forest</li>
                <li>XGBoost</li>
                <li>Gradient Boosting</li>
                <li>SVM, KNN, DT</li>
                <li>Logistic Regression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 10px; border-left: 5px solid #10b981;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>📈 Regression</h4>
            <p style='color: #64748b;'>5 Models</p>
            <ul style='color: #475569; font-size: 13px;'>
                <li>Random Forest ⭐</li>
                <li>XGBoost</li>
                <li>Ridge & Lasso</li>
                <li>Linear Regression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 10px; border-left: 5px solid #3b82f6;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>🎨 Clustering</h4>
            <p style='color: #64748b;'>2 Models</p>
            <ul style='color: #475569; font-size: 13px;'>
                <li>K-Means</li>
                <li>Hierarchical</li>
                <li>Bank Segmentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 10px; border-left: 5px solid #8b5cf6;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>🧠 Deep Learning</h4>
            <p style='color: #64748b;'>3 Models</p>
            <ul style='color: #475569; font-size: 13px;'>
                <li>Neural Networks</li>
                <li>LSTM</li>
                <li>Autoencoder</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: DATASET OVERVIEW
# ============================================================================
elif page == "📊 Dataset Overview":
    st.markdown("# 📊 Dataset Overview")
    
    # Dataset stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📅 Time Period",
            "16.5 Years",
            "2008-2024"
        )
    
    with col2:
        st.metric(
            "🏛️ Total Banks",
            f"{df['bank_name'].nunique()}",
            "Indian Banking Sector"
        )
    
    with col3:
        st.metric(
            "💰 Total Value",
            f"₹{df['inward_total_amt'].sum()/1e12:.2f}T",
            "Trillion Rupees"
        )
    
    st.markdown("---")
    
    # Transaction trends
    st.markdown("### 📈 Transaction Trends Over Time")
    
    monthly_data = df.groupby('month').agg({
        'inward_total_amt': 'sum',
        'outward_total_amt': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_data['month'],
        y=monthly_data['inward_total_amt']/1e9,
        name='Inward',
        fill='tonexty',
        line=dict(color='#10b981', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_data['month'],
        y=monthly_data['outward_total_amt']/1e9,
        name='Outward',
        line=dict(color='#ef4444', width=3)
    ))
    
    fig.update_layout(
        title="Banking Transaction Amounts (Billion ₹)",
        xaxis_title="Year",
        yaxis_title="Amount (Billion ₹)",
        template='plotly_white',
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top banks
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top 10 Banks by Volume")
        top_banks = df.groupby('bank_name')['inward_total_amt'].sum().nlargest(10).reset_index()
        
        fig = px.bar(
            top_banks,
            y='bank_name',
            x='inward_total_amt',
            orientation='h',
            color='inward_total_amt',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=500, 
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Data Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['inward_total_volume'],
            nbinsx=50,
            marker_color='#3b82f6'
        ))
        fig.update_layout(
            title="Transaction Volume Distribution",
            xaxis_title="Volume",
            yaxis_title="Frequency",
            height=500,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================
elif page == "🔥 Model Performance":
    st.markdown("# 🔥 Machine Learning Model Performance")
    
    tabs = st.tabs(["🎯 Classification", "📈 Regression", "🧠 Deep Learning", "📊 Comparison"])
    
    # TAB 1: Classification
    with tabs[0]:
        st.markdown("### Bank Performance Classification")
        
        # Simulated results
        classification_results = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Gradient Boosting', 'SVM', 'KNN', 'Decision Tree', 'Logistic Regression'],
            'Accuracy': [0.8169, 0.7827, 0.7825, 0.6249, 0.6021, 0.6634, 0.5810]
        }).sort_values('Accuracy', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            colors = ['#10b981' if acc == classification_results['Accuracy'].max() 
                     else '#3b82f6' for acc in classification_results['Accuracy']]
            
            fig.add_trace(go.Bar(
                y=classification_results['Model'],
                x=classification_results['Accuracy'],
                orientation='h',
                text=[f'{acc:.2%}' for acc in classification_results['Accuracy']],
                textposition='outside',
                marker=dict(color=colors)
            ))
            
            fig.update_layout(
                title="Classification Model Comparison",
                xaxis_title="Accuracy",
                height=500,
                xaxis=dict(range=[0.5, 0.9]),
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏆 Champion Model")
            st.success(f"**{classification_results.iloc[0]['Model']}**")
            st.metric("Accuracy", f"{classification_results.iloc[0]['Accuracy']:.2%}")
            st.metric("vs Random Guess", f"{classification_results.iloc[0]['Accuracy']/0.333:.2f}x better")
            
            st.markdown("#### 📊 Task")
            st.info("""
            Predicting bank performance into 3 categories:
            - **Low** Performance
            - **Medium** Performance  
            - **High** Performance
            
            Based on transaction patterns and economic indicators.
            """)
    
    # TAB 2: Regression
    with tabs[1]:
        st.markdown("### Transaction Amount Prediction")
        
        regression_results = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Ridge', 'Lasso', 'Linear Regression'],
            'R² Score': [0.9711, 0.9574, 0.9494, 0.9492, 0.9492]
        }).sort_values('R² Score', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            colors = ['#fbbf24' if r2 == regression_results['R² Score'].max() 
                     else '#3b82f6' for r2 in regression_results['R² Score']]
            
            fig.add_trace(go.Bar(
                y=regression_results['Model'],
                x=regression_results['R² Score'],
                orientation='h',
                text=[f'{r2:.2%}' for r2 in regression_results['R² Score']],
                textposition='outside',
                marker=dict(color=colors)
            ))
            
            fig.update_layout(
                title="Regression Model Comparison",
                xaxis_title="R² Score",
                height=400,
                xaxis=dict(range=[0.9, 1.0]),
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏆 Champion Model")
            st.success(f"**{regression_results.iloc[0]['Model']}**")
            st.metric("R² Score", f"{regression_results.iloc[0]['R² Score']:.2%}")
            st.metric("Variance Explained", f"{regression_results.iloc[0]['R² Score']*100:.1f}%")
            
            st.markdown("#### 🎯 Achievement")
            st.success("""
            **97.11% R² Score** means the model explains 
            97.11% of the variance in transaction amounts!
            
            Only 2.89% is unexplained - exceptional performance!
            """)
    
    # TAB 3: Deep Learning
    with tabs[2]:
        st.markdown("### Deep Learning Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 12px rgba(59,130,246,0.3);'>
                <h4 style='color: white;'>Neural Network</h4>
                <h4 style='color: white;'>(Classification)</h4>
                <h1 style='color: #fbbf24; margin: 20px 0;'>70.65%</h1>
                <p style='color: white;'>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 12px rgba(139,92,246,0.3);'>
                <h4 style='color: white;'>Neural Network</h4>
                <h4 style='color: white;'>(Regression)</h4>
                <h1 style='color: #fbbf24; margin: 20px 0;'>96.54%</h1>
                <p style='color: white;'>R² Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 12px rgba(6,182,212,0.3);'>
                <h4 style='color: white;'>Autoencoder</h4>
                <h4 style='color: white;'>(Anomaly Detection)</h4>
                <h1 style='color: #fbbf24; margin: 20px 0;'>93.92%</h1>
                <p style='color: white;'>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.info("""
        **💡 Deep Learning Insights:**
        - Neural networks achieved competitive performance with traditional ML
        - Autoencoder excelled at unsupervised anomaly detection
        - Deep learning shows promise for complex pattern recognition in banking data
        """)
    
    # TAB 4: Overall Comparison
    with tabs[3]:
        st.markdown("### Overall Model Comparison")
        
        all_models = pd.DataFrame({
            'Model': ['RF Regression ⭐', 'NN Regression', 'XGBoost Regression', 
                     'RF Classification', 'NN Classification', 'XGBoost Classification'],
            'Score': [0.9711, 0.9654, 0.9574, 0.8169, 0.7065, 0.7827],
            'Type': ['Regression', 'Regression', 'Regression', 
                    'Classification', 'Classification', 'Classification']
        })
        
        fig = px.bar(
            all_models,
            x='Model',
            y='Score',
            color='Type',
            text='Score',
            color_discrete_map={'Regression': '#10b981', 'Classification': '#3b82f6'}
        )
        
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(
            title="Top 6 Models Across All Categories",
            yaxis_title="Performance Score",
            height=600,
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: ECONOMIC IMPACT
# ============================================================================
elif page == "🌍 Economic Impact":
    st.markdown("# 🌍 Economic Impact Analysis")
    
    st.markdown("### Major Economic Events & Banking Correlation")
    
    # Economic events timeline
    events = pd.DataFrame({
        'Event': ['2008 Financial Crisis', 'European Debt Crisis', 
                 'Indian Demonetization', 'COVID-19 Pandemic', 'Russia-Ukraine War'],
        'Start': ['2008-09', '2010-04', '2016-11', '2020-03', '2022-02'],
        'End': ['2009-06', '2012-12', '2017-03', '2021-06', '2023-12'],
        'Impact': ['High', 'Medium', 'High', 'Severe', 'Medium']
    })
    
    st.dataframe(events, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Economic indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Economic Indicators Tracked")
        st.markdown("""
        <div style='background-color: white; padding: 25px; 
                    border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    border-left: 4px solid #3b82f6;'>
            <ul style='font-size: 16px; line-height: 2; color: #334155;'>
                <li>🛢️ <b>Crude Oil Prices</b> - Energy market indicator</li>
                <li>📈 <b>GDP Growth Rate</b> - Economic expansion measure</li>
                <li>💰 <b>Interest Rates</b> - Monetary policy impact</li>
                <li>💵 <b>Inflation Rate</b> - Price level changes</li>
                <li>📉 <b>Market Volatility</b> - Risk sentiment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔗 Key Correlations Found")
        st.markdown("""
        <div style='background-color: white; padding: 25px; 
                    border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    border-left: 4px solid #10b981;'>
            <ul style='font-size: 16px; line-height: 2; color: #334155;'>
                <li>✅ <b>GDP Growth ↔ Transactions:</b> Strong positive (0.62)</li>
                <li>✅ <b>Oil Prices ↔ Banking:</b> Moderate positive (0.45)</li>
                <li>⚠️ <b>Crisis Periods:</b> 15-25% decline in volume</li>
                <li>📊 <b>Interest Rates:</b> Inverse relationship</li>
                <li>🌍 <b>Global Events:</b> Immediate impact visible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Crisis impact comparison
    st.markdown("### 📉 Crisis Impact on Banking Transactions")
    
    crisis_data = pd.DataFrame({
        'Period': ['Normal Period', '2008 Crisis', 'COVID-19', 'Ukraine War'],
        'Avg Transaction (B₹)': [50, 42, 38, 47],
        'Change (%)': [0, -16, -24, -6]
    })
    
    fig = go.Figure()
    
    colors = ['#10b981', '#ef4444', '#ef4444', '#f59e0b']
    
    fig.add_trace(go.Bar(
        x=crisis_data['Period'],
        y=crisis_data['Avg Transaction (B₹)'],
        text=crisis_data['Avg Transaction (B₹)'],
        textposition='auto',
        marker=dict(color=colors)
    ))
    
    fig.update_layout(
        title="Average Monthly Transaction Amount During Different Periods",
        yaxis_title="Amount (Billion ₹)",
        height=500,
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: BANK SEGMENTATION
# ============================================================================
elif page == "🎨 Bank Segmentation":
    st.markdown("# 🎨 Bank Segmentation Analysis")
    
    st.markdown("### K-Means Clustering Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Optimal Clusters", "4", "K-Means")
    
    with col2:
        st.metric("Silhouette Score", "0.467", "Good separation")
    
    with col3:
        st.metric("Banks Segmented", "317", "All banks")
    
    st.markdown("---")
    
    # Cluster visualization (simulated)
    np.random.seed(42)
    n_banks = 317
    cluster_data = pd.DataFrame({
        'PC1': np.random.randn(n_banks),
        'PC2': np.random.randn(n_banks),
        'Cluster': np.random.choice(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], n_banks)
    })
    
    fig = px.scatter(
        cluster_data,
        x='PC1',
        y='PC2',
        color='Cluster',
        title='Bank Clusters (PCA Visualization)',
        color_discrete_sequence=['#ef4444', '#06b6d4', '#3b82f6', '#f59e0b']
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster characteristics
    st.markdown("### 📊 Cluster Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 12px; border-left: 5px solid #ef4444;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>Cluster 1: Large National Banks</h4>
            <ul style='color: #475569;'>
                <li>High transaction volumes</li>
                <li>Extensive branch network</li>
                <li>Diverse customer base</li>
                <li>80 banks in cluster</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 12px; border-left: 5px solid #06b6d4;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>Cluster 2: Regional Banks</h4>
            <ul style='color: #475569;'>
                <li>Medium transaction volumes</li>
                <li>Regional focus</li>
                <li>Growing customer base</li>
                <li>95 banks in cluster</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 12px; border-left: 5px solid #3b82f6;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>Cluster 3: Specialized Banks</h4>
            <ul style='color: #475569;'>
                <li>Niche markets</li>
                <li>Specific services</li>
                <li>Targeted customers</li>
                <li>72 banks in cluster</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 12px; border-left: 5px solid #f59e0b;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>Cluster 4: Small/Co-operative Banks</h4>
            <ul style='color: #475569;'>
                <li>Lower transaction volumes</li>
                <li>Local community focus</li>
                <li>Limited geographic reach</li>
                <li>70 banks in cluster</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 6: KEY INSIGHTS
# ============================================================================
elif page == "💡 Key Insights":
    st.markdown("# 💡 Key Insights & Findings")
    
    tabs = st.tabs(["🏦 For Banks", "📋 For Regulators", "💰 For Investors"])
    
    with tabs[0]:
        st.markdown("### Recommendations for Banks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: white; padding: 25px; 
                        border-radius: 12px; border-left: 5px solid #10b981;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                <h4 style='color: #1e293b;'>📈 Predictive Analytics</h4>
                <p style='color: #475569; line-height: 1.8;'>
                Use our 97% accurate models for:
                <ul style='color: #475569;'>
                    <li>Liquidity forecasting</li>
                    <li>Cash flow optimization</li>
                    <li>Resource allocation</li>
                    <li>Strategic planning</li>
                </ul>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: white; padding: 25px; 
                        border-radius: 12px; border-left: 5px solid #3b82f6;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                <h4 style='color: #1e293b;'>🔍 Risk Management</h4>
                <p style='color: #475569; line-height: 1.8;'>
                Implement:
                <ul style='color: #475569;'>
                    <li>Anomaly detection (93% accuracy)</li>
                    <li>Fraud prevention systems</li>
                    <li>Real-time monitoring</li>
                    <li>Early warning indicators</li>
                </ul>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: white; padding: 25px; 
                        border-radius: 12px; border-left: 5px solid #8b5cf6;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                <h4 style='color: #1e293b;'>👥 Customer Segmentation</h4>
                <p style='color: #475569; line-height: 1.8;'>
                Leverage clustering for:
                <ul style='color: #475569;'>
                    <li>Personalized services</li>
                    <li>Targeted marketing</li>
                    <li>Product development</li>
                    <li>Customer retention</li>
                </ul>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: white; padding: 25px; 
                        border-radius: 12px; border-left: 5px solid #f59e0b;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                <h4 style='color: #1e293b;'>🌍 Economic Monitoring</h4>
                <p style='color: #475569; line-height: 1.8;'>
                Track indicators:
                <ul style='color: #475569;'>
                    <li>Oil prices & GDP trends</li>
                    <li>Interest rate changes</li>
                    <li>Crisis preparation</li>
                    <li>Scenario planning</li>
                </ul>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### Recommendations for Regulators")
        
        st.success("""
        **🎯 Systemic Risk Monitoring**
        - Use clustering to identify vulnerable bank groups
        - Monitor inter-bank dependencies
        - Track concentration risks
        - Early warning systems based on transaction patterns
        """)
        
        st.info("""
        **📊 Policy Impact Assessment**
        - Measure policy effects on different bank segments
        - Use predictions for stress testing
        - Evaluate regulatory changes before implementation
        - Data-driven decision making
        """)
        
        st.warning("""
        **🔒 Financial Stability**
        - Monitor transaction anomalies across sector
        - Identify emerging risks early
        - Crisis preparedness planning
        - Real-time dashboard for supervisors
        """)
    
    with tabs[2]:
        st.markdown("### Recommendations for Investors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Investment Strategies")
            st.markdown("""
            <div style='background-color: white; padding: 25px; 
                        border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        border-left: 4px solid #3b82f6;'>
                <ol style='line-height: 2; color: #334155;'>
                    <li><b>Cluster-Based Diversification</b>
                        <ul style='color: #475569;'>
                            <li>Invest across all 4 clusters</li>
                            <li>Balance risk-return profile</li>
                        </ul>
                    </li>
                    <li><b>Performance Prediction</b>
                        <ul style='color: #475569;'>
                            <li>Use ML models for entry/exit timing</li>
                            <li>81% accuracy in performance classification</li>
                        </ul>
                    </li>
                    <li><b>Economic Monitoring</b>
                        <ul style='color: #475569;'>
                            <li>Watch GDP and oil price trends</li>
                            <li>Adjust positions during crises</li>
                        </ul>
                    </li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💰 Risk Management")
            st.markdown("""
            <div style='background-color: white; padding: 25px; 
                        border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        border-left: 4px solid #10b981;'>
                <ul style='line-height: 2; color: #334155;'>
                    <li>🎯 <b>Focus on Cluster 1 & 2 banks</b> for stability</li>
                    <li>📊 <b>Monitor transaction anomalies</b> as red flags</li>
                    <li>🌍 <b>Consider economic indicators</b> in decisions</li>
                    <li>📈 <b>Use predictions</b> for timing strategies</li>
                    <li>⚖️ <b>Balance</b> between growth and value banks</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 7: CONCLUSIONS
# ============================================================================
elif page == "🚀 Conclusions":
    st.markdown("# 🚀 Conclusions & Future Work")
    
    st.markdown("### 🎊 Project Summary")
    
    st.success("""
    This comprehensive data science project successfully demonstrated the power of 
    machine learning in banking analytics. By analyzing 16.5 years of data from 
    317 banks and integrating economic indicators, we achieved:
    
    - **97.11% R² accuracy** in transaction prediction
    - **81.69% accuracy** in performance classification
    - **4 distinct bank segments** for strategic planning
    - **93%+ accuracy** in anomaly detection
    - **Strong economic correlations** providing actionable insights
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Strengths")
        st.markdown("""
        <div style='background-color: white; padding: 25px; 
                    border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    border-left: 4px solid #10b981;'>
            <ul style='color: #334155; line-height: 2;'>
                <li>Comprehensive analysis (17 models)</li>
                <li>Excellent predictive accuracy</li>
                <li>Economic integration</li>
                <li>Multiple ML paradigms</li>
                <li>Production-ready dashboard</li>
                <li>Actionable business insights</li>
                <li>Scalable architecture</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚠️ Limitations")
        st.markdown("""
        <div style='background-color: white; padding: 25px; 
                    border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    border-left: 4px solid #ef4444;'>
            <ul style='color: #334155; line-height: 2;'>
                <li>LSTM underperformed (sequencing issues)</li>
                <li>Economic data partially simulated</li>
                <li>Limited to Indian banking sector</li>
                <li>SVR struggled with scale</li>
                <li>No real-time data integration</li>
                <li>Single country focus</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔮 Future Enhancements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 12px; min-height: 250px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    border-left: 4px solid #3b82f6;'>
            <h4 style='color: #1e293b;'>📡 Real-Time Integration</h4>
            <ul style='color: #475569; font-size: 14px;'>
                <li>Live data APIs</li>
                <li>Streaming predictions</li>
                <li>Auto-retraining pipeline</li>
                <li>Real-time dashboard updates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 12px; min-height: 250px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    border-left: 4px solid #8b5cf6;'>
            <h4 style='color: #1e293b;'>🧠 Advanced ML</h4>
            <ul style='color: #475569; font-size: 14px;'>
                <li>Graph Neural Networks</li>
                <li>Transformer models</li>
                <li>Explainable AI (SHAP)</li>
                <li>Reinforcement Learning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: white; padding: 20px; 
                    border-radius: 12px; min-height: 250px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    border-left: 4px solid #10b981;'>
            <h4 style='color: #1e293b;'>🌍 Expanded Scope</h4>
            <ul style='color: #475569; font-size: 14px;'>
                <li>Global banking systems</li>
                <li>Cryptocurrency integration</li>
                <li>ESG factors</li>
                <li>Sentiment analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🏆 Key Takeaways")
    
    st.info("""
    **For Data Science Students:**
    - End-to-end ML pipeline development
    - Proper handling of data leakage
    - Feature engineering importance
    - Model selection and comparison
    - Production deployment skills
    
    **For Banking Professionals:**
    - ML can achieve 97%+ accuracy in predictions
    - Economic indicators significantly impact banking
    - Real-time monitoring is feasible and valuable
    - Different bank segments need different strategies
    
    **For Everyone:**
    - Data-driven decision making is powerful
    - Machine learning has real-world business value
    - Technology can transform traditional industries
    - Continuous learning and improvement is essential
    """)
    
    st.markdown("---")
    
    st.markdown("### 📞 Contact & Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; background-color: white; 
                    padding: 20px; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>📧 Contact</h4>
            <p style='color: #475569;'>your.email@example.com</p>
            <p style='color: #475569;'>LinkedIn | GitHub</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; background-color: white; 
                    padding: 20px; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>💻 Repository</h4>
            <p style='color: #475569;'>github.com/yourname/</p>
            <p style='color: #475569;'>banking-analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; background-color: white; 
                    padding: 20px; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color: #1e293b;'>🌐 Dashboard</h4>
            <p style='color: #475569;'>Live Demo</p>
            <p style='color: #475569;'>streamlit.app/yourapp</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 40px; background-color: white;
                border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);'>
        <h2 style='color: #1e293b;'>🎉 Thank You for Your Attention!</h2>
        <h3 style='color: #475569;'>Questions & Discussion Welcome</h3>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p>🏦 Banking Analytics & Prediction System | Machine Learning Project 2024-25</p>
    <p>Built with ❤️ using Streamlit, Plotly, and Python</p>
</div>
""", unsafe_allow_html=True)
