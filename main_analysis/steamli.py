import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels

# Page configuration
st.set_page_config(
    page_title="Campaign Finance Effectiveness Analysis",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Summary",
    "Interactive Explorer", 
    "Scenario Simulator",
    "Deep Dive"
])

# -----------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------

@st.cache_data
def load_data():
    """
    Replace this with your actual data loading.
    For now, generating synthetic data to demonstrate structure.
    """
    np.random.seed(42)
    n_candidates = 1000  # Subset for demo; you'll have ~19k per election
    
    data = pd.DataFrame({
        'candidate_id': range(n_candidates),
        'name': [f"Candidate {i}" for i in range(n_candidates)],
        'election_year': np.random.choice([2020, 2022, 2024], n_candidates),
        'party': np.random.choice(['Party A', 'Party B', 'Party C', 'Independent'], n_candidates),
        'district_type': np.random.choice(['Urban', 'Suburban', 'Rural'], n_candidates),
        'spending': np.random.exponential(50000, n_candidates) + 10000,
        'vote_share': np.random.beta(2, 5, n_candidates) * 100,
        'incumbent': np.random.choice([True, False], n_candidates),
        'won': np.random.choice([True, False], n_candidates),
        "tf_self": np.random.exponential(30000, n_candidates) + 10000,
        "tf_party": np.random.exponential(20000, n_candidates) + 10000,
        "num_candidates": np.random.choice([5, 15, 25, 30], n_candidates)
    })
    
    # Add some correlation between spending and vote share
    data['vote_share'] = data['vote_share'] + (data['spending'] / 5000) + np.random.normal(0, 5, n_candidates)
    data['vote_share'] = data['vote_share'].clip(0, 100)
    
    return data

@st.cache_resource
def train_model(data):
    """Train regression model and identify anomalies"""
    X = data[['spending', "num_candidates"]].values
    y = data['vote_share'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions and residuals
    predictions = model.predict(X)
    residuals = y - predictions
    
    # Identify anomalies (residuals > 2 std devs)
    std_residual = np.std(residuals)
    data['predicted_vote_share'] = predictions
    data['residual'] = residuals
    data['is_anomaly'] = np.abs(residuals) > 2 * std_residual
    data['anomaly_type'] = data['residual'].apply(
        lambda x: 'Overperformer' if x > 2 * std_residual 
        else ('Underperformer' if x < -2 * std_residual else 'Normal')
    )
    
    return model, data

# Load data
data = load_data()
model, data = train_model(data)

# -----------------------------
# PAGE 1: SUMMARY
# -----------------------------

if page == "Summary":
    st.title("Campaign Finance Effectiveness Analysis")
    st.markdown("### Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", f"{len(data):,}")
    with col2:
        avg_spending = data['spending'].mean()
        st.metric("Avg Spending", f"€{avg_spending:,.0f}")
    with col3:
        anomaly_pct = (data['is_anomaly'].sum() / len(data)) * 100
        st.metric("Anomalies", f"{anomaly_pct:.1f}%")
    with col4:
        r_squared = model.score(data[['spending', "num_candidates"]], data['vote_share'])
        st.metric("Model R²", f"{r_squared:.3f}")
    
    st.markdown("---")
    
    # Main visualization: Spending vs Vote Share
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Spending Effectiveness Overview")
        
        # Add toggle for normalization
        normalize_by_candidates = st.checkbox("Normalize by number of candidates", value=False)
        
        # Prepare data based on normalization choice
        if normalize_by_candidates:
            plot_data = data.copy()
            plot_data['spending_plot'] = plot_data['spending'] / plot_data['num_candidates']
            plot_data['vote_share_plot'] = plot_data['vote_share'] / plot_data['num_candidates']
            x_label = 'Spending per Candidate (€)'
            y_label = 'Vote Share per Candidate (%)'
        else:
            plot_data = data.copy()
            plot_data['spending_plot'] = plot_data['spending']
            plot_data['vote_share_plot'] = plot_data['vote_share']
            x_label = 'Total Spending (€)'
            y_label = 'Vote Share (%)'
        
        fig = px.scatter(
            plot_data,
            x='spending_plot',
            y='vote_share_plot',
            color='anomaly_type',
            color_discrete_map={
                'Normal': '#1f77b4',
                'Overperformer': '#2ca02c',
                'Underperformer': '#d62728'
            },
            hover_data=['name', 'party', 'district_type', 'num_candidates'],
            labels={'spending_plot': x_label, 'vote_share_plot': y_label},
            title="Campaign Spending vs Vote Share" + (" (Normalized)" if normalize_by_candidates else ""),
            trendline="ols"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Findings")
        
        overperformers = data[data['anomaly_type'] == 'Overperformer']
        underperformers = data[data['anomaly_type'] == 'Underperformer']
        
        st.markdown(f"""
        **Overperformers** ({len(overperformers)} candidates)
        - Avg spending: €{overperformers['spending'].mean():,.0f}
        - Avg vote share: {overperformers['vote_share'].mean():.1f}%
        - Most efficient in: {overperformers['district_type'].mode()[0] if len(overperformers) > 0 else 'N/A'}
        
        **Underperformers** ({len(underperformers)} candidates)
        - Avg spending: €{underperformers['spending'].mean():,.0f}
        - Avg vote share: {underperformers['vote_share'].mean():.1f}%
        - Most common in: {underperformers['district_type'].mode()[0] if len(underperformers) > 0 else 'N/A'}
        """)
        
        st.info("💡 Every €10,000 spent is associated with approximately "
                f"{model.coef_[0] * 10000:.2f} percentage point increase in vote share.")
    
    # Party comparison
    st.subheader("Spending Efficiency by Party")
    party_stats = data.groupby('party').agg({
        'spending': 'mean',
        'vote_share': 'mean',
        'is_anomaly': 'sum'
    }).round(2)
    party_stats.columns = ['Avg Spending', 'Avg Vote Share', 'Anomaly Count']
    st.dataframe(party_stats, use_container_width=True)

# -----------------------------
# PAGE 2: INTERACTIVE EXPLORER
# -----------------------------
elif page == "Interactive Explorer":
    st.title("Interactive Data Explorer")
    
    # SIDEBAR (Filters for people to interact with the data)
    st.sidebar.header("Filters")
    
    selected_years = st.sidebar.multiselect(
        "Election Year",
        options=sorted(data['election_year'].unique()), # What options are available to select
        default=sorted(data['election_year'].unique()) # What options are pre-selected
    )
    
    selected_parties = st.sidebar.multiselect(
        "Party",
        options=sorted(data['party'].unique()),
        default=sorted(data['party'].unique())
    )
    
    # selected_districts = st.sidebar.multiselect(
    #     "District Type",
    #     options=sorted(data['district_type'].unique()),
    #     default=sorted(data['district_type'].unique())
    # )
    
    spending_range = st.sidebar.slider(
        "Spending Range (€)",
        min_value=int(data['spending'].min()),
        max_value=int(data['spending'].max()),
        value=(int(data['spending'].min()), int(data['spending'].max())),
        step=1000
    )
    
    show_anomalies_only = st.sidebar.checkbox("Show Anomalies Only", value=False)
    
    # Filter data
    filtered_data = data[
        (data['election_year'].isin(selected_years)) &
        (data['party'].isin(selected_parties)) &
        # (data['district_type'].isin(selected_districts)) &
        (data['spending'] >= spending_range[0]) &
        (data['spending'] <= spending_range[1])
    ]
    
    if show_anomalies_only:
        filtered_data = filtered_data[filtered_data['is_anomaly']]
    
    st.info(f"Showing {len(filtered_data):,} candidates (filtered from {len(data):,} total)")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Distribution", "Top Performers", "Funding Breakdown"])
    
    with tab1:
        color_by = st.selectbox("Color by", ['party', 'district_type', 'anomaly_type', 'election_year'])
        
        fig = px.scatter(
            filtered_data,
            x='spending',
            y='vote_share',
            color=color_by,
            hover_data=['name', 'party', 'district_type', 'election_year'],
            labels={'spending': 'Total Spending (€)', 'vote_share': 'Vote Share (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filtered_data, x='spending', nbins=30, title="Spending Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(filtered_data, x='vote_share', nbins=30, title="Vote Share Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Most Efficient Campaigns (Overperformers)")
        top_overperformers = filtered_data[
            filtered_data['anomaly_type'] == 'Overperformer'
        ].nlargest(10, 'residual')[['name', 'party', 'spending', 'vote_share', 'predicted_vote_share', 'residual']]
        st.dataframe(top_overperformers, use_container_width=True)
        
        st.subheader("Least Efficient Campaigns (Underperformers)")
        top_underperformers = filtered_data[
            filtered_data['anomaly_type'] == 'Underperformer'
        ].nsmallest(10, 'residual')[['name', 'party', 'spending', 'vote_share', 'predicted_vote_share', 'residual']]
        st.dataframe(top_underperformers, use_container_width=True)
    
    with tab4:
        st.subheader("Funding Source by Party")
        
        # TF represents total funding, so any columns marked tf_ are subsidiaries
        tf_columns = [col for col in filtered_data.columns if col.startswith('tf_')]
        
        if len(tf_columns) == 0:
            st.warning("No spending category columns (tf_*) found in the data.")
        else:
            # Group by party and calculate average spending
            party_spending = filtered_data.groupby('party').agg({
                'spending': 'mean',
                **{col: 'mean' for col in tf_columns}
            }).reset_index()
            
            num_parties = len(party_spending)
            cols_per_row = 3
            
            for idx, row in party_spending.iterrows():
                if idx % cols_per_row == 0:
                    cols = st.columns(cols_per_row)
                
                party_name = row['party']
                spending = row['spending']
                
                # Calculate spending breakdown
                spending_breakdown = {}
                accounted_spending = 0
                
                for col in tf_columns:
                    category_name = col.replace('tf_', '').replace('_', ' ').title()
                    amount = row[col]
                    if amount > 0: # NOTE: Necessary? UNCLEAR. 
                        spending_breakdown[category_name] = amount
                        accounted_spending += amount
                
                # Some spending may be unspecified because of mistakes in candidate recording, or lack of reporting due to spending <800. This attempts to account for that.
                unspecified = spending - accounted_spending
                if unspecified > 0:
                    spending_breakdown['Unspecified'] = unspecified
                
                # Pie charts have the party & spending breakdown
                if spending_breakdown:
                    fig = px.pie(
                        values=list(spending_breakdown.values()),
                        names=list(spending_breakdown.keys()),
                        title=f"{party_name}<br>Avg Total: €{spending:,.0f}"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    with cols[idx % cols_per_row]:
                        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PAGE 3: SCENARIO SIMULATOR
# -----------------------------

elif page == "Scenario Simulator":
    st.title("🎯 Scenario Simulator")
    st.markdown("Input campaign parameters to see predicted outcomes and efficiency analysis.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Campaign Parameters")
        
        sim_spending = st.number_input(
            "Total Spending (€)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=5000
        )
        
        sim_party = st.selectbox("Party", options=sorted(data['party'].unique()))
        sim_district = st.selectbox("District Type", options=sorted(data['district_type'].unique()))
        sim_incumbent = st.checkbox("Incumbent", value=False)
        
        st.markdown("---")
        st.button("🔄 Run Simulation", type="primary")
    
    with col2:
        st.subheader("Predicted Outcome")
        
        # Make prediction
        predicted_vote = model.predict([[sim_spending]])[0]
        
        # Calculate confidence interval (simplified)
        residual_std = np.std(data['residual'])
        ci_lower = predicted_vote - 1.96 * residual_std
        ci_upper = predicted_vote + 1.96 * residual_std
        
        # Display prediction
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Predicted Vote Share", f"{predicted_vote:.1f}%")
        with col_b:
            st.metric("95% CI Lower", f"{ci_lower:.1f}%")
        with col_c:
            st.metric("95% CI Upper", f"{ci_upper:.1f}%")
        
        # Visualization showing where this falls
        fig = go.Figure()
        
        # Add all data points
        fig.add_trace(go.Scatter(
            x=data['spending'],
            y=data['vote_share'],
            mode='markers',
            marker=dict(color='lightgray', size=5, opacity=0.5),
            name='All Candidates',
            hoverinfo='skip'
        ))
        
        # Add regression line
        x_range = np.linspace(data['spending'].min(), data['spending'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Expected Performance'
        ))
        
        # Add simulated candidate
        fig.add_trace(go.Scatter(
            x=[sim_spending],
            y=[predicted_vote],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Your Scenario'
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=[sim_spending, sim_spending],
            y=[ci_lower, ci_upper],
            mode='lines',
            line=dict(color='red', width=2),
            name='95% Confidence'
        ))
        
        fig.update_layout(
            title="Your Scenario in Context",
            xaxis_title="Total Spending (€)",
            yaxis_title="Vote Share (%)",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find similar candidates
        st.subheader("Similar Historical Campaigns")
        similar = data[
            (data['party'] == sim_party) &
            (data['district_type'] == sim_district) &
            (data['spending'].between(sim_spending * 0.8, sim_spending * 1.2))
        ].sort_values('spending')
        
        if len(similar) > 0:
            st.dataframe(
                similar[['name', 'election_year', 'spending', 'vote_share', 'won', 'anomaly_type']].head(10),
                use_container_width=True
            )
        else:
            st.warning("No similar campaigns found with these parameters.")
        
        # Efficiency score
        st.subheader("Efficiency Analysis")
        avg_similar_vote = similar['vote_share'].mean() if len(similar) > 0 else predicted_vote
        
        st.markdown(f"""
        - **Your predicted vote share**: {predicted_vote:.1f}%
        - **Average for similar campaigns**: {avg_similar_vote:.1f}%
        - **Estimated ROI**: €{sim_spending / max(predicted_vote, 0.1):,.0f} per percentage point
        """)

# -----------------------------
# PAGE 4: DEEP DIVE
# -----------------------------

elif page == "Deep Dive":
    st.title("📈 Deep Dive & Methodology")
    
    tab1, tab2, tab3 = st.tabs(["Model Diagnostics", "Methodology", "Download Data"])
    
    with tab1:
        st.subheader("Regression Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            r_squared = model.score(data[['spending']], data['vote_share'])
            st.metric("R² Score", f"{r_squared:.4f}")
            st.metric("Intercept", f"{model.intercept_:.4f}")
            st.metric("Coefficient", f"{model.coef_[0]:.6f}")
            
            st.markdown(f"""
            **Interpretation**: 
            - Every €1,000 increase in spending is associated with a 
            {model.coef_[0] * 1000:.3f} percentage point increase in vote share.
            - The model explains {r_squared*100:.1f}% of variance in vote share.
            """)
        
        with col2:
            # Residual plot
            fig = px.scatter(
                data,
                x='predicted_vote_share',
                y='residual',
                color='is_anomaly',
                title="Residual Plot",
                labels={'predicted_vote_share': 'Predicted Vote Share (%)', 'residual': 'Residual'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of residuals
        fig = px.histogram(
            data,
            x='residual',
            nbins=50,
            title="Distribution of Residuals"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Methodology")
        st.markdown("""
        ### Data Collection
        - **Source**: Municipal election financial data
        - **Scope**: Three election cycles with approximately 19,000 candidates each
        - **Variables**: Campaign spending, vote share, party affiliation, district characteristics
        
        ### Analysis Approach
        1. **Linear Regression Model**: We model vote share as a function of campaign spending
           - Formula: `Vote Share = β₀ + β₁(Spending) + ε`
        
        2. **Anomaly Detection**: Candidates are flagged as anomalies if their residual 
           (actual - predicted vote share) exceeds 2 standard deviations
           - **Overperformers**: Achieved >2 SD better results than predicted
           - **Underperformers**: Achieved >2 SD worse results than predicted
        
        3. **Efficiency Metrics**: We calculate ROI as spending per percentage point of vote share
        
        ### Limitations
        - Model uses only spending as predictor; other factors (candidate quality, timing, 
          messaging, external events) also matter
        - Correlation does not imply causation
        - Results vary by context (district type, election year, political climate)
        
        ### Recommendations for Use
        - Use predictions as guidelines, not guarantees
        - Consider district-specific factors
        - Examine overperformers for strategic insights
        - Benchmark against similar historical campaigns
        """)
    
    with tab3:
        st.subheader("Download Filtered Data")
        
        # Allow users to select what to download
        download_anomalies = st.checkbox("Include only anomalies", value=False)
        
        if download_anomalies:
            download_data = data[data['is_anomaly']]
        else:
            download_data = data
        
        csv = download_data.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="campaign_finance_analysis.csv",
            mime="text/csv"
        )
        
        st.info(f"Ready to download: {len(download_data):,} records")
        
        # Preview
        st.subheader("Data Preview")
        st.dataframe(download_data.head(100), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard analyzes campaign spending effectiveness across municipal elections.
Use the navigation above to explore findings, filter data, and simulate scenarios.
""")