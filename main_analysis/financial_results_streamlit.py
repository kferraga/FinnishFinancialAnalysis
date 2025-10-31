import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import os.path
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
    "Interactive Data",
    "Scenario Simulator",
    "Methodology"
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
    dir_name = os.path.abspath(os.path.dirname(__file__))
    location = os.path.join(dir_name, 'files\\outputs\\financial_analysis_data.csv')

    data = pd.read_csv(location)
    return data

@st.cache_resource
def train_model(data):
    """Train regression model and identify anomalies"""
    X = data[['total_raised', "num_candidates_in_municipality", "incumbency_None", "multi_election_Kyllä"]].values
    y = data['vote_prct'].values
    
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

# Other useful data
largest_municipality_info = {
    # I want the name of the normalized municipality to change based on which has the most votes so the scale doesn't change.
    "municipality": data.loc[data['num_candidates_in_municipality'].idxmax(), 'municipality'],
    "year": data.loc[data['num_candidates_in_municipality'].idxmax(), 'year']}

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
        avg_spending = data['total_raised'].mean()
        st.metric("Avg Spending", f"€{avg_spending:,.0f}")
    with col3:
        anomaly_pct = (data['is_anomaly'].sum() / len(data)) * 100
        st.metric("Anomalies", f"{anomaly_pct:.1f}%")
    with col4:
        r_squared = model.score(data[['total_raised', "num_candidates_in_municipality", "incumbency_None", "multi_election_Kyllä"]], data['vote_prct'])
        st.metric("Model R²", f"{r_squared:.3f}")
    
    st.markdown("---")
    
    # Main visualization: Spending vs Vote Share
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Spending Effectiveness Overview")

        normalize_by_candidates = st.checkbox("Normalize by number of candidates", value=True,
                                              help=f"Normalizes vote percentages based on the municipal election with the most candidates ({largest_municipality_info["year"]} {largest_municipality_info["municipality"]}), thereby simulating how they would preform in said election. This accounts for varying competition levels - a 5% vote share means more in a 100-candidate race than a 20-candidate race."
                                              )

        # Prepare data based on normalization choice
        if normalize_by_candidates:
            plot_data = data.copy()
            plot_data['spending_plot'] = plot_data['total_raised']
            plot_data['vote_share_plot'] = plot_data['normalized_vote_prct']
            x_label = 'Total Spending (€)'
            y_label = f'{largest_municipality_info["year"]} {largest_municipality_info["municipality"]} Normalized Vote Share (%)'
        else:
            plot_data = data.copy()
            plot_data['spending_plot'] = plot_data['total_raised']
            plot_data['vote_share_plot'] = plot_data['vote_prct']
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
            hover_data=['full_name', 'party', 'municipality', 'year', "num_candidates_in_municipality"],
            labels={'spending_plot': x_label, 'vote_share_plot': y_label},
            title="Campaign Spending vs Vote Share" + (" (Normalized)" if normalize_by_candidates else ""),
            trendline="ols"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Anomaly Stats")
        
        overperformers = data[data['anomaly_type'] == 'Overperformer']
        normal = data[data['anomaly_type'] == 'Normal']
        underperformers = data[data['anomaly_type'] == 'Underperformer']
        
        st.markdown(f"""
        **Overperformers** ({len(overperformers)} candidates)
        - Avg raised: €{overperformers['total_raised'].mean():,.0f}
        - Avg vote share: {overperformers['vote_prct'].mean():.1f}%
        
        **Underperformers** ({len(underperformers)} candidates)
        - Avg raised: €{underperformers['total_raised'].mean():,.0f}
        - Avg vote share: {underperformers['vote_prct'].mean():.1f}%
        
        **Normal Candidates** ({len(normal)} candidates)
        - Avg raised: €{normal['total_raised'].mean():,.0f}
        - Avg vote share: {normal['vote_prct'].mean():.1f}%
        """)

        st.info("Every €10,000 spent is associated with approximately a "
                f"{model.coef_[0] * 10000:.2f}% point increase in vote share.")
    
    # Party comparison
    st.subheader("Spending Efficiency by Party", help="NOTE: This is ONLY for candidates who reported financial information.")
    party_stats = data.groupby('party').agg({
        'total_raised': ['mean', "median"],
        'vote_prct': ['mean', "median"],
        'is_anomaly': ['sum', 'count']
    }).round(2)
    party_stats.columns = ['Avg Spending', "Median Spending", 'Avg Vote Share', "Median Vote Share", 'Anomaly Count', 'Total Candidates']

    # Calculate percentage
    party_stats['Anomaly %'] = (party_stats['Anomaly Count'] / party_stats['Total Candidates'] * 100).round(2)

    # Select and reorder columns
    party_stats = party_stats[['Avg Spending', "Median Spending", 'Avg Vote Share', "Median Vote Share", 'Total Candidates', 'Anomaly %']]

    st.dataframe(party_stats, use_container_width=True)

# -----------------------------
# PAGE 2: INTERACTIVE Data
# -----------------------------
elif page == "Interactive Data":
    st.title("Interactive Data")
    
    # SIDEBAR (Filters for people to interact with the data)
    st.sidebar.header("Filters")
    
    selected_years = st.sidebar.multiselect(
        "Election Year",
        options=sorted(data['year'].unique()), # What options are available to select
        default=sorted(data['year'].unique()) # What options are pre-selected
    )
    
    selected_parties = st.sidebar.multiselect(
        "Party",
        options=sorted(data['party'].unique()),
        default=sorted(data['party'].unique())
    )
    
    selected_districts = st.sidebar.multiselect(
        "Municipality",
        options=sorted(data['municipality'].unique()),
        default=sorted(data['municipality'].unique())
    )
    
    spending_range = st.sidebar.slider(
        "Spending Range (€)",
        min_value=int(data['total_raised'].min()),
        max_value=int(data['total_raised'].max()),
        value=(int(data['total_raised'].min()), int(data['total_raised'].max())),
        step=1000
    )
    
    show_anomalies_only = st.sidebar.checkbox("Show Anomalies Only", value=False)
    
    # Filter data
    filtered_data = data[
        (data['year'].isin(selected_years)) &
        (data['party'].isin(selected_parties)) &
        # (data['municipality'].isin(selected_districts)) &
        (data['total_raised'] >= spending_range[0]) &
        (data['total_raised'] <= spending_range[1])
    ]
    
    if show_anomalies_only:
        filtered_data = filtered_data[filtered_data['is_anomaly']]
    
    st.info(f"Showing {len(filtered_data):,} candidates (filtered from {len(data):,} total)")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scatter Plot", "Distribution", "Anomalies", "Funding Breakdown", "Spending Breakdown"])
    
    with tab1:
        normalize_by_candidates_scatter = st.checkbox("Normalize by number of candidates", value=True, key="normalize_scatter",
                                              help=f"Normalizes vote percentages based on the municipal election with the most candidates ({largest_municipality_info["year"]} {largest_municipality_info["municipality"]}), thereby simulating how they would preform in said election. This accounts for varying competition levels - a 5% vote share means more in a 100-candidate race than a 20-candidate race."
                                              )
        vote_column = 'normalized_vote_prct' if normalize_by_candidates_scatter else 'vote_prct'
        title = f'{largest_municipality_info["year"]} {largest_municipality_info["municipality"]} Normalized Vote Share (%)' if normalize_by_candidates_scatter else "Vote Share (%)"

        fig = px.scatter(
            filtered_data,
            x='total_raised',
            y=vote_column,
            hover_data=['full_name', 'party', 'municipality', 'year'],
            labels={'total_raised': 'Total Spending (€)', vote_column: title}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        normalize_by_candidates_distribution = st.checkbox("Normalize by number of candidates", value=True, key="normalize_distribution",
                                              help=f"Normalizes vote percentages based on the municipal election with the most candidates ({largest_municipality_info["year"]} {largest_municipality_info["municipality"]}), thereby simulating how they would preform in said election. This accounts for varying competition levels - a 5% vote share means more in a 100-candidate race than a 20-candidate race."
                                              )
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filtered_data, x='total_raised', nbins=50, title="Spending Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            vote_column = 'normalized_vote_prct' if normalize_by_candidates_distribution else 'vote_prct'
            title = "Vote Share Distribution (Normalized)" if normalize_by_candidates_distribution else "Vote Share Distribution"

            fig = px.histogram(filtered_data, x=vote_column, nbins=50, title=title)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Strongest Overperformers")
        top_overperformers = filtered_data[
            filtered_data['anomaly_type'] == 'Overperformer'
        ].nlargest(10, 'residual')[['full_name', 'party', 'total_raised', 'vote_prct', 'predicted_vote_share', 'residual']]
        st.dataframe(top_overperformers, use_container_width=True)
        
        st.subheader("Worst Underperformers")
        top_underperformers = filtered_data[
            filtered_data['anomaly_type'] == 'Underperformer'
        ].nsmallest(10, 'residual')[['full_name', 'party', 'total_raised', 'vote_prct', 'predicted_vote_share', 'residual']]
        st.dataframe(top_underperformers, use_container_width=True)

    with tab4:
        st.subheader("Funding Source by Party")

        # TF represents total funding, so any columns marked tf_ are subsidiaries
        tf_columns = [col for col in filtered_data.columns if col.startswith('tf_')]

        if len(tf_columns) == 0:
            st.warning("No total_raised category columns (tf_*) found in the data.")
        else:
            all_categories = [col.replace('tf_', '').replace('_', ' ').title() for col in tf_columns]
            all_categories.append('Unspecified')

            color_palette = px.colors.qualitative.Set3
            color_map = {category: color_palette[i % len(color_palette)]
                         for i, category in enumerate(all_categories)} # Consistent color mapping for comparison purposes

            # Group by party and calculate average total_raised
            party_spending = filtered_data.groupby('party').agg({
                'total_raised': 'mean',
                **{col: 'mean' for col in tf_columns}
            }).reset_index()

            num_parties = len(party_spending)
            cols_per_row = 2

            for idx, row in party_spending.iterrows():
                if idx % cols_per_row == 0:
                    cols = st.columns(cols_per_row)

                party_name = row['party']
                total_raised = row['total_raised']

                # Calculate total_raised breakdown
                spending_breakdown = {}
                accounted_spending = 0

                for col in tf_columns:
                    category_name = col.replace('tf_', '').replace('_', ' ').title()
                    amount = row[col]
                    if amount > 0: # NOTE: Necessary? UNCLEAR.
                        spending_breakdown[category_name] = amount
                        accounted_spending += amount

                # Some total_raised may be unspecified because of mistakes in candidate recording, or lack of reporting due to total_raised <800. This attempts to account for that.
                unspecified = total_raised - accounted_spending
                if unspecified > 0:
                    spending_breakdown['Unspecified'] = unspecified

                # Pie charts have the party & total_raised breakdown
                if spending_breakdown:

                    fig = px.pie(
                        values=list(spending_breakdown.values()),
                        names=list(spending_breakdown.keys()),
                        title=f"{party_name}<br>Avg Total: €{total_raised:,.0f}",
                        color=list(spending_breakdown.keys()),
                        color_discrete_map=color_map
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')

                    with cols[idx % cols_per_row]:
                        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Spending by Party")

        # TF represents total funding, so any columns marked tf_ are subsidiaries
        sl_columns = [col for col in filtered_data.columns if col.startswith('sl_')]

        if len(sl_columns) == 0:
            st.warning("No total_expenses category columns (sl_*) found in the data.")
        else:
            all_categories = [col.replace('sl_', '').replace('_', ' ').title() for col in tf_columns]
            all_categories.append('Unspecified')

            color_palette = px.colors.qualitative.Set3
            color_map = {category: color_palette[i % len(color_palette)]
                         for i, category in enumerate(all_categories)} # Consistent color mapping for comparison purposes


            # Group by party and calculate average total_raised
            party_spending = filtered_data.groupby('party').agg({
                'total_expenses': 'mean',
                **{col: 'mean' for col in sl_columns}
            }).reset_index()

            num_parties = len(party_spending)
            cols_per_row = 2

            for idx, row in party_spending.iterrows():
                if idx % cols_per_row == 0:
                    cols = st.columns(cols_per_row)

                party_name = row['party']
                total_expenses = row['total_expenses']

                # Calculate total_expenses breakdown
                spending_breakdown = {}
                accounted_spending = 0

                for col in sl_columns:
                    category_name = col.replace('sl_', '').replace('_', ' ').title()
                    amount = row[col]
                    if amount > 0:
                        spending_breakdown[category_name] = amount
                        accounted_spending += amount


                unspecified = total_expenses - accounted_spending
                if unspecified > 0:
                    spending_breakdown['Unspecified'] = unspecified

                if spending_breakdown:

                    fig = px.pie(
                        values=list(spending_breakdown.values()),
                        names=list(spending_breakdown.keys()),
                        title=f"{party_name}<br>Avg Total: €{total_expenses:,.0f}",
                        color=list(spending_breakdown.keys()),
                        color_discrete_map=color_map
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')

                    with cols[idx % cols_per_row]:
                        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PAGE 3: SCENARIO SIMULATOR
# -----------------------------

elif page == "Scenario Simulator":
    st.title("Scenario Simulator")
    st.markdown("Input campaign parameters to see predicted outcomes.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Campaign Parameters")
        
        sim_spending = st.number_input(
            "Total Spending (€)",
            min_value=0,
            max_value=100000,
            value=2000,
            step=1000
        )

        sim_municipality = st.selectbox("Municipality", options=sorted(data['municipality'].unique()))
        sim_year = st.selectbox("Year", options=sorted(data['year'].unique()))
        sim_incumbent = st.checkbox("Previous Political Office?", value=False)
        sim_multi_election = st.checkbox("Ran in Multiple Elections (2025 Onwards)?", value=False, help="Starting in the 2025 municipal elections, candidates who ran in two races (i.e. municipal and welfare) submitted a joint financial statement. To isolate the effects this may have, this data was included in the regression analysis. More investigation is required to ensure the correct approach.")

        sim_no_incumbency_key = False if sim_incumbent else True # NOTE: Currently, the system uses a binary where no incumbency = True. Reversing user selection
        sim_num_candidates = (
            data.loc[(data['municipality'] == sim_municipality) & (data['year'] == sim_year), 'num_candidates_in_municipality']
            .iloc[0]  # Expect all num_candidates, given the same municipality and year, to be the same.
        )
        st.markdown("---")
        st.button("🔄 Run Simulation", type="primary")
    
    with col2:
        st.subheader("Predicted Outcome")
        
        # Make prediction
        predicted_vote = model.predict([[sim_spending,sim_num_candidates,sim_no_incumbency_key,sim_multi_election]])[0]
        
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

        filtered_data = data[
            (data['municipality'] == sim_municipality)
            ]

        # Add all data points
        fig.add_trace(go.Scatter(
            x=filtered_data['total_raised'],
            y=filtered_data['vote_prct'],
            mode='markers',
            marker=dict(color='lightgray', size=5, opacity=0.5),
            name='All Candidates',
            hoverinfo='skip'
        ))

        # Add simulated candidate
        fig.add_trace(go.Scatter(
            x=[sim_spending],
            y=[predicted_vote],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Scenario'
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
            title="All Candidates From Municipality",
            xaxis_title="Total Spending (€)",
            yaxis_title="Vote Share (%)",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find similar candidates
        st.subheader("Similar Historical Campaigns")
        similar = data[
            (data['municipality'] == sim_municipality) &
            (data['total_raised'].between(sim_spending * 0.8, sim_spending * 1.2))
        ].sort_values('total_raised')
        
        if len(similar) > 0:
            st.dataframe(
                similar[['full_name', 'year', "municipality", 'total_raised', 'vote_prct', 'anomaly_type']].head(10),
                use_container_width=True
            )
        else:
            st.warning("No similar campaigns found with these parameters.")
        
        # Efficiency score
        st.subheader("Efficiency Analysis")
        avg_similar_vote = similar['vote_prct'].mean() if len(similar) > 0 else predicted_vote
        
        st.markdown(f"""
        - **Your predicted vote share**: {predicted_vote:.1f}%
        - **Average for similar campaigns**: {avg_similar_vote:.1f}%
        """)

# -----------------------------
# PAGE 4: DEEP DIVE
# -----------------------------

elif page == "Methodology":
    st.title("Methodology")
    
    tab1, tab2 = st.tabs(["Model Diagnostics", "Methodology"])
    
    with tab1:
        st.subheader("Regression Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            r_squared = model.score(data[['total_raised', "num_candidates_in_municipality", "incumbency_None", "multi_election_Kyllä"]], data['vote_prct'])
            st.metric("R² Score", f"{r_squared:.4f}")
            st.metric("Intercept", f"{model.intercept_:.4f}")
            st.metric("Coefficient", f"{model.coef_[0]:.6f}")
            
            st.markdown(f"""
            **Interpretation**: 
            - Every €1,000 increase in total_raised is associated with a 
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
        all_candidate_description = {
            'count': 136320.0,
            'mean': 72.842972,
            'std': 221.544293,
            'min': 0.0,
            '25%': 16.0,
            '50%': 35.0,
            '75%': 72.0,
            'max': 29745.0
        } # Can preform df["Total number of votes"].describe() on municipal_election_results_by_candidate.csv in outputs to gather this.
        financial_candidate_description = data["Total number of votes"].describe().to_dict()


        st.markdown("""
        ### Data Collection
        - **Source**: Municipal election financial data provided by Valtiontalouden Tarkastusvirasto; Municipal election results from Oikeusministerio.
        - **Focus**: Candidates with reported financial data.
        
        ### Analysis Approach
        1. **Linear Regression Model**: Modeled vote share as a function of campaign total_raised alongside statistically relevant factors
           - Formula: `Vote Share = β₀ + β₁(Total Raised) + β₂(Number of Candidates in Municipal Election) + β₃(No Previous Office) + β₄(Reported for Two Elections) + ε`
        
        2. **Anomaly Detection**: Candidates are flagged as anomalies if their residual 
           (actual - predicted vote share) exceeds 2 standard deviations
           - **Overperformers**: Achieved >2 SD better results than predicted
           - **Underperformers**: Achieved >2 SD worse results than predicted
        
        3. **Other Explored Factors**: Factors considered but not included.
            - Age, Gender: Potential correlating variables that weren't included due to P values >0.14.
            - % Self Funding, breaking Total Funding by category: Self funding is by far the most prevalent form of funding, so I imagined that it could have statistical relevancy, but its P was = 0.8. There was simply to much noise for individualized funding categories (with a limited dataset, other potential correlatory variables such as party) that they weren't included, but it could be good for a greater regression equation.
            - Party, Municipality: Municipality has clear explanatory impact (Brought R^2 to 0.6), Party less so though still significant (extra 0.05 R^2). Number of candidates used as a proxy for municipality due to continuous regression, categorical was seen as a bit to noisy for this specific explanation but could be relevant later.
        
        ### Limitations
        - The dataset is ONLY for candidates who submitted financial information, which is only required from successful candidates. You can see through the box plot below how the std and means differ between all candidates and those who submitted financial information. Likely still relevancy in candidate success - more who didn't succeed submitted financial information then I expected - but define caveat that these results are biased to candidates who were more likely to be successful and about half of the vote was not included in this dataframe.
        - Correlation does not imply causation
        - Results vary by context (district type, election year, political climate)
        - Not necessarily a limitation, but overpreformers and underpreformers does not mean 'bad' or 'good'. A candidate could be an underpreformer because they spent the money for the party instead.
        """)

        fig = go.Figure()

        # All candidates
        fig.add_trace(go.Box(
            x=['All Candidates'],
            q1=[all_candidate_description['25%']],
            median=[all_candidate_description['50%']],
            q3=[all_candidate_description['75%']],
            lowerfence=[all_candidate_description['min']],
            upperfence=[all_candidate_description['max']],
            mean=[all_candidate_description['mean']],
            marker_color='#AEC6CF',
            boxmean='sd',
            name='All Candidates'
        ))

        # Financial candidates
        fig.add_trace(go.Box(
            x=['Candidates With Financial Data'],
            q1=[financial_candidate_description['25%']],
            median=[financial_candidate_description['50%']],
            q3=[financial_candidate_description['75%']],
            lowerfence=[financial_candidate_description['min']],
            upperfence=[financial_candidate_description['max']],
            mean=[financial_candidate_description['mean']],
            marker_color='#FFB347',
            boxmean='sd',
            name='Candidates W. Financial'
        ))

        fig.update_layout(
            title='Vote Distribution Comparison',
            yaxis_title='Number of Votes',
            yaxis=dict(range=[0, 600]),  # Limit y-axis to 5000
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # with tab3:
    #     st.subheader("Download Filtered Data")
    #
    #     # Allow users to select what to download
    #     download_anomalies = st.checkbox("Include only anomalies", value=False)
    #
    #     if download_anomalies:
    #         download_data = data[data['is_anomaly']]
    #     else:
    #         download_data = data
    #
    #     csv = download_data.to_csv(index=False)
    #
    #     st.download_button(
    #         label="Download CSV",
    #         data=csv,
    #         file_name="campaign_finance_analysis.csv",
    #         mime="text/csv"
    #     )
    #
    #     st.info(f"Ready to download: {len(download_data):,} records")
    #
    #     # Preview
    #     st.subheader("Data Preview")
    #     st.dataframe(download_data.head(100), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard analyzes campaign finance effectiveness across municipal elections.
The navigation above contains regression findings, a data explorer, and a deeper overview of the model.
""")