import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import plotly.express as px 
import plotly.graph_objects as go 
from dateutil.relativedelta import relativedelta
import shap

# Initialize session state variables
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Login"
    if 'last_login' not in st.session_state:
        st.session_state.last_login = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'login_attempted' not in st.session_state:
        st.session_state.login_attempted = False
    if 'hr_data' not in st.session_state:
        st.session_state.hr_data = None    

# Load data
def load_data():
    df_deployment = pd.read_csv('Deployment_Used_Extended_Dataset_with_Resignation.csv')
    df_performance = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')
    
    st.session_state.df_deployment = df_deployment
    st.session_state.df_performance = df_performance
    
    return df_deployment, df_performance 

# =================================================================================================================================================================== #
# Functions for Both HR and Talents # 

# Login Page
def login_page():
    st.title("Employee Churn Analytics Dashboard Login")
    st.write("Please enter your credentials to access the HR analytics dashboard")
    
    with st.form("login_form"): 
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            st.write(f"Entered username: '{username}', password: '{password}'")
            
            username = username.strip()
            password = password.strip()
            
            # Check credentials
            if username == "hr" and password == "ilovetowork":  
                st.session_state.logged_in = True
                st.session_state.role = "HR"
                st.session_state.current_page = "HR Dashboard"
                st.session_state.last_login = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.hr_data = load_data()  
                st.success("HR login successful! Loading data...")
                st.rerun()
            
            elif username == "talent" and password == "ilovetoworkalso":
                st.session_state.logged_in = True
                st.session_state.role = "Talent"   
                st.session_state.current_page = "Talent Insights"
                st.session_state.last_login = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.hr_data = load_data()  
                st.success("Talent Acquisition login successful! Loading data...")
                st.rerun()  
            
            else:
                st.error("Invalid username or password")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#  Navigation bar 
def create_navbar():
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Role**: {st.session_state.role}")
    
    # Common pages for all roles
    pages = {
        "Predict Employee Churn": "Predict Employee Churn"
    }
    
    # HR-specific pages
    if st.session_state.role == "HR":
        pages.update({
            "HR Dashboard": "HR Dashboard",
            "Employee Performance": "Overview of Employee Performance",
            "Actionable Insights": "Actionable Insights for Retention and Productivity"
        })
    
    # Talent-specific pages
    elif st.session_state.role == "Talent":
        pages.update({
            "Talent Insights": "Talent Insights",
            "Hiring Recommendations": "Hiring Recommendations"
        })
    
    # Navigation buttons
    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
            st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.current_page = "Login"
        st.session_state.role = None
        st.rerun()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Functions for HR and Talent  #

# Total Employees excluding resigned employees
def get_total_employees(df):
    active_employees = df[df['Resigned'] == False]
    return active_employees.shape[0]

# Churn rate for current quarter and previous quarter
def get_churn_rate(df, start_date, end_date):
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], dayfirst=True)
    df['Resignation_Date'] = pd.to_datetime(df['Resignation_Date'], errors='coerce')

    # Employees who resigned in this quarter
    resigned_in_period = df[(df['Resignation_Date'] >= start_date) & (df['Resignation_Date'] <= end_date)]
    active_employees_at_start = df[(df['Hire_Date'] < start_date) & (df['Resigned'] == False)]

    if active_employees_at_start.shape[0] == 0:
        return 0.0

    # Churn rate (Resigned employees / Active employees at the current quarter) 
    churn_rate = (resigned_in_period.shape[0] / active_employees_at_start.shape[0]) * 100
    return churn_rate

# Get the start and end dates of the current and previous quarters
def get_quarter_dates(year, quarter):
    """Returns (start_date, end_date) for given year and quarter (1-4)"""
    if quarter == 1:
        return pd.to_datetime(f"{year}-01-01"), pd.to_datetime(f"{year}-03-31")
    elif quarter == 2:
        return pd.to_datetime(f"{year}-04-01"), pd.to_datetime(f"{year}-06-30")
    elif quarter == 3:
        return pd.to_datetime(f"{year}-07-01"), pd.to_datetime(f"{year}-09-30")
    else:
        return pd.to_datetime(f"{year}-10-01"), pd.to_datetime(f"{year}-12-31") 

# Get the current year and quarter
def get_current_quarter():
    today = pd.to_datetime('today')
    current_year = today.year
    current_month = today.month
    return current_year, (current_month - 1) // 3 + 1  

# Get the comparison quarters based on current date
def get_comparison_quarters():
    """Returns (q1_year, q1_num, q1_start, q1_end, q2_year, q2_num, q2_start, q2_end)"""
    current_year, current_q = get_current_quarter()
    
    if current_q >= 3:  # Compare Q1 vs Q2 of current year
        return (
            current_year, 1, *get_quarter_dates(current_year, 1),
            current_year, 2, *get_quarter_dates(current_year, 2)
        )
    else:  # Compare Q3 vs Q4 of previous year
        return (
            current_year-1, 3, *get_quarter_dates(current_year-1, 3),
            current_year-1, 4, *get_quarter_dates(current_year-1, 4)
        )

# Calculate hires in a given period
def get_hired_in_period(df, start_date, end_date):
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], format='%m/%d/%Y')  
    hired_in_period = df[(df['Hire_Date'] >= start_date) & (df['Hire_Date'] <= end_date)]
    return hired_in_period.shape[0]

# Calculate active employees at the end of a period
def get_active_employees_at_end_of_period(df, start_date, end_date):
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], dayfirst=True)
    df_period = df[(df['Hire_Date'] <= end_date)]  
    active_employees = df_period[df_period['Resigned'] == False]
    return active_employees.shape[0]

# Calculate average satisfaction for the previous quarter 
def calculate_average_satisfaction_last_quarter(df):
    last_quarter_cutoff = pd.to_datetime("2025-04-01")
    df_last_quarter = df[df['Hire_Date'] < last_quarter_cutoff]
    avg_satisfaction_last_quarter = df_last_quarter['Employee_Satisfaction_Score'].mean() if not df_last_quarter.empty else 0 
    return avg_satisfaction_last_quarter    

# Calculate average satisfaction for this quarter (all active employees)
def calculate_average_satisfaction_this_quarter(df):
    avg_satisfaction_this_quarter = df['Employee_Satisfaction_Score'].mean() if not df.empty else 0
    return avg_satisfaction_this_quarter

# Get quarterly metrics
def get_quarterly_metrics(df, year, quarter):
    """Returns all metrics for a given quarter"""
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')
    
    # Get the start and end date for the quarter
    q_start, q_end = get_quarter_dates(year, quarter)
    
    # Filter data for the given quarter
    quarter_data = df[
        (df['Hire_Date'] >= q_start) & 
        (df['Hire_Date'] <= q_end)
    ]
    
    active_employees = get_active_employees_at_end_of_period(df, q_start, q_end)
    hires = get_hired_in_period(df, q_start, q_end)
    churn = get_churn_rate(df, q_start, q_end)
    satisfaction = quarter_data['Employee_Satisfaction_Score'].mean()
    
    return {
        'employees': active_employees,
        'hires': hires,
        'churn': churn,
        'satisfaction': satisfaction if not pd.isna(satisfaction) else 0,
        'year': year,
        'quarter': quarter,
        'start': q_start,
        'end': q_end
    }

# Get previous quarter
def get_previous_quarter(year, quarter):
        if quarter == 1:
            return (year-1, 4)
        return (year, quarter-1)
    
# Get sorted quarter metrics for trends
def get_sorted_quarters_for_trends(df, current_year, current_quarter):
    """Returns sorted quarter metrics for trends"""
    
    # Generate last 8 quarters
    quarters = []
    year, quarter = current_year, current_quarter

    for _ in range(8):
        year, quarter = get_previous_quarter(year, quarter)
        if (year < current_year) or (year == current_year and quarter < current_quarter):
            quarters.append((year, quarter))
    
    # Sort the quarters properly in chronological order
    quarters.sort(key=lambda x: (x[0], x[1]))  

    quarter_metrics = []
    for year, quarter in quarters:
        metrics = get_quarterly_metrics(df, year, quarter)
        if metrics:
            quarter_metrics.append(metrics)
    
    return quarter_metrics

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
    # Employee Metrics Page  #

def display_employee_metrics(q1, q2):
    """Display employee metrics and return button state"""
    employee_change = ((q2['employees'] - q1['employees']) / q1['employees']) * 100 if q1['employees'] > 0 else 0
    st.metric(
        "Total Employees", 
        f"{q2['employees']:,}",
        delta=f"{employee_change:.1f}% from Q{q1['quarter']}",
        delta_color="normal" if employee_change >= 0 else "inverse"
    )
    return st.button("View Employee Details", key="employee_details")

def display_churn_metrics(q1, q2):
    """Display churn metrics and return button state"""
    churn_change = q2['churn'] - q1['churn']
    st.metric(
        "Churn Rate",
        f"{q2['churn']:.2f}%",
        delta=f"{churn_change:.2f}% from Q{q1['quarter']}",
        delta_color="normal" if churn_change >= 0 else "inverse"
    )
    return st.button("View Churn Details", key="churn_details")

def display_employee_details(df_deployment, q1, q2):
    """Show detailed employee information"""
    st.subheader(f"Employee Details – Q{q2['quarter']} {q2['year']}")
    hire_comparison = ((q2['hires'] - q1['hires']) / q1['hires']) * 100 if q1['hires'] > 0 else 0

    tab1, tab2 = st.tabs(["Hire Comparison", "Department Breakdown"])

    with tab1:
        st.markdown(f"**Employee Hire Comparison (Q{q1['quarter']} vs Q{q2['quarter']}):** {hire_comparison:.1f}%")
        comparison_data = pd.DataFrame({
            'Period': [f"Q{q1['quarter']} {q1['year']}", f"Q{q2['quarter']} {q2['year']}"],
            'Hires': [q1['hires'], q2['hires']]
        })
        
        fig = px.bar(
            comparison_data,
            x='Period',
            y='Hires',
            text='Hires',  
            color='Period', 
            color_discrete_sequence=['#1f77b4', '#ff7f0e'] 
        )
        
        fig.update_traces(
            textposition='outside',  
            textfont_size=14,
            marker_line_width=0  
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title=None,
            yaxis_title='Hires',
            plot_bgcolor='rgba(0,0,0,0)',  
            paper_bgcolor='rgba(0,0,0,0)',  
            margin=dict(t=30), 
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            yaxis=dict(
                gridcolor='rgba(211,211,211,0.3)'
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Current Employees per Department")
        current_emp = (
            df_deployment[df_deployment['Resigned'] == False]
            .groupby('Department')['Employee_ID']
            .count()
            .reset_index(name='Current Employees')
        )
        fig_emp = px.bar(
            current_emp.sort_values('Current Employees', ascending=False),
            x='Department',
            y='Current Employees',
            text='Current Employees',
        )
        fig_emp.update_traces(marker_color='#1f77b4', textposition='outside')
        fig_emp.update_layout(xaxis_title='', yaxis_title='Employees', showlegend=False)
        st.plotly_chart(fig_emp, use_container_width=True)


def display_churn_details(df_deployment, current_year, current_quarter):
    """Show detailed churn information"""
    st.subheader("Churn Details")

    tab1, tab2 = st.tabs(["Churn Rate Trend", "Resignations by Department"])

    with tab1:
        completed_quarters = []
        for i in range(6):
            q = current_quarter - 1 - i
            year = current_year
            if q <= 0:
                q += 4
                year -= 1
            completed_quarters.append({'year': year, 'quarter': q})

        quarter_metrics = []
        all_data = get_sorted_quarters_for_trends(df_deployment, current_year, current_quarter)
        for q in completed_quarters:
            match = next((item for item in all_data 
                        if item['year'] == q['year'] and item['quarter'] == q['quarter']), None)
            if match:
                quarter_metrics.append(match)

        if quarter_metrics:
            trend_df = pd.DataFrame({
                'Quarter': [f"{q['year']}'Q{q['quarter']}" for q in quarter_metrics],
                'Date': [pd.to_datetime(f"{q['year']}-{q['quarter']*3}-01") for q in quarter_metrics],
                'Churn Rate': [q['churn'] for q in quarter_metrics]
            }).sort_values('Date')

            fig = px.line(
                trend_df,
                x='Quarter',
                y='Churn Rate',
                range_y=[0, 5],
                markers=True,
                line_shape='linear',
                title='Churn Rate Trend'
            )
            fig.update_traces(
                line=dict(color='red', width=2),
                marker=dict(color='red', size=8, line=dict(color='white', width=1))
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
                xaxis=dict(tickangle=45),
                font=dict(color='white')
            )
            fig.add_trace(go.Scatter(
                x=trend_df['Quarter'],
                y=trend_df['Churn Rate'],
                fill='tozeroy',
                fillcolor='rgba(30,136,229,0.2)',  
                line=dict(width=0),
                showlegend=False
            ))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        today = pd.to_datetime("today")
        current_year = today.year
        current_quarter = (today.month - 1) // 3 + 1

        if current_quarter == 1:        
            last_qtr = 4
            last_year = current_year - 1
        else:                             
            last_qtr = current_quarter - 1
            last_year = current_year

        q_start = pd.to_datetime(f"{last_year}-{(last_qtr-1)*3 + 1}-01")
        q_end = pd.to_datetime(f"{last_year}-{last_qtr*3}-01") + pd.offsets.MonthEnd(1)

        resigned_qtr = df_deployment[
            (df_deployment['Resigned'] == True) &
            (pd.to_datetime(df_deployment['Resignation_Date'], errors='coerce') >= q_start) &
            (pd.to_datetime(df_deployment['Resignation_Date'], errors='coerce') <= q_end)
        ]

        resigned_df = (resigned_qtr
                    .groupby('Department')['Employee_ID']
                    .count()
                    .reset_index(name='Resigned Count'))

        st.markdown(f"#### Resigned Employees per Department – Q{last_qtr} {last_year}")

        fig_resign = px.bar(
            resigned_df.sort_values('Resigned Count', ascending=False),
            x='Department',
            y='Resigned Count',
            text='Resigned Count',
            title='Resignations by Department'
        )
        fig_resign.update_traces(marker_color='#1f77b4', textposition='outside')
        fig_resign.update_layout(xaxis_title='', yaxis_title='Resigned', showlegend=False)
        st.plotly_chart(fig_resign, use_container_width=True)

# Only HR #
def display_satisfaction_trend(
    df_deployment: pd.DataFrame,
    current_year: int,
    current_quarter: int
) -> None:
    """Display satisfaction trend visualization
    Args:
        df_deployment: DataFrame containing employee deployment data
        current_year: The current year as integer
        current_quarter: The current quarter (1-4)
    """
    st.subheader("Satisfaction Trend")

    # Get quarter data for trend analysis
    desired_quarters = []
    latest_complete_quarter = 2  
    latest_complete_year = 2025
    for i in range(6):  
        quarter_offset = i
        year = latest_complete_year - (quarter_offset // 4)
        quarter = latest_complete_quarter - (quarter_offset % 4)
        if quarter <= 0:
            quarter += 4
            year -= 1
        desired_quarters.append({'year': year, 'quarter': quarter})
    
    # Get matching quarter metrics
    quarter_metrics = []
    all_quarters = get_sorted_quarters_for_trends(df_deployment, current_year, current_quarter)
    for q in desired_quarters:
        matching_quarter = next(
            (item for item in all_quarters 
             if item['year'] == q['year'] and item['quarter'] == q['quarter']), 
            None
        )
        if matching_quarter:
            quarter_metrics.append(matching_quarter)
    
    if not quarter_metrics:
        st.warning("No satisfaction data available for the specified quarters")
        return

    # Prepare trend data
    quarter_metrics = sorted(quarter_metrics, key=lambda x: (x['year'], x['quarter']))
    trend_df = pd.DataFrame({
        'Quarter': [f"{q['year']}'Q{q['quarter']}" for q in quarter_metrics],
        'Satisfaction': [q['satisfaction'] for q in quarter_metrics],
        'Date': [q['end'] for q in quarter_metrics]  
    }).sort_values('Date')

    # Calculate average
    avg_satisfaction = trend_df['Satisfaction'].mean()

    # Create visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df['Quarter'],
        y=trend_df['Satisfaction'],
        mode='lines+markers',
        name='Satisfaction',
        line=dict(color='skyblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=trend_df['Quarter'],
        y=[avg_satisfaction] * len(trend_df),
        mode='lines',
        name='Average',
        line=dict(color='red', width=2, dash='dot')
    ))
    fig.update_layout(
        title='Satisfaction Trend',
        xaxis_title='Quarter',
        yaxis_title='Satisfaction Score',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show latest value
    latest = trend_df.iloc[-1]
    st.markdown(f"**Latest:** {latest['Satisfaction']:.1f} average statisfaction score in {latest['Quarter']}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Churn Prediction Page 
def churn_prediction():
    st.title("Employee Churn Prediction")
    st.write("""This tool helps predict whether an employee will leave the company.""")
    
    with st.expander("📖 Variable Explanations"):
        st.markdown("""   
        *Performance Score*: Employee's performance rating (1-5 scale)  
        *Satisfaction Score*: Employee's satisfaction with their job (1-5 scale)  
        *Monthly Salary*: Employee's monthly pay  
        *Work Hours*: Typical hours worked per week  
        *Overtime Hours*: Extra hours worked beyond the normal schedule per year  
        *Promotions*: Number of promotions received per year  
        *Remote Work Level*: The proportion of time spent working remotely (None/Low/High)  
        *Job Title*: Employee's role  
        *Health Condition Index*: Employee's health status score (1-10 scale)  
    """)

    if st.session_state.hr_data is None:
        st.error("HR data not loaded. Please login again.")
        return  
    
    with st.form("churn_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            performance_score = st.slider(
                "Performance Score (1-5)", 1, 5, 3,
                help="Employee's performance rating (1-5 scale)"
            )
            employee_satisfaction = st.slider(
                "Satisfaction Score (1-5)", 1, 5, 3,
                help="Employee's satisfaction with their job (1-5 scale)"
            )
            monthly_salary = st.number_input(
                "Monthly Salary ($)", min_value=1000, max_value=20000, value=5000,
                help="Employee's monthly pay in dollars"
            )
            work_hours_per_week = st.slider(
                "Work Hours/Week", 10, 80, 40,
                help="Typical hours worked per week"
            )
            
        with col2:
            overtime_hours = st.slider(
                "Overtime Hours/Year", 0, 60, 30,
                help="Extra hours worked beyond the normal schedule per year"
            )
            promotions = st.slider(
                "Promotions/Year", 0, 5, 3,
                help="Number of promotions received per year"
            )
            remote_work_level = st.selectbox(
                "Remote Work Level", ["None", "Low", "High"],
                help="Time spent working remotely (None/Low/High)  "
            )
            job_title = st.selectbox(
                "Job Title", ["Manager", "Technician", "Other"],
                help="Employee's role in the company"
            )
            health_index = st.slider(
                "Health Condition Index (1-10)", 1, 10, 5,
                help="Employee's health status score (1-10 scale)"
            )
        
        submitted = st.form_submit_button("Predict Churn Risk")
        
        if submitted:
            try:
                model = joblib.load('employee_churn_model.pkl')

                input_data = pd.DataFrame([[performance_score, 
                                            1 if remote_work_level == "Low" else 2 if remote_work_level == "High" else 0, 
                                            promotions,
                                            1 if job_title == "Technician" else 0,
                                            1 if job_title == "Manager" else 0,
                                            monthly_salary,
                                            work_hours_per_week,
                                            overtime_hours,
                                            employee_satisfaction,
                                            health_index]], 
                                           columns=['Performance_Score', 'Remote_Work_Level', 'Promotions', 
                                                    'Job_Title_Technician', 'Job_Title_Manager', 'Monthly_Salary', 
                                                    'Work_Hours_Per_Week', 'Overtime_Hours',
                                                    'Employee_Satisfaction_Score', 'Health_Index'])

                # Make prediction
                prediction = model.predict(input_data)
                probabilities = model.predict_proba(input_data)
                prob_churn = probabilities[0][1]
                
                st.subheader("Prediction Results")
                
                # Recommendations based on churn risk 
                if prob_churn > 0.7:
                    st.error(f"High Risk of Churn: {prob_churn:.1%}")
                    st.write("This employee has a high probability of leaving. Immediate action recommended to retain this employee.")
                    st.subheader("Recommended Actions:")
                    st.write("""
                    - Schedule 1:1 meeting to discuss concerns.
                    - Identify key issues affecting satisfaction.
                    - Review compensation and benefits.
                    - Consider career development opportunities.
                    - Assess and adjust workload balance if needed.
                    """)

                elif prob_churn > 0.4:
                    st.warning(f"Moderate Risk of Churn: {prob_churn:.1%}")
                    st.write("This employee shows some risk factors. Consider taking proactive measures to overcome potential concerns.")
                    st.subheader("Recommended Actions:")
                    st.write("""
                    - Check-in with the employee to understand their concerns.
                    - Provide feedback channels for open discussion.
                    - Monitor performance and satisfaction over time.
                    - Consider additional training or support.
                    """)

                else:
                    st.success(f"Low Risk of Churn: {prob_churn:.1%}")
                    st.write("This employee appears stable. Continue regular engagement.")
                    st.subheader("Recommended Actions:")
                    st.write("""
                    - Continue regular check-ins to maintain engagement.
                    - Ensure consistent career development opportunities.
                    - Watch for any signs of dissatisfaction.
                    """)

                # SHAP (Feature Importance)
                try:
                    explainer = shap.TreeExplainer(model)  
                    shap_values = explainer.shap_values(input_data)

                    feature_contributions = pd.DataFrame(shap_values[0], index=input_data.columns, columns=["SHAP Value"])
                    feature_contributions['Abs_SHAP Value'] = feature_contributions['SHAP Value'].abs()
                    feature_contributions = feature_contributions.sort_values(by='Abs_SHAP Value', ascending=False)

                    # Only select positive SHAP values (which indicate positive contribution to churn risk)
                    feature_contributions_positive = feature_contributions[feature_contributions['SHAP Value'] > 0]

                    fig = px.bar(
                        feature_contributions_positive.head(7), 
                        x='SHAP Value', 
                        y=feature_contributions_positive.head(7).index, 
                        title='Top Influencing Features Based on SHAP Values',
                        text=feature_contributions_positive.head(7)['Abs_SHAP Value'].round(1).astype(str) + '%',
                        color='SHAP Value',
                        color_continuous_scale='Blues',
                        orientation='h'
                    )

                    fig.update_traces(
                        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.2f}<br>Percentage: %{text}', 
                    )

                    fig.update_layout(
                        title={'font': {'size': 25, 'color': 'white'}, 'x': 0.5 , 'xanchor': 'center'},
                        xaxis_title='SHAP Value',
                        yaxis_title='Features',
                        font=dict(color='white'),
                        plot_bgcolor='rgba(0,0,0,0)',  
                        paper_bgcolor='rgba(0,0,0,0)',  
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error with SHAP explanation: {str(e)}")
                    st.write("Falling back to Feature Importance visualization...")

                    # Feature importance fallback (if SHAP fails)
                    st.markdown("---")
                    st.subheader("Key Factors Influencing This Prediction")
                    feature_importance = pd.Series(model.feature_importances_, index=input_data.columns)
                    top_features = feature_importance.sort_values(ascending=False).head(7)

                    # Calculate the percentage for each feature
                    total_importance = top_features.sum()
                    top_features_percentage = (top_features / total_importance) * 100

                    # Create the horizontal bar chart 
                    fig = px.bar(
                        x=top_features.values,
                        y=top_features.index,
                        labels={'x': 'Importance Score', 'y': 'Features'},
                        title='Top Influencing Factors',
                        text=top_features_percentage.round(1).astype(str) + '%', 
                        color=top_features.values,  
                        color_continuous_scale='Blues',
                        orientation='h' 
                    )

                    fig.update_traces(
                        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<br>Percentage: %{text}', 
                    )

                    fig.update_layout(
                        title={'font': {'size': 25, 'color': 'white'}, 'x': 0.5 , 'xanchor': 'center'},
                        xaxis_title='Importance Score',
                        yaxis_title='Features',
                        font=dict(color='white'),
                        plot_bgcolor='rgba(0,0,0,0)',  
                        paper_bgcolor='rgba(0,0,0,0)', 
                    )

                    # Show the plot
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.write("Please try again or contact support")

# =================================================================================================================================================================== #
# Only for HR Functions #

# Recent activity
def recent_activity():
    if 'df_deployment' not in st.session_state:
        st.warning("HR data not loaded. Please log in again.")
        return
    
    df = st.session_state.df_deployment

    required_columns = ['Employee_ID', 'Hire_Date', 'Job_Title', 
                        'Employee_Satisfaction_Score', 'Performance_Score',
                        'Years_At_Company', 'Resigned', 'Resignation_Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {', '.join(missing_columns)}")
        return

    activity_data = []

    # 1. New Hires (Top 5 most recent)
    recent_hires = df.sort_values(by='Hire_Date', ascending=False).head(5)
    for _, row in recent_hires.iterrows():
        satisfaction = row['Employee_Satisfaction_Score']
        performance = row['Performance_Score']
        years = row['Years_At_Company']

        # Priority based on satisfaction
        if satisfaction >= 4:
            priority = "Low"
        elif satisfaction == 3:
            priority = "Medium"
        else:
            priority = "High"

        activity_data.append({
            "Employee ID": str(row['Employee_ID']),
            "Date": pd.to_datetime(row['Hire_Date']).date(),
            "Job Title": row['Job_Title'],
            "Activity Status": "New Hire",
            "Performance Score": performance,
            "Employee Satisfaction": f"{satisfaction:.2f}",
            "Years At Company": years,
            "Priority": priority
        })

    # 2. Resignations (Top 5 most recent)
    recent_resignations = df[df['Resigned'] == True].sort_values(by='Resignation_Date', ascending=False).head(5)
    for _, row in recent_resignations.iterrows():
        satisfaction = row['Employee_Satisfaction_Score']
        performance = row['Performance_Score']
        years = row['Years_At_Company']

        # Refined logic for resignations
        if performance >= 4 or years >= 5:
            priority = "High"
        elif performance == 3:
            priority = "Medium"
        else:
            priority = "Low"

        activity_data.append({
            "Employee ID": str(row['Employee_ID']),
            "Date": pd.to_datetime(row['Resignation_Date']).date(),
            "Job Title": row['Job_Title'],
            "Activity Status": "Resignation",
            "Performance Score": performance,
            "Employee Satisfaction": f"{satisfaction:.2f}",
            "Years At Company": years,
            "Priority": priority
        })

    # Display section
    st.subheader("Recent Activity")

    if activity_data:
        activity_df = pd.DataFrame(activity_data)

        # Priority sorting order
        priority_order = pd.CategoricalDtype(['High', 'Medium', 'Low'], ordered=True)
        activity_df['Priority'] = activity_df['Priority'].astype(priority_order)

        # Sort
        activity_df = activity_df.sort_values(['Date', 'Priority'], ascending=[False, True])
        activity_df.index = np.arange(1, len(activity_df) + 1)

        color_map = {
        'High': 'background-color: rgba(255, 0, 0, 0.15); color: white;',     # very light red
        'Medium': 'background-color: rgba(255, 200, 0, 0.15); color: white;', # very light yellow
        'Low': 'background-color: rgba(0, 200, 0, 0.15); color: white;'       # very light green
    }

        def highlight_priority(val):
            return color_map.get(val, '')

        styled_df = activity_df.style.applymap(highlight_priority, subset=['Priority'])

        st.dataframe(styled_df, use_container_width=True)

    else:
        st.write("No recent activity found.")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# HR Dashboard Page (Home Page)
def hr_dashboard():
    if st.session_state.role != "HR":
        st.warning("You don't have permission to access this page")
        return
    
    if st.session_state.hr_data is None:
        st.error("HR data not loaded. Please login again.")
        return

    df_deployment, df_performance = st.session_state.hr_data
    
    today = pd.to_datetime('today')
    current_year = today.year
    current_quarter = (today.month - 1) // 3 + 1
    
    # Get previous quarters for comparison
    q2_year, q2_quarter = get_previous_quarter(current_year, current_quarter)
    q1_year, q1_quarter = get_previous_quarter(q2_year, q2_quarter)
    
    # Get quarterly metrics for comparison
    q1 = get_quarterly_metrics(df_deployment, q1_year, q1_quarter)
    q2 = get_quarterly_metrics(df_deployment, q2_year, q2_quarter)
    
    # Calculate satisfaction change
    satisfaction_change = q2['satisfaction'] - q1['satisfaction']
    satisfaction_change_rounded = round(satisfaction_change, 1)
    
    # Dashboard Metrics 
    col1, col2, col3 = st.columns(3)
    
    # Employee Metrics
    with col1:
        employee_details_button = display_employee_metrics(q1, q2)
    
    # Satisfaction Metrics
    with col2:
        st.metric(
            "Avg. Satisfaction", 
            f"{q2['satisfaction']:.1f}/10",
            delta=f"{satisfaction_change_rounded:.1f} from Q{q1['quarter']}"
        )

        satisfaction_details_button = st.button("View Satisfaction Details", key="satisfaction_details")

    # Churn Metrics
    with col3:
        churn_details_button = display_churn_metrics(q1, q2)
    
    st.markdown("---")
    
    if employee_details_button:
        display_employee_details(df_deployment, q1, q2)

    if satisfaction_details_button:
        display_satisfaction_trend(df_deployment, current_year, current_quarter)
    
    if churn_details_button:
        display_churn_details(df_deployment, current_year, current_quarter)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #                   
    # Quick actions
    st.subheader("Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button("Run Churn Prediction"):
            st.session_state.current_page = "Predict Employee Churn"
            st.rerun()
    with action_col2:
        if st.button("View Performance"):
            st.session_state.current_page = "Overview of Employee Performance"
            st.rerun()
    with action_col3:
        if st.button("Get Insights"):
            st.session_state.current_page = "Actionable Insights for Retention and Productivity"
            st.rerun()

    st.markdown("---")

    recent_activity()

# =================================================================================================================================================================== #
# HR Third Page #
# Employee Performance Overview 
# Create department comparison charts

def create_dept_chart(data, x_col, y_col, title, company_avg, line_color="red"):

    data['Pct_Diff'] = ((data[y_col] - company_avg) / company_avg * 100)
    
    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        text=[f"{val:.1f} ({pct:+.1f}%)" for val, pct in zip(data[y_col], data['Pct_Diff'])],
        color_discrete_sequence=["#4B78B1"]  
    )
    
    fig.update_traces(
        textposition="outside",
        textfont_size=10,
        marker_color="#157DEC",
        textangle=0,
        opacity=0.9  
    )
    
    fig.add_hline(
        y=company_avg, 
        line_dash="dash", 
        line_color=line_color,
        line_width=2,
        annotation_text=f"Company Avg: {company_avg:.1f}",
        annotation_position="top right",
        annotation_font=dict(size=12, color="black"),
        annotation_bgcolor="rgba(255,255,255,0.7)",  
        annotation_y=company_avg * 1.25  
    )
    
    y_max = max(data[y_col].max() * 1.5, company_avg * 1.5)  
    fig.update_layout(
        yaxis_range=[0, y_max],
        margin=dict(t=100, b=40),  
        plot_bgcolor="rgba(0,0,0,0)" 
    )
    return fig

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Employee Performance Page
def employee_performance():
    st.title("Employee Performance Overview")
    
    if 'df_deployment' not in st.session_state:
        st.warning("HR data not loaded. Please log in again.")
        return
    
    df = st.session_state.df_deployment 
    
    # Calculate company-wide averages
    company_avg_perf = df['Performance_Score'].mean()
    company_avg_satisfaction = df['Employee_Satisfaction_Score'].mean()
    company_avg_overtime = df['Overtime_Hours'].mean()
    
    # 1. Department Performance
    st.subheader("Department Performance")
    
    dept_performance = df.groupby('Department').agg(
        Avg_Performance=('Performance_Score', 'mean'),
        Avg_Satisfaction=('Employee_Satisfaction_Score', 'mean'),
        Avg_Overtime=('Overtime_Hours', 'mean'),
        Employee_Count=('Employee_ID', 'count')
    ).reset_index()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:  
        st.metric("Total Departments", len(dept_performance))
    with col2:
        st.metric("Highest Performing Dept", 
                 dept_performance.loc[dept_performance['Avg_Performance'].idxmax()]['Department'])
    with col3:
        st.metric("Most Satisfied Dept", 
                 dept_performance.loc[dept_performance['Avg_Satisfaction'].idxmax()]['Department'])
    
    # Department comparison charts
    tab1, tab2, tab3 = st.tabs(["Performance", "Satisfaction", "Overtime"])

    with tab1:
        fig = create_dept_chart(
            dept_performance, 
            'Department', 
            'Avg_Performance',
            'Average Performance Score by Department',
            company_avg_perf
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = create_dept_chart(
            dept_performance, 
            'Department', 
            'Avg_Satisfaction',
            'Average Satisfaction by Department',
            company_avg_satisfaction
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = create_dept_chart(
            dept_performance, 
            'Department', 
            'Avg_Overtime',
            'Average Overtime Hours by Department',
            company_avg_overtime
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # 2. Quarterly Trends
    st.subheader("Quarterly Trends")

    df['Quarter'] = df['Hire_Date'].dt.to_period('Q').astype(str)
    df['Quarter'] = df['Quarter'].apply(lambda x: f"{x[:4]}'Q{x[5:]}") 
    current_quarter = pd.to_datetime('today').to_period('Q')
    df['Quarter_Period'] = df['Quarter'].apply(lambda x: pd.to_datetime(x[:4] + '-' + x[5:]).to_period('Q'))
    df = df[df['Quarter_Period'] <= current_quarter]

    # Select the latest 6 quarters 
    latest_quarters = df['Quarter'].sort_values(ascending=False).unique()[:6]
    df_filtered = df[df['Quarter'].isin(latest_quarters)]

    quarterly_trends = df_filtered.groupby('Quarter').agg(
        Avg_Performance=('Performance_Score', 'mean'),
        Avg_Satisfaction=('Employee_Satisfaction_Score', 'mean'),
        Hire_Count=('Employee_ID', 'count')
    ).reset_index().sort_values('Quarter')

    fig = px.line(quarterly_trends, 
                x='Quarter', 
                y='Avg_Performance',
                title='Performance Score Trend (Last 6 Quarters)',
                markers=True,
                text=quarterly_trends['Avg_Performance'].round(2))

    fig.update_traces(textposition="top center",
                    line=dict(width=2),
                    marker=dict(size=10))

    fig.add_hline(y=company_avg_perf, line_dash="dash", line_color="red",
                annotation_text=f"Company Avg: {company_avg_perf:.1f}")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. Overtime Analysis
    st.subheader("Overtime Analysis")

    df['Overtime_Category'] = pd.cut(df['Overtime_Hours'],
                                    bins=[-1, 5, 10, 15, 20, 100],
                                    labels=['0-5 hrs', '6-10 hrs', '11-15 hrs', '16-20 hrs', '20+ hrs'])

    overtime_summary = df.groupby('Overtime_Category').agg(
        Avg_Performance=('Performance_Score', 'mean'),
        Employee_Count=('Employee_ID', 'count'),
        Avg_Satisfaction=('Employee_Satisfaction_Score', 'mean')
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(overtime_summary,
                    x='Overtime_Category',
                    y='Avg_Performance',
                    color='Avg_Performance',
                    title='Average Performance by Overtime Range',
                    text='Avg_Performance')  
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside') 
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(overtime_summary,
                    names='Overtime_Category',
                    values='Employee_Count',
                    title='Distribution of Employees by Overtime Hours')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # 4. Sick Days Analysis
    st.subheader("Sick Days Analysis")

    sick_by_dept = df.groupby('Department')['Sick_Days'].mean().reset_index()
    fig = px.bar(
        sick_by_dept,
        x='Department',
        y='Sick_Days',
        title='Average Sick Days by Department',
        color='Sick_Days',
        color_continuous_scale='Blues',
        text='Sick_Days'  
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')  
    fig.update_layout(yaxis_title="Average Sick Days")
    st.plotly_chart(fig, use_container_width=True)

    # Executive Report Section
    st.markdown("---")
    st.subheader("Overall Performance Report Summary Download")

    # Create summary data for report
    overtime_summary = df.groupby('Overtime_Hours').agg(
        Avg_Performance=('Performance_Score', 'mean'),
        Employee_Count=('Employee_ID', 'count')
    ).reset_index()

    # Calculate sick days metrics
    max_sick_dept = df.groupby('Department')['Sick_Days'].mean().idxmax()
    min_sick_dept = df.groupby('Department')['Sick_Days'].mean().idxmin()

    report_data = {
        "Department Performance": dept_performance,
        "Quarterly Trends": quarterly_trends,
        "Overtime Analysis": overtime_summary,
        "Performance Correlations": df[['Employee_Satisfaction_Score', 'Performance_Score', 'Department', 'Job_Title']],
        "Sick Days Analysis": df.groupby('Department').agg({
            'Sick_Days': 'mean',
            'Performance_Score': 'mean'
        }).reset_index()
    }

    # Convert to Excel
    def create_excel_report(data_dict):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        return output.getvalue()

    excel_report = create_excel_report(report_data)

    st.download_button(
        label="📊 Download Full Performance Report (Excel)",
        data=excel_report,
        file_name="employee_performance_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download the full performance report in Excel format."
    )

    # Text summary version
    text_report = f"""
    Employee Performance Report
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    === Key Metrics ===
    - Total Departments: {len(dept_performance)}
    - Highest Performing Department: {dept_performance.loc[dept_performance['Avg_Performance'].idxmax()]['Department']}
    - Most Satisfied Department: {dept_performance.loc[dept_performance['Avg_Satisfaction'].idxmax()]['Department']}

    === Overtime Insights ===
    - Highest performing overtime range: {overtime_summary.loc[overtime_summary['Avg_Performance'].idxmax()]['Overtime_Hours']}
    - Most common overtime range: {overtime_summary.loc[overtime_summary['Employee_Count'].idxmax()]['Overtime_Hours']}
    - Percentage in most common range: {overtime_summary['Employee_Count'].max()/overtime_summary['Employee_Count'].sum()*100:.1f}%

    === Sick Days Insights ===
    - Department with most sick days: {max_sick_dept}
    - Department with least sick days: {min_sick_dept}
    - Average sick days across company: {df['Sick_Days'].mean():.1f} days

    === Recommendations ===
    1. Investigate why {dept_performance.loc[dept_performance['Avg_Performance'].idxmax()]['Department']} performs best
    2. Address satisfaction in {dept_performance.loc[dept_performance['Avg_Satisfaction'].idxmin()]['Department']}
    3. Monitor employees with {overtime_summary.loc[overtime_summary['Employee_Count'].idxmax()]['Overtime_Hours']} overtime for burnout
    4. Investigate high sick days in {max_sick_dept} department
    5. Review wellness programs in departments with above-average sick days
    6. Analyze correlation between performance and sick days
    """
    
    st.download_button(
        label="📝 Download Quick Summary (Text)",
        data=text_report,
        file_name="performance_summary.txt",
        mime="text/plain",
        help="Download a quick summary of the performance report in text format."
    )

# =================================================================================================================================================================== #
# HR Fourth Page # 

# Generate dynamic insights based on real-time data
def generate_dynamic_insights(df, insight_type, churn_rate):

    high_performers = df[df['Performance_Score'] > 4]  
    low_satisfaction = df[df['Employee_Satisfaction_Score'] < 2.5] 
    high_overtime = df[df['Overtime_Hours'] > 25]  
    optimal_performance = df[df['Performance_Score'] == 5]  
    
    # Retention Insights
    if insight_type == "Retention":
        insights = []

        if churn_rate > 20:
            insights.append(f"High churn risk (churn rate: {churn_rate:.2f}%). Consider retention strategies.")
            insights.append("Suggested Actions: Consider improving engagement and career development programs, review compensation, and increase job satisfaction.")

        if len(high_performers) > 0:
            insights.append(f"High-performing employees: {len(high_performers)} found. Reward and retain them.")
            insights.append("Suggested Actions: Reward high performers with promotions, incentives, or leadership opportunities.")

        if len(low_satisfaction) > 0:
            insights.append(f"Employees with low satisfaction: {len(low_satisfaction)} need immediate attention.")
            insights.append("Suggested Actions: Schedule 1:1 meetings, understand their concerns, and offer support.")
        return insights
    
    # Productivity Insights
    elif insight_type == "Productivity":
        insights = []

        if len(high_overtime) > 0:
            insights.append(f"Employees working >25 overtime hours: {len(high_overtime)}. Monitor for burnout.")
            insights.append("Suggested Actions: Consider workload adjustments or introducing wellness programs.")

        if len(optimal_performance) > 0:
            insights.append(f"Optimal performers: {len(optimal_performance)} found. Foster leadership roles.")
            insights.append("Suggested Actions: Nurture these employees for future leadership roles or mentorship programs.")

        return insights
    
    # Engagement Insights
    elif insight_type == "Engagement":
        insights = []

        high_engaged = df[(df['Performance_Score'] > 4) & (df['Employee_Satisfaction_Score'] > 4)]  
        if len(high_engaged) > 0:
            insights.append(f"Highly engaged employees: {len(high_engaged)} found. Leverage them for mentorship roles.")
            insights.append("Suggested Actions: Leverage high-engagement employees for mentoring new hires or driving culture initiatives.")

        low_engaged = df[(df['Performance_Score'] <= 3) & (df['Employee_Satisfaction_Score'] <= 3)]  
        if len(low_engaged) > 0:
            insights.append(f"Low engagement risk: {len(low_engaged)} employees with both low satisfaction and performance.")
            insights.append("Suggested Actions: Address underlying issues, including performance coaching, or reassignment to more suitable roles.")

        return insights
    
    # Compensation Insights
    elif insight_type == "Compensation":
        insights = []

        underpaid_performers = df[(df['Performance_Score'] == 5) & (df['Monthly_Salary'] < df['Monthly_Salary'].median())]  
        if len(underpaid_performers) > 0:
            insights.append(f"{len(underpaid_performers)} high performers are underpaid. Consider salary adjustments.")
            insights.append("Suggested Actions: Ensure competitive compensation packages for high performers to avoid attrition.")
        

        low_salary_high_satisfaction = df[(df['Monthly_Salary'] < df['Monthly_Salary'].median()) & (df['Employee_Satisfaction_Score'] > 4)]  
        if len(low_salary_high_satisfaction) > 0:
            insights.append(f"{len(low_salary_high_satisfaction)} employees with low salary and high satisfaction found.")
            insights.append("Suggested Actions: Review salary adjustments to ensure satisfaction remains high.")
        
        return insights

    return []

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#  Actionable‑Insights Page
def actionable_insights():

    st.set_page_config(layout="centered")
    st.title("Actionable Insights for Retention and Productivity")

    # Insight Selector
    st.subheader("🔍 Insight Selector", anchor=False)
    insight_type = st.selectbox(
        "What type of insights would you like to view and analyse?",
        ["Compensation", "Productivity", "Retention", "Engagement"],
        index=1,
        help="Select the HR metric you want to analyze"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    generate_clicked = st.button("✨ Generate Insights")

    if "df_deployment" not in st.session_state:
        st.error("HR data not loaded. Please log in again.")
        return

    df = st.session_state.df_deployment
    dummy_churn = 8.6

    if generate_clicked:
        today = pd.to_datetime("today")
        current_year = today.year
        current_quarter = (today.month - 1) // 3 + 1
        start_date, end_date = get_quarter_dates(current_year, current_quarter)
        churn_rate = get_churn_rate(df, start_date, end_date)

        insights = generate_dynamic_insights(df, insight_type, churn_rate)
        st.subheader(f"{insight_type} Insights")

        if insights:
            insight_text = ""
            numbered = 1
            for i in range(0, len(insights), 2):
                main = insights[i]
                action = insights[i+1] if i+1 < len(insights) else None

                st.markdown(f"**{numbered}. {main}**")
                if action:
                    for prefix in ["Suggested Actions:", "Suggested Action:"]:
                        if action.startswith(prefix):
                            action = action.replace(prefix, "").strip()
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;👉 Suggested Action: {action}", unsafe_allow_html=True)

                    insight_text += f"{numbered}. {main}\n"
                    insight_text += f"    👉 Suggested Action: {action}\n"
                else:
                    insight_text += f"{numbered}. {main}\n"

                numbered += 1

        # Action Plan and Report
        st.markdown("---")
        st.subheader("Report Summary and Action Plan")

        action_plans = {
            "Retention": """Sample Retention Action Plan:
1. Identify at-risk employees (churn risk >40%)
2. Schedule stay interviews within 2 weeks
3. Create personalized development plans
4. Review compensation benchmarks
5. Establish mentorship pairings""",
            "Productivity": """Sample Productivity Action Plan:
1. Identify teams with overtime >50 hrs
2. Introduce flexible work schedules or task rotation
3. Launch wellness or fatigue-monitoring initiatives
4. Monitor impact of training programs on performance
5. Adjust workloads for balance and sustainability""",
            "Engagement": """Sample Engagement Action Plan:
1. Conduct anonymous engagement surveys
2. Launch mentorship or buddy programs
3. Recognize and reward contributions regularly
4. Organize quarterly engagement townhalls
5. Support career growth and learning pathways""",
            "Compensation": """Sample Compensation Action Plan:
1. Benchmark current salary against market rates
2. Identify underpaid high performers
3. Develop a transparent performance-based bonus system
4. Review compensation equity across roles and departments
5. Communicate compensation philosophy to employees"""
        }

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""Employee {insight_type} Insights Report
Generated on: {timestamp}

--- INSIGHTS ---
{insight_text}

--- ACTION PLAN ---
{action_plans.get(insight_type, "No action plan available.")}
"""

        st.download_button(
            label="📄 Download Full Insight & Action Plan Report",
            data=report,
            file_name=f"{insight_type.lower()}_insight_report.txt",
            mime="text/plain",
            help="Download the full insight and action plan report in text format."
        )

# =================================================================================================================================================================== #
# Talent Acquisition Page (Talent team only) (Home Page)
def talent_dashboard():
    if st.session_state.role != "Talent":
        st.warning("You don't have permission to access this page")
        return
    
    if 'df_deployment' not in st.session_state:
        st.warning("HR data not loaded. Please log in again.")
        return
    
    df_deployment = st.session_state.df_deployment
    
    # Convert date columns to datetime
    df_deployment['Hire_Date'] = pd.to_datetime(df_deployment['Hire_Date'], errors='coerce')
    df_deployment['Resignation_Date'] = pd.to_datetime(df_deployment['Resignation_Date'], errors='coerce')
    
    # Get quarter info
    today = pd.to_datetime('today')
    current_year = today.year
    current_quarter = (today.month - 1) // 3 + 1
    
    # Get previous quarters
    def get_previous_quarters(year, quarter, count):
        quarters = []
        for _ in range(count):
            if quarter == 1:
                quarter = 4
                year -= 1
            else:
                quarter -= 1
            quarters.append((year, quarter))
        return quarters
    
    # Get last 2 quarters for comparison
    previous_quarters = get_previous_quarters(current_year, current_quarter, 2)
    q2_year, q2_quarter = previous_quarters[0]
    q1_year, q1_quarter = previous_quarters[1]
    
    # Get metrics
    q1 = get_quarterly_metrics(df_deployment, q1_year, q1_quarter)
    q2 = get_quarterly_metrics(df_deployment, q2_year, q2_quarter)
    
    # Calculate Average Tenure
    def calculate_avg_tenure(df, quarter_end_date):
        active_employees = df[(df['Hire_Date'] <= quarter_end_date) & 
                            ((df['Resignation_Date'] > quarter_end_date) | (df['Resignation_Date'].isna()))]
        return active_employees['Years_At_Company'].mean() if not active_employees.empty else 0
    
    q1_avg_tenure = calculate_avg_tenure(df_deployment, q1['end'])
    q2_avg_tenure = calculate_avg_tenure(df_deployment, q2['end'])
    avg_tenure_change = q2_avg_tenure - q1_avg_tenure
    
    # Three main metrics at the top
    col1, col2, col3 = st.columns(3)
    
    with col1:
        employee_button = display_employee_metrics(q1, q2)
    
    with col2:
        st.metric(
            "Average Tenure", 
            f"{q2_avg_tenure:.1f} years",
            delta=f"{avg_tenure_change:.1f} years from Q{q1['quarter']}",
            delta_color="inverse" if avg_tenure_change >= 0 else "normal"
        )
        tenure_button = st.button("View Tenure Details", key="tenure_details")
    
    with col3:
        churn_button = display_churn_metrics(q1, q2)

    # Display detailed sections based on button clicks
    if employee_button:
        st.markdown("---")
        display_employee_details(df_deployment, q1, q2)
    
    if churn_button:
        st.markdown("---")
        display_churn_details(df_deployment, current_year, current_quarter)
    
    if tenure_button:
        st.markdown("---")
        st.subheader("Average Tenure Trend")
        
        # Get last 6 quarters
        last_six_quarters = get_previous_quarters(current_year, current_quarter, 6)
        quarters = []
        
        for year, quarter in last_six_quarters:
            quarter_data = get_quarterly_metrics(df_deployment, year, quarter)
            quarter_data['avg_tenure'] = calculate_avg_tenure(df_deployment, quarter_data['end'])
            quarters.append(quarter_data)
        
        # Sort chronologically
        quarters = sorted(quarters, key=lambda x: (x['year'], x['quarter']))
        
        trend_df = pd.DataFrame({
            'Quarter': [f"Q{q['quarter']} {q['year']}" for q in quarters],
            'Average Tenure': [q['avg_tenure'] for q in quarters],
            'Date': [q['end'] for q in quarters]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df['Quarter'],
            y=trend_df['Average Tenure'],
            mode='lines+markers',
            name='Average Tenure',
            line=dict(color='skyblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=trend_df['Quarter'],
            y=[trend_df['Average Tenure'].mean()] * len(trend_df),
            mode='lines',
            name='Average',
            line=dict(color='red', width=2, dash='dot'),  
            hoverlabel=dict(font=dict(color='black')),  
            hoverinfo='y+name'
        ))
        fig.update_layout(
            title='Average Tenure Trend (Last 6 Quarters)',
            xaxis_title='Quarter',
            yaxis_title='Tenure (years)',
            height=400,
            autosize=True,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quick navigation to hiring recommendations
    st.markdown("---")
    if st.button("Get Hiring Recommendations", key="go_to_recommendations"):
        st.session_state.current_page = "Hiring Recommendations"
        st.rerun()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #\
# Second Page - Churn Risk Prediction # 
# Talents Third Page #
                                       
def hiring_recommendations():
    if st.session_state.role != "Talent":
        st.warning("You don't have permission to access this page")
        return
        
    st.title("Hiring Recommendations")
    
    if 'df_deployment' not in st.session_state:
        st.warning("HR data not loaded. Please log in again.")
        return
    
    df = st.session_state.df_deployment
    st.subheader("Education and Experience Distribution of Top Performers")
    
    successful_employees = df[
        (df['Performance_Score'] >= 4) & 
        (df['Employee_Satisfaction_Score'] >= 4) & 
        (df['Resigned'] == False)
    ]
    
    if not successful_employees.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            education_dist = successful_employees['Education_Level'].value_counts()
            fig1 = px.pie(
                education_dist,
                names=education_dist.index,
                values=education_dist.values,
                title='Education Distribution'
            )
            st.plotly_chart(fig1, use_container_width=True, key=f"education_chart_{str(np.random.randint(100000))}")
        
        with col2:
            fig2 = px.box(
                successful_employees,
                y='Years_At_Company',
                title='Experience Distribution of Top Performers',
                labels={'Years_At_Company': 'Years of Experience When Hired'}
            )
            fig2.update_traces(
                hoverinfo='y',
                hovertemplate='<b>%{y:.1f} years</b>',
                boxpoints=False
            )
            q1, med, q3 = np.percentile(successful_employees['Years_At_Company'], [25, 50, 75])
            fig2.add_annotation(
                x=0.5, y=med,
                text=f"<b>Median: {med:.1f} years</b>",
                showarrow=False,
                yshift=10
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"experience_chart_{str(np.random.randint(100000))}")
    
    # Hiring priority recommendations
    st.markdown("---")
    st.subheader("Priority Hiring Recommendations")
    
    # Get departments with highest churn
    resigned = df[df['Resigned'] == True]
    churn_by_dept = resigned.groupby('Department').size().reset_index(name='Resignations')
    total_by_dept = df.groupby('Department').size().reset_index(name='Total')
    churn_data = pd.merge(churn_by_dept, total_by_dept, on='Department')
    churn_data['Churn Rate'] = (churn_data['Resignations'] / churn_data['Total']) * 100
    
    high_churn_depts = churn_data.sort_values('Churn Rate', ascending=False).head(3)
    
    for _, row in high_churn_depts.iterrows():
        with st.expander(f"{row['Department']} Department (Churn Rate: {row['Churn Rate']:.1f}%)"):
            st.write(f"**Recommended Actions:**")
            st.write(f"- Prioritize hiring for {row['Department']} roles")
            st.write(f"- Target candidates with 2+ years experience in similar roles")
            st.write(f"- Look for evidence of stability in previous positions")
            st.write(f"- Consider offering competitive benefits for these roles")

    # Downloadable report
    st.markdown("---")
    st.subheader("Generate Hiring Strategy Report")
    
    report_text = f"""
    Hiring Strategy Recommendations
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    --- Priority Departments ---
    {high_churn_depts[['Department', 'Churn Rate']].to_string(index=False)}
    
    ---  Education and Experience Distribution of Top Performers ---
    Education: {education_dist.index[0]} ({(education_dist.values[0]/len(successful_employees)*100):.1f}%)
    Average Experience When Hired: {successful_employees['Years_At_Company'].mean():.1f} years
    
    --- Recommended Actions ---
    1. Prioritize hiring for {high_churn_depts.iloc[0]['Department']}
    2. Target candidates with 2+ years experience
    3. Implement skills testing for technical roles
    """
    
    st.download_button(
        label="📄 Download Hiring Strategy Report",
        data=report_text,
        file_name="hiring_recommendations.txt",
        mime="text/plain",
        key=f"download_{str(np.random.randint(100000))}"
    )

# =================================================================================================================================================================== #
def header():
    if st.session_state.role == "HR":
        st.title("HR Analytics Dashboard")
        st.write(f"Last login: {st.session_state.last_login}")
    elif st.session_state.role == "Talent":
        st.title("Talent Acquisition Dashboard")
        st.write(f"Last login: {st.session_state.last_login}")
    
    # Refresh button for both HR and Talent Acquisition
    if st.button("Refresh Data"):
        st.session_state.hr_data = load_data()
        st.success("Data refreshed!")
        st.rerun()

    st.markdown("---")

# MAIN FUNCTION
def main():
    init_session_state()
    
    st.set_page_config(
        page_title="Employee Churn Analytics Dashboard System",
        page_icon="📊",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main {padding-top: 1rem;}
    .stButton>button {border-radius: 8px;}
    .stAlert {border-radius: 8px;}
    </style>
    """, unsafe_allow_html=True)
    
    # Check authentication
    if not st.session_state.logged_in:
        login_page()
    else:
        create_navbar()
        header()
        
        # Page routing based on role
        if st.session_state.role == "HR":
            if st.session_state.current_page == "HR Dashboard":
                hr_dashboard()
            elif st.session_state.current_page == "Predict Employee Churn":
                churn_prediction()
            elif st.session_state.current_page == "Overview of Employee Performance":
                employee_performance()
            elif st.session_state.current_page == "Actionable Insights for Retention and Productivity":
                actionable_insights()

        
        elif st.session_state.role == "Talent":
            if st.session_state.current_page == "Talent Insights":
                talent_dashboard()
            elif st.session_state.current_page == "Hiring Recommendations":
                hiring_recommendations()
            elif st.session_state.current_page == "Predict Employee Churn":
                churn_prediction()

if __name__ == "__main__":
    main()

## cd "C:\Users\Man Wei\Documents\FYP\Yeong Man Wei-TP065870-APU3F2411CS(DA)-Source Code"
## streamlit run HR_Analytics_System..py

# Run the app with the following command in terminal with changing directory to the script location:
## streamlit run "C:\Users\Man Wei\Documents\FYP\fypData\HR_Analytics_System.py"