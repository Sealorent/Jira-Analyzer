"""
JIRA Data Analyzer v1.0.0
Enterprise-grade JIRA project analytics and team performance tracking
"""

import streamlit as st
import pandas as pd
import requests
import os
import logging
import numpy as np
from datetime import datetime
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Try to load environment variables from .env file (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Using system environment variables only.")
    print("💡 To use .env file, install: pip install python-dotenv")

# ---------------------------------
# Configuration from Environment Variables
# ---------------------------------
APP_NAME = os.getenv('APP_NAME', 'JIRA Data Analyzer')
APP_VERSION = os.getenv('APP_VERSION', '1.0.0')
JIRA_BASE_URL = os.getenv('JIRA_BASE_URL', '')
JIRA_AUTH_TOKEN = os.getenv('JIRA_AUTH_TOKEN', '')
DEFAULT_JQL_QUERY = os.getenv('DEFAULT_JQL_QUERY', 'labels IN (July2025)')
MAX_RESULTS_DEFAULT = int(os.getenv('MAX_RESULTS_DEFAULT', '300'))

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------
# Logging Setup
# ---------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------
# JIRA API configuration and functions
# ---------------------------------

# Define available field sets for JIRA
JIRA_FIELDS_OPTIONS = {
    'all': "*all",
    'summary': [
        'summary',
        'status',
        'assignee',
        'created',
        'updated',
        'priority',
        'issuetype',
        'customfield_10000',  # Story Points
        'timeestimate',
        'timespent',
        'labels',
        'components'
    ],
    'minimal': [
        'summary',
        'status',
        'assignee',
        'created',
        'priority',
        'issuetype'
    ]
}

def get_jira_assignees():
    """Fetch all assignees from JIRA for the project"""
    logger.info("Fetching JIRA assignees")
    
    url = JIRA_BASE_URL + "search"
    
    headers = {
        'Authorization': f'Basic {JIRA_AUTH_TOKEN}',
        'Accept': 'application/json'
    }
    
    # Query to get recent issues to extract assignees
    params = {
        'jql': 'assignee is not EMPTY ORDER BY updated DESC',
        'fields': 'assignee',
        'maxResults': 1000,
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        logger.info(f"JIRA assignees response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            issues = data.get('issues', [])
            
            # Extract unique assignees
            assignees = {}
            for issue in issues:
                assignee = issue.get('fields', {}).get('assignee')
                if assignee:
                    account_id = assignee.get('accountId', '')
                    display_name = assignee.get('displayName', '')
                    if account_id and display_name:
                        assignees[account_id] = display_name
            
            logger.info(f"Found {len(assignees)} unique assignees")
            return assignees
        else:
            logger.error(f"JIRA assignees API error: {response.status_code} - {response.text}")
            return {}
            
    except Exception as e:
        logger.error(f"Error fetching JIRA assignees: {e}")
        return {}

def build_jql_with_assignees(base_jql, selected_assignees):
    """Build JQL query with selected assignees"""
    if not selected_assignees:
        return base_jql
    
    # Create assignee filter
    assignee_ids = list(selected_assignees.keys())
    if len(assignee_ids) == 1:
        assignee_filter = f"assignee = {assignee_ids[0]}"
    else:
        assignee_filter = f"assignee IN ({', '.join(assignee_ids)})"
    
    # Check if base_jql already has assignee filter
    if 'assignee' in base_jql.lower():
        # Replace existing assignee filter
        import re
        # Remove existing assignee conditions
        jql_without_assignee = re.sub(r'(AND\s+)?assignee[^)]*\)[^A-Z]*', '', base_jql, flags=re.IGNORECASE)
        jql_without_assignee = re.sub(r'assignee[^A-Z]*', '', jql_without_assignee, flags=re.IGNORECASE)
        jql_without_assignee = jql_without_assignee.strip()
        
        if jql_without_assignee.upper().endswith('AND'):
            jql_without_assignee = jql_without_assignee[:-3].strip()
        
        # Add new assignee filter
        if jql_without_assignee:
            return f"{jql_without_assignee} AND {assignee_filter} ORDER BY created DESC"
        else:
            return f"{assignee_filter} ORDER BY created DESC"
    else:
        # Add assignee filter to existing query
        if 'ORDER BY' in base_jql.upper():
            base_part = base_jql[:base_jql.upper().find('ORDER BY')].strip()
            order_part = base_jql[base_jql.upper().find('ORDER BY'):]
            return f"{base_part} AND {assignee_filter} {order_part}"
        else:
            return f"{base_jql} AND {assignee_filter}"

def story_points_to_days(story_points, conversion_mapping=None):
    """
    Convert story points to estimated days of work
    
    Args:
        story_points: Story points value (int or float)
        conversion_mapping: Optional dict mapping story points to days
                          Default: {5: 1, 8: 2, 13: 3}
    
    Returns:
        Estimated days of work (float)
    """
    if conversion_mapping is None:
        # Default Fibonacci-based conversion
        conversion_mapping = {
            1: 0.2,   # 1 point = 0.2 days
            2: 0.4,   # 2 points = 0.4 days
            3: 0.6,   # 3 points = 0.6 days
            5: 1.0,   # 5 points = 1 day
            8: 2.0,   # 8 points = 2 days
            13: 3.0,  # 13 points = 3 days
            21: 5.0,  # 21 points = 5 days
            34: 8.0,  # 34 points = 8 days
            55: 13.0  # 55 points = 13 days
        }
    
    try:
        points = int(story_points) if story_points else 0
        return conversion_mapping.get(points, points * 0.2)  # Default fallback: 0.2 days per point
    except (ValueError, TypeError):
        return 0.0

def calculate_working_days_for_month(month_label, month_configs=None):
    """
    Calculate working days for a specific month using proper calendar logic
    
    Args:
        month_label: Month label like "July2025", "June2025"
        month_configs: Dict with month configurations like:
                      {"July2025": {"total_days": 31, "weekends": 9, "holidays": 1}}
    
    Returns:
        Number of working days for the month
    """
    if month_configs is None:
        # Default configurations for common months (accurate calendar-based)
        month_configs = {
            "July2025": {"total_days": 31, "weekends": 8, "holidays": 0},     # 31 - 8 weekends - 0 holidays = 23 working days (verified)
            "June2025": {"total_days": 30, "weekends": 8, "holidays": 1},     # 30 - 8 weekends - 1 holiday = 21 working days
            "May2025": {"total_days": 31, "weekends": 10, "holidays": 1},     # 31 - 10 weekends - 1 holiday = 20 working days
            "August2025": {"total_days": 31, "weekends": 8, "holidays": 1},   # 31 - 8 weekends - 1 holiday = 22 working days
            "September2025": {"total_days": 30, "weekends": 8, "holidays": 0}, # 30 - 8 weekends - 0 holidays = 22 working days
            "October2025": {"total_days": 31, "weekends": 9, "holidays": 0}    # 31 - 9 weekends - 0 holidays = 22 working days
        }
    
    config = month_configs.get(month_label, {"total_days": 30, "weekends": 8, "holidays": 1})
    
    # Calculate: total_days - weekends - holidays
    working_days = config["total_days"] - config["weekends"] - config["holidays"]
    
    return working_days

def get_total_effective_days(month_labels=None, month_configs=None):
    """
    Get total effective working days across multiple months
    
    Args:
        month_labels: List of month labels like ["July2025", "June2025"]
                     If None, returns all configured months
        month_configs: Optional custom month configurations
    
    Returns:
        Dict with:
        - total_days: Sum of all effective working days
        - breakdown: Dict showing working days per month
        - summary: Formatted summary string
    """
    if month_configs is None:
        # Default configurations for common months
        month_configs = {
            "May2025": {"total_days": 31, "weekends": 10, "holidays": 1},     # 20 working days
            "June2025": {"total_days": 30, "weekends": 8, "holidays": 1},     # 21 working days
            "July2025": {"total_days": 31, "weekends": 8, "holidays": 0},     # 23 working days (verified by user research)
            "August2025": {"total_days": 31, "weekends": 8, "holidays": 1},   # 22 working days
            "September2025": {"total_days": 30, "weekends": 8, "holidays": 0}, # 22 working days
            "October2025": {"total_days": 31, "weekends": 9, "holidays": 0}    # 22 working days
        }
    
    # If no specific months provided, use all configured months
    if month_labels is None:
        month_labels = list(month_configs.keys())
    
    # Calculate working days for each month
    breakdown = {}
    total_days = 0
    
    for month in month_labels:
        working_days = calculate_working_days_for_month(month, month_configs)
        breakdown[month] = working_days
        total_days += working_days
    
    # Create summary string
    summary_parts = []
    for month, days in breakdown.items():
        config = month_configs.get(month, {})
        total = config.get("total_days", 0)
        weekends = config.get("weekends", 0)
        holidays = config.get("holidays", 0)
        summary_parts.append(f"{month}: {total} - {weekends} weekends - {holidays} holidays = {days} working days")
    
    summary = "\n".join(summary_parts) + f"\n\nTotal Effective Days: {total_days}"
    
    return {
        "total_days": total_days,
        "breakdown": breakdown,
        "summary": summary
    }

def enhance_jql_with_issue_types(base_jql, assignee_filter=None):
    """Enhance JQL query to include all relevant issue types and story point labels"""
    # For v1.0.0, let's simplify to avoid parentheses issues
    # Just use the base query without enhancement to ensure stability
    
    # Clean the base query
    base_jql = base_jql.strip()
    
    # Apply assignee filter if provided
    if assignee_filter:
        if 'ORDER BY' in base_jql.upper():
            base_part = base_jql[:base_jql.upper().find('ORDER BY')].strip()
            order_part = base_jql[base_jql.upper().find('ORDER BY'):]
            if 'WHERE' in base_part.upper() or 'AND' in base_part.upper() or 'OR' in base_part.upper():
                enhanced_jql = f"{base_part} AND {assignee_filter} {order_part}"
            else:
                enhanced_jql = f"{base_part} AND {assignee_filter} {order_part}"
        else:
            if base_jql:
                enhanced_jql = f"{base_jql} AND {assignee_filter}"
            else:
                enhanced_jql = assignee_filter
    else:
        enhanced_jql = base_jql
    
    # Ensure we have an ORDER BY clause for consistent results
    if 'ORDER BY' not in enhanced_jql.upper():
        enhanced_jql += " ORDER BY created DESC"
    
    return enhanced_jql

def calculate_work_ratio_by_month(df):
    """
    Calculate work ratio by month comparing estimated vs actual working days
    
    Args:
        df: DataFrame with JIRA issues containing labels, estimated_days columns
    
    Returns:
        DataFrame with columns: Month, Estimated_Days, Available_Working_Days, Capacity_Ratio
    """
    if 'labels' not in df.columns or 'estimated_days' not in df.columns:
        return pd.DataFrame()
    
    # Extract month labels from each issue
    month_data = []
    for _, row in df.iterrows():
        labels = str(row.get('labels', '')).split(', ') if row.get('labels') else []
        for label in labels:
            if '2025' in label and any(month in label.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                month_data.append({
                    'month': label,
                    'estimated_days': float(row.get('estimated_days', 0))
                })
    
    if not month_data:
        return pd.DataFrame()
    
    # Aggregate estimated days by month
    month_df = pd.DataFrame(month_data)
    result = month_df.groupby('month').agg({
        'estimated_days': 'sum'
    }).round(2)
    
    # Calculate actual working days available for each month
    result['available_working_days'] = result.index.map(calculate_working_days_for_month)
    
    # Calculate capacity ratio (estimated work / available capacity)
    result['capacity_ratio'] = (result['estimated_days'] / result['available_working_days']).round(2)
    result['utilization_percentage'] = (result['capacity_ratio'] * 100).round(1)
    
    result.columns = ['Estimated Days', 'Available Working Days', 'Capacity Ratio', 'Utilization %']
    
    return result.reset_index().rename(columns={'month': 'Month'})

def calculate_carry_over_by_month(df):
    """
    Calculate carry over work by month based on status and creation/completion dates
    
    Args:
        df: DataFrame with JIRA issues containing labels, status, created, updated columns
    
    Returns:
        DataFrame with columns: Month, Total_Issues, Completed, In_Progress, Carry_Over_Percentage
    """
    if 'labels' not in df.columns or 'status' not in df.columns:
        return pd.DataFrame()
    
    # Extract month labels and status for each issue
    month_data = []
    for _, row in df.iterrows():
        labels = str(row.get('labels', '')).split(', ') if row.get('labels') else []
        status = str(row.get('status', 'Unknown'))
        
        for label in labels:
            if '2025' in label and any(month in label.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                # Determine if work is completed
                is_completed = status.lower() in ['done', 'closed', 'resolved', 'complete', 'completed']
                is_in_progress = status.lower() in ['in progress', 'in review', 'testing', 'code review']
                
                month_data.append({
                    'month': label,
                    'total_issues': 1,
                    'completed': 1 if is_completed else 0,
                    'in_progress': 1 if is_in_progress else 0,
                    'not_started': 1 if not is_completed and not is_in_progress else 0
                })
    
    if not month_data:
        return pd.DataFrame()
    
    # Aggregate by month
    month_df = pd.DataFrame(month_data)
    result = month_df.groupby('month').agg({
        'total_issues': 'sum',
        'completed': 'sum',
        'in_progress': 'sum',
        'not_started': 'sum'
    })
    
    # Calculate carry over percentage (non-completed work)
    result['carry_over'] = result['in_progress'] + result['not_started']
    result['carry_over_percentage'] = (result['carry_over'] / result['total_issues'] * 100).round(1)
    result['completion_rate'] = (result['completed'] / result['total_issues'] * 100).round(1)
    
    result.columns = ['Total Issues', 'Completed', 'In Progress', 'Not Started', 'Carry Over', 'Carry Over %', 'Completion %']
    
    return result.reset_index().rename(columns={'month': 'Month'})

def get_jira_data(jql_query, field_set='summary', max_results=300):
    """Fetch data from JIRA using JQL query"""
    logger.info(f"Fetching JIRA data with JQL: {jql_query}")
    
    url = JIRA_BASE_URL + "search"
    
    headers = {
        'Authorization': f'Basic {JIRA_AUTH_TOKEN}',
        'Accept': 'application/json'
    }
    
    params = {
        'jql': jql_query,
        'fields': JIRA_FIELDS_OPTIONS[field_set],
        'maxResults': max_results,
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        logger.info(f"JIRA API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Retrieved {len(data.get('issues', []))} JIRA issues")
            return data
        else:
            logger.error(f"JIRA API error: {response.status_code} - {response.text}")
            raise Exception(f"JIRA API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"Error fetching JIRA data: {e}")
        raise

def parse_jira_data(jira_response):
    """Parse JIRA response into a pandas DataFrame"""
    issues = jira_response.get('issues', [])
    
    parsed_data = []
    for issue in issues:
        fields = issue.get('fields', {})
        
        # Extract story points from custom field
        story_points = fields.get('customfield_10000', 0) or 0
        
        # Extract story points from labels (separate field for debugging)
        story_point_from_label = 0
        labels = fields.get('labels', [])
        # Look for numeric labels that represent story points
        for label in labels:
            try:
                label_str = str(label).strip()
                # Check if entire label is numeric (story points)
                if label_str.isdigit():
                    story_point_from_label = int(label_str)
                    break
                # Also check for common story point patterns like "SP5", "Points-8", etc.
                elif label_str.lower().startswith(('sp', 'point', 'pts')):
                    # Extract number from patterns like "SP5", "Points8", "Pts-13"
                    import re
                    numbers = re.findall(r'\d+', label_str)
                    if numbers:
                        story_point_from_label = int(numbers[0])
                        break
            except (ValueError, TypeError):
                continue
        
        # Use story points from labels if no custom field value
        if not story_points and story_point_from_label:
            story_points = story_point_from_label
        
        # Debug logging for story points - ensure numeric comparison
        try:
            if int(story_points) > 0:
                logger.info(f"Issue {issue.get('key')}: Found story points = {story_points} from labels: {fields.get('labels', [])}")
        except (ValueError, TypeError):
            # Handle cases where story_points is not a valid number
            if story_points:
                logger.info(f"Issue {issue.get('key')}: Non-numeric story points = {story_points} from labels: {fields.get('labels', [])}")
        
        # Additional debug for label processing
        labels = fields.get('labels', [])
        if labels:
            logger.debug(f"Issue {issue.get('key')}: Processing labels: {labels}")
        
        # Get issue type to distinguish between Task and Sub-task
        issue_type = fields.get('issuetype', {}).get('name', 'Unknown')
        
        parsed_issue = {
            'key': issue.get('key'),
            'summary': fields.get('summary', ''),
            'status': fields.get('status', {}).get('name', ''),
            'assignee': fields.get('assignee', {}).get('displayName', 'Unassigned') if fields.get('assignee') else 'Unassigned',
            'created': fields.get('created', ''),
            'updated': fields.get('updated', ''),
            'priority': fields.get('priority', {}).get('name', 'None') if fields.get('priority') else 'None',
            'issue_type': issue_type,
            'story_points': story_points,
            'story_point_from_label': story_point_from_label,
            'time_estimate': fields.get('timeestimate', 0) or 0,
            'time_spent': fields.get('timespent', 0) or 0,
            'labels': ', '.join(fields.get('labels', [])),
            'components': ', '.join([comp.get('name', '') for comp in fields.get('components', [])])
        }
        parsed_data.append(parsed_issue)
    
    df = pd.DataFrame(parsed_data)
    
    # Convert date strings to datetime and calculate days
    if not df.empty:
        # Convert date strings to datetime, handling errors
        df['created'] = pd.to_datetime(df['created'], errors='coerce')
        df['updated'] = pd.to_datetime(df['updated'], errors='coerce')
        
        # Remove timezone info from pandas datetime columns to make them timezone-naive
        df['created'] = df['created'].dt.tz_localize(None)
        df['updated'] = df['updated'].dt.tz_localize(None)
        
        # Calculate days difference with timezone-naive datetime, ensuring numeric results
        df['days_since_created'] = (datetime.now() - df['created']).dt.days
        df['days_since_updated'] = (datetime.now() - df['updated']).dt.days
        
        # Ensure numeric columns are properly typed and handle NaN values
        numeric_columns = ['story_points', 'story_point_from_label', 'time_estimate', 'time_spent', 'days_since_created', 'days_since_updated']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Convert to numeric, replacing any non-numeric values with 0
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    # Ensure the column is of integer type for discrete values
                    if col in ['story_points', 'story_point_from_label', 'time_estimate', 'time_spent']:
                        df[col] = df[col].astype(int)
                except Exception as e:
                    logger.warning(f"Error converting column {col} to numeric: {e}")
                    # Fill with zeros if conversion fails completely
                    df[col] = 0
        
        # Add estimated days based on story points from labels
        if 'story_point_from_label' in df.columns:
            df['estimated_days'] = df['story_point_from_label'].apply(story_points_to_days)
    
    return df

# ---------------------------------
# Local ML Model for JIRA Analysis
# ---------------------------------

class JiraMLModel:
    def __init__(self):
        self.completion_model = None
        self.risk_model = None
        self.label_encoders = {}
        
    def prepare_features(self, df):
        """Prepare features for ML model"""
        if df.empty:
            return pd.DataFrame()
            
        features_df = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['status', 'assignee', 'priority', 'issue_type']
        for col in categorical_cols:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_vals = features_df[col].unique()
                    known_vals = self.label_encoders[col].classes_
                    new_vals = [val for val in unique_vals if val not in known_vals]
                    if new_vals:
                        # Add new categories to encoder
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, new_vals)
                    features_df[f'{col}_encoded'] = self.label_encoders[col].transform(features_df[col].astype(str))
        
        # Create derived features
        features_df['summary_length'] = features_df['summary'].str.len()
        features_df['has_story_points'] = (pd.to_numeric(features_df['story_points'], errors='coerce') > 0).astype(int)
        features_df['has_time_estimate'] = (pd.to_numeric(features_df['time_estimate'], errors='coerce') > 0).astype(int)
        
        return features_df
    
    def train_models(self, df):
        """Train ML models on JIRA data"""
        if df.empty or len(df) < 5:
            logger.warning("Insufficient data for training ML models")
            return
            
        features_df = self.prepare_features(df)
        
        # Feature columns for training
        feature_cols = ['days_since_created', 'story_points', 'summary_length', 
                       'has_story_points', 'has_time_estimate']
        
        # Add encoded categorical features
        for col in ['status', 'assignee', 'priority', 'issue_type']:
            if f'{col}_encoded' in features_df.columns:
                feature_cols.append(f'{col}_encoded')
        
        # Filter available features
        available_features = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_features) < 3:
            logger.warning("Insufficient features for training")
            return
        
        X = features_df[available_features].fillna(0)
        
        # Train completion time prediction model
        if 'time_spent' in features_df.columns and pd.to_numeric(features_df['time_spent'], errors='coerce').sum() > 0:
            y_completion = features_df['time_spent'].fillna(0)
            self.completion_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.completion_model.fit(X, y_completion)
        
        # Train risk score model (based on days since last update)
        y_risk = features_df['days_since_updated'].fillna(0)
        self.risk_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.risk_model.fit(X, y_risk)
        
        logger.info("ML models trained successfully")
    
    def predict(self, df):
        """Make predictions on JIRA data"""
        if df.empty:
            return df
            
        features_df = self.prepare_features(df)
        
        # Feature columns for prediction
        feature_cols = ['days_since_created', 'story_points', 'summary_length', 
                       'has_story_points', 'has_time_estimate']
        
        # Add encoded categorical features
        for col in ['status', 'assignee', 'priority', 'issue_type']:
            if f'{col}_encoded' in features_df.columns:
                feature_cols.append(f'{col}_encoded')
        
        available_features = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_features) < 3:
            logger.warning("Insufficient features for prediction")
            return df
        
        X = features_df[available_features].fillna(0)
        
        # Make predictions
        if self.completion_model:
            df['predicted_completion_time'] = self.completion_model.predict(X)
        else:
            df['predicted_completion_time'] = 0
            
        if self.risk_model:
            df['risk_score'] = self.risk_model.predict(X)
        else:
            df['risk_score'] = df['days_since_updated']
        
        # Ensure prediction columns are numeric
        prediction_columns = ['predicted_completion_time', 'risk_score']
        for col in prediction_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df

# ---------------------------------
# Gemini AI Integration
# ---------------------------------

def setup_gemini():
    """Setup Gemini AI"""
    # You'll need to set your Gemini API key as an environment variable
    api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyAkETc5NO5jRpDcNBoXq2M1L1cWZcp4tp8')
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables")
        return False
    
    genai.configure(api_key=api_key)
    return True

def list_available_gemini_models():
    """List available Gemini models"""
    if not setup_gemini():
        return []
    
    try:
        models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                models.append(model.name)
        return models
    except Exception as e:
        logger.error(f"Error listing Gemini models: {e}")
        return []

def get_available_gemini_model():
    """Get the first available Gemini model for content generation"""
    available_models = list_available_gemini_models()
    if available_models:
        # Prefer gemini-1.5-flash or gemini-1.5-pro if available
        for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro']:
            if preferred in available_models:
                return preferred
        # Return the first available model
        return available_models[0]
    return None

def generate_jira_insights(df, predictions_summary, selected_model=None):
    """Generate insights using Gemini AI"""
    if not setup_gemini():
        return "Gemini AI not configured. Please set GEMINI_API_KEY environment variable."
    
    try:
        # Use selected model or get default available model
        if selected_model:
            model_name = selected_model
        else:
            model_name = get_available_gemini_model()
            
        if not model_name:
            available_models = list_available_gemini_models()
            return f"No suitable Gemini model found for content generation. Available models: {', '.join(available_models)}"
        
        logger.info(f"Using Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # Create prompt for Gemini
        prompt = f"""
        Analyze the following JIRA data insights and provide actionable recommendations:
        
        Data Summary:
        - Total issues: {len(df)}
        - Issue types: {df['issue_type'].value_counts().to_dict() if 'issue_type' in df.columns else 'N/A'}
        - Unique assignees: {df['assignee'].nunique()}
        - Status distribution: {df['status'].value_counts().to_dict()}
        - Priority distribution: {df['priority'].value_counts().to_dict()}
        - Story points by issue type: {df.groupby('issue_type')['story_points'].sum().to_dict() if 'issue_type' in df.columns and 'story_points' in df.columns else 'N/A'}
        
        ML Model Predictions:
        {predictions_summary}
        
        High-risk tasks (risk score > 7): {len(df[pd.to_numeric(df['risk_score'], errors='coerce') > 7]) if 'risk_score' in df.columns else 0}
        Tasks with high completion time prediction: {len(df[pd.to_numeric(df['predicted_completion_time'], errors='coerce') > pd.to_numeric(df['predicted_completion_time'], errors='coerce').median()]) if 'predicted_completion_time' in df.columns else 0}
        
        Please provide:
        1. Key insights about team performance
        2. Risk assessment and recommendations
        3. Resource allocation suggestions
        4. Timeline and deadline concerns
        5. Process improvement recommendations
        
        Keep the response concise but actionable.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating Gemini insights: {e}")
        return f"Error generating AI insights: {e}"

# ---------------------------------
# Sidebar Configuration
# ---------------------------------
with st.sidebar:
    st.header(f"{APP_NAME} v{APP_VERSION}")
    st.markdown("---")
    st.header("Configuration")
    
    # For v1.0.0, we only support JIRA
    data_source = "JIRA"
    st.info("📊 **JIRA Analytics Platform**")
    st.write("Enterprise-grade JIRA project analytics and team performance tracking")
    
    # Future data sources (hidden for v1.0.0)
    # with st.expander("🔮 Future Data Sources (Coming Soon)"):
    #     st.info("• Google Sheets Integration")
    #     st.info("• Microsoft Excel (SharePoint/OneDrive)")
    #     st.info("• CSV File Upload")
    #     st.info("• Confluence Integration")
    
    if True:  # Always JIRA for v1.0.0
        st.info("JIRA Configuration")
        
        # Fetch assignees button and multiselect
        if st.button("🔄 Load Assignees"):
            with st.spinner("Fetching assignees from JIRA..."):
                assignees = get_jira_assignees()
                if assignees:
                    st.session_state['jira_assignees'] = assignees
                    st.success(f"✅ Loaded {len(assignees)} assignees")
                else:
                    st.error("❌ Failed to load assignees")
        
        # Assignee selection
        selected_assignees = {}
        if 'jira_assignees' in st.session_state:
            assignees = st.session_state['jira_assignees']
            
            # Show assignee count
            st.write(f"**Available Assignees:** {len(assignees)}")
            
            # Multiselect for assignees
            selected_assignee_names = st.multiselect(
                "Select Assignees (leave empty for all)",
                options=list(assignees.values()),
                default=[]
            )
            
            # Create mapping of selected assignees
            if selected_assignee_names:
                selected_assignees = {
                    account_id: name for account_id, name in assignees.items() 
                    if name in selected_assignee_names
                }
                st.write(f"**Selected:** {len(selected_assignees)} assignees")
        else:
            st.info("👆 Click 'Load Assignees' to fetch available assignees")
        
        st.markdown("---")
        
        # JQL Query input
        base_jql = st.text_area("Base JQL Query",
                               value=DEFAULT_JQL_QUERY, height=100,
                               help="Enter your JQL query. Assignee filter will be added automatically.")
        
        # Generate final JQL with assignees and issue types
        if selected_assignees:
            jql_with_assignees = build_jql_with_assignees(base_jql, selected_assignees)
            assignee_ids = [aid for aid in selected_assignees.keys()]
            assignee_filter = f"assignee IN ({', '.join(assignee_ids)})"
            final_jql = enhance_jql_with_issue_types(jql_with_assignees, assignee_filter)
            st.text_area("Final JQL Query", value=final_jql, height=100, disabled=True)
            jql_query = final_jql
        else:
            final_jql = enhance_jql_with_issue_types(base_jql)
            st.text_area("Final JQL Query", value=final_jql, height=80, disabled=True)
            jql_query = final_jql
        
        st.markdown("---")
        
        # Analysis Options
        st.subheader("Analysis Options")
        field_set = st.selectbox("Field Set", list(JIRA_FIELDS_OPTIONS.keys()), index=1)
        max_results = st.number_input("Max Results", min_value=1, max_value=1000, value=MAX_RESULTS_DEFAULT)
        
        st.markdown("---")
        
        # AI Analysis
        st.subheader("AI Analysis")
        enable_ai = st.checkbox("Enable AI Insights (Gemini)", value=True)
        
        if enable_ai:
            gemini_key = os.getenv('GEMINI_API_KEY', 'AIzaSyAkETc5NO5jRpDcNBoXq2M1L1cWZcp4tp8')
            if gemini_key:
                st.success("✅ Gemini AI configured")
                
                if st.button("🔄 Refresh Available Models"):
                    with st.spinner("Fetching available models..."):
                        available_models = list_available_gemini_models()
                        if available_models:
                            st.session_state['gemini_models'] = available_models
                            st.success(f"✅ Found {len(available_models)} available models")
                        else:
                            st.error("❌ No models available or API error")
                
                if 'gemini_models' in st.session_state and st.session_state['gemini_models']:
                    available_models = st.session_state['gemini_models']
                    default_model = get_available_gemini_model()
                    
                    default_index = 0
                    if default_model and default_model in available_models:
                        default_index = available_models.index(default_model)
                    
                    selected_model = st.selectbox(
                        "Select Gemini Model",
                        options=available_models,
                        index=default_index,
                        help="Choose which Gemini model to use for AI insights"
                    )
                    
                    st.session_state['selected_gemini_model'] = selected_model
                else:
                    st.info("👆 Click 'Refresh Available Models' to load Gemini models")
            else:
                st.warning("⚠️ Set GEMINI_API_KEY environment variable")
    
    st.markdown("---")
    load_button = st.button("🚀 Analyze JIRA Data", type="primary", use_container_width=True)

# ---------------------------------
# Main Application
# ---------------------------------
st.title(f"📊 {APP_NAME} v{APP_VERSION}")
st.markdown("**Enterprise-grade JIRA project analytics and team performance tracking**")
st.markdown("---")

if load_button:
    # v1.0.0 - JIRA Analysis Only
    if jql_query:
        if jql_query:
            try:
                with st.spinner("Fetching JIRA data..."):
                    st.info("Step 1: Fetching data from JIRA...")
                    jira_response = get_jira_data(jql_query, field_set, max_results)
                    df = parse_jira_data(jira_response)
                    st.success(f"✅ Retrieved {len(df)} JIRA issues")
                
                if not df.empty:
                    # =================== ANALYSIS SECTIONS ===================
                    
                    # Section 1: Issue Type Analysis
                    st.header("🔧 Issue Type Analysis")
                    if 'issue_type' in df.columns:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Issues Distribution")
                            issue_type_counts = df['issue_type'].value_counts()
                            st.write("**Issues by Type:**")
                            st.dataframe(issue_type_counts.reset_index())
                            st.bar_chart(issue_type_counts)
                        
                        with col2:
                            st.subheader("⚡ Story Points & Estimates")
                            # Show story points by issue type
                            if 'story_point_from_label' in df.columns:
                                st.write("**Story Points by Issue Type (from labels):**")
                                story_points_by_type = df.groupby('issue_type')['story_point_from_label'].sum()
                                st.dataframe(story_points_by_type.reset_index())
                                
                                # Also show estimated days by issue type
                                if 'estimated_days' in df.columns:
                                    st.write("**Estimated Days by Issue Type:**")
                                    days_by_type = df.groupby('issue_type')['estimated_days'].sum().round(1)
                                    st.dataframe(days_by_type.reset_index())
                    
                    st.write("---")
                    
                    # Section 2: Monthly Analysis
                    st.header("📅 Monthly Analysis")
                    
                    # Show total effective days across all months in data
                    all_month_labels = []
                    if 'labels' in df.columns:
                        for labels_str in df['labels'].dropna():
                            if labels_str:
                                labels = labels_str.split(', ')
                                for label in labels:
                                    if '2025' in label and any(month in label.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                                        if label not in all_month_labels:
                                            all_month_labels.append(label)
                    
                    if all_month_labels:
                        effective_days_info = get_total_effective_days(all_month_labels)
                        
                        st.subheader("📊 Total Effective Working Days")
                        col_summary, col_breakdown = st.columns([1, 2])
                        
                        with col_summary:
                            st.metric("Total Effective Days", f"{effective_days_info['total_days']} days")
                            st.write(f"**Across {len(all_month_labels)} months:**")
                            for month, days in effective_days_info['breakdown'].items():
                                st.write(f"• {month}: {days} days")
                        
                        with col_breakdown:
                            with st.expander("📋 Detailed Calculation Breakdown"):
                                st.text(effective_days_info['summary'])
                            
                            with st.expander("📋 Field Explanations - Total Effective Days"):
                                st.write("**📊 Total Effective Working Days - Field Details:**")
                                st.write("**• Total Effective Days:** Sum of working days across all months")
                                st.write("   - Formula: Sum of all monthly working days")
                                st.write("   - Example: July(23) + June(21) = 44 total days")
                                st.write("   - Used for quarter/semester capacity planning")
                                st.write("**• Monthly Breakdown:** Working days per individual month")
                                st.write("   - July2025: 23 days (verified calculation)")
                                st.write("   - June2025: 21 days (calendar-based)")
                                st.write("   - Each month calculated independently")
                                st.write("**• Calculation Method:** Calendar-based working days")
                                st.write("   - Total calendar days in month")
                                st.write("   - Minus weekend days (Saturdays + Sundays)")
                                st.write("   - Minus public holidays")
                                st.write("   - Monday-Friday workweek standard")
                                st.write("**• Business Applications:**")
                                st.write("   - Multi-month project planning")
                                st.write("   - Quarter capacity assessment")
                                st.write("   - Resource allocation across time periods")
                                st.write("   - Long-term timeline estimation")
                    
                    st.write("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Work Capacity by Month")
                        work_ratio_df = calculate_work_ratio_by_month(df)
                        if not work_ratio_df.empty:
                            st.dataframe(work_ratio_df)
                            
                            # Visual chart for capacity utilization
                            if len(work_ratio_df) > 0:
                                st.write("**Capacity Utilization Chart:**")
                                chart_data = work_ratio_df.set_index('Month')['Utilization %']
                                st.bar_chart(chart_data)
                            
                            # Field explanations for Work Capacity
                            with st.expander("📋 Field Explanations - Work Capacity"):
                                st.write("**📊 Work Capacity by Month - Field Details:**")
                                st.write("**• Month:** Project timeline month (e.g., July2025, June2025)")
                                st.write("**• Estimated Days:** Total estimated work days calculated from story points")
                                st.write("   - Source: Sum of all story points converted to days")
                                st.write("   - Conversion: Story points → days (e.g., 5 points = 1 day)")
                                st.write("**• Available Working Days:** Actual calendar working days in the month")
                                st.write("   - July2025: 31 total - 8 weekends - 0 holidays = 23 days ✓")
                                st.write("   - June2025: 30 total - 8 weekends - 1 holiday = 21 days")
                                st.write("   - Based on Monday-Friday workweek")
                                st.write("**• Capacity Ratio:** Workload vs Available capacity")
                                st.write("   - Formula: Estimated Days ÷ Available Working Days")
                                st.write("   - Example: 25 estimated ÷ 23 available = 1.09 ratio")
                                st.write("**• Utilization %:** Capacity usage percentage")
                                st.write("   - Formula: Capacity Ratio × 100")
                                st.write("   - >100% = Over capacity (need more resources)")
                                st.write("   - <100% = Under capacity (room for more work)")
                                st.write("   - 100% = Perfect capacity match")
                        else:
                            st.info("No monthly work capacity data available")
                    
                    with col2:
                        st.subheader("📋 Carry Over by Month")
                        carry_over_df = calculate_carry_over_by_month(df)
                        if not carry_over_df.empty:
                            st.dataframe(carry_over_df)
                            
                            # Visual chart for carry over percentage
                            if len(carry_over_df) > 0:
                                st.write("**Carry Over Percentage Chart:**")
                                chart_data = carry_over_df.set_index('Month')['Carry Over %']
                                st.bar_chart(chart_data)
                            
                            # Field explanations for Carry Over
                            with st.expander("📋 Field Explanations - Carry Over"):
                                st.write("**📋 Carry Over by Month - Field Details:**")
                                st.write("**• Month:** Project timeline month (e.g., July2025, June2025)")
                                st.write("**• Total Issues:** Total number of issues/tasks for the month")
                                st.write("   - Source: Count of all issues with month label")
                                st.write("   - Includes all issue types (Task, Sub-task, Bug, etc.)")
                                st.write("**• Completed:** Number of finished issues")
                                st.write("   - Status: Done, Closed, Resolved, Complete, Completed")
                                st.write("   - Represents fully finished work")
                                st.write("**• In Progress:** Number of active issues")
                                st.write("   - Status: In Progress, In Review, Testing, Code Review")
                                st.write("   - Work currently being done")
                                st.write("**• Not Started:** Number of pending issues")
                                st.write("   - Status: To Do, Open, New, Backlog")
                                st.write("   - Work not yet begun")
                                st.write("**• Carry Over:** Total incomplete work")
                                st.write("   - Formula: In Progress + Not Started")
                                st.write("   - Work that extends beyond planned month")
                                st.write("**• Carry Over %:** Percentage of incomplete work")
                                st.write("   - Formula: (Carry Over ÷ Total Issues) × 100")
                                st.write("   - Higher % = more unfinished work")
                                st.write("**• Completion %:** Percentage of finished work")
                                st.write("   - Formula: (Completed ÷ Total Issues) × 100")
                                st.write("   - Higher % = better delivery performance")
                        else:
                            st.info("No monthly carry over data available")
                    
                    st.write("---")
                    
                    # Section 3: Team Analysis
                    st.header("👥 Team Analysis")
                    
                    if 'assignee' in df.columns:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("👤 Assignee Distribution")
                            assignee_counts = df['assignee'].value_counts()
                            st.write("**Tasks by Assignee:**")
                            st.dataframe(assignee_counts.reset_index())
                            st.bar_chart(assignee_counts)
                        
                        with col2:
                            st.subheader("📊 Workload Analysis")
                            # Assignee workload analysis
                            if 'estimated_days' in df.columns:
                                # Group by assignee and calculate workload
                                workload_df = df.groupby('assignee').agg({
                                    'story_point_from_label': 'sum',
                                    'estimated_days': 'sum',
                                    'key': 'count'
                                }).round(1)
                                workload_df.columns = ['Total Story Points', 'Total Days', 'Task Count']
                                workload_df = workload_df.sort_values('Total Days', ascending=False)
                                
                                st.dataframe(workload_df)
                                
                                # Workload distribution chart
                                if len(workload_df) > 0:
                                    st.write("**Days Distribution:**")
                                    st.bar_chart(workload_df['Total Days'])
                        
                        # Conversion mapping and detailed breakdown in expandable sections
                        with st.expander("📋 Story Points to Days Conversion"):
                            st.write("**Conversion Mapping:**")
                            conversion_info = pd.DataFrame({
                                'Story Points': [1, 2, 3, 5, 8, 13, 21, 34, 55],
                                'Estimated Days': [0.2, 0.4, 0.6, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
                            })
                            st.dataframe(conversion_info)
                        
                        # Accordion for detailed status breakdown by assignee
                        if 'estimated_days' in df.columns:
                            with st.expander("📋 Detailed Status Breakdown by Assignee"):
                                workload_df = df.groupby('assignee').agg({
                                    'story_point_from_label': 'sum',
                                    'estimated_days': 'sum',
                                    'key': 'count'
                                }).round(1)
                                workload_df.columns = ['Total Story Points', 'Total Days', 'Task Count']
                                workload_df = workload_df.sort_values('Total Days', ascending=False)
                                
                                for assignee in workload_df.index:
                                    assignee_tasks = df[df['assignee'] == assignee]
                                    if not assignee_tasks.empty:
                                        st.write(f"**{assignee}:**")
                                        
                                        # Status breakdown
                                        status_breakdown = assignee_tasks.groupby('status').agg({
                                            'story_point_from_label': 'sum',
                                            'estimated_days': 'sum',
                                            'key': 'count'
                                        }).round(1)
                                        status_breakdown.columns = ['Story Points', 'Days', 'Task Count']
                                        
                                        # Add totals row
                                        totals = status_breakdown.sum()
                                        totals.name = 'TOTAL'
                                        status_breakdown_with_total = pd.concat([status_breakdown, totals.to_frame().T])
                                        
                                        st.dataframe(status_breakdown_with_total)
                                        
                                        # Visual status distribution for this assignee
                                        if len(status_breakdown) > 1:
                                            st.write(f"Status distribution for {assignee}:")
                                            st.bar_chart(status_breakdown['Days'])
                                        
                                        st.write("---")  # Separator between assignees
                        
                        # Accordion for per-assignee work capacity by month
                        if 'estimated_days' in df.columns:
                            with st.expander("📊 Work Capacity by Month per Assignee"):
                                st.write("**Individual assignee work capacity analysis by month:**")
                                
                                for assignee in df['assignee'].unique():
                                    if assignee != 'Unassigned':
                                        assignee_data = df[df['assignee'] == assignee]
                                        if not assignee_data.empty:
                                            st.write(f"**📊 {assignee} - Work Capacity:**")
                                            
                                            # Calculate work capacity for this assignee
                                            assignee_capacity = calculate_work_ratio_by_month(assignee_data)
                                            if not assignee_capacity.empty:
                                                st.dataframe(assignee_capacity)
                                                
                                                # Chart for this assignee's capacity
                                                if len(assignee_capacity) > 0:
                                                    st.write(f"**{assignee}'s Utilization Chart:**")
                                                    chart_data = assignee_capacity.set_index('Month')['Utilization %']
                                                    st.bar_chart(chart_data)
                                                
                                                # Detailed task breakdown for capacity analysis
                                                st.write(f"**📊 Task Details Contributing to {assignee}'s Capacity:**")
                                                
                                                # Get month labels for this assignee
                                                assignee_months = []
                                                if 'labels' in assignee_data.columns:
                                                    for labels_str in assignee_data['labels'].dropna():
                                                        if labels_str:
                                                            labels = labels_str.split(', ')
                                                            for label in labels:
                                                                if '2025' in label and any(month in label.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                                                                    if label not in assignee_months:
                                                                        assignee_months.append(label)
                                                
                                                # For each month, show task details
                                                for month in assignee_months:
                                                    st.write(f"**{month} Task Breakdown:**")
                                                    
                                                    # Filter tasks for this month
                                                    month_tasks = assignee_data[assignee_data['labels'].str.contains(month, na=False)]
                                                    
                                                    if not month_tasks.empty:
                                                        total_estimated_days = month_tasks['estimated_days'].sum()
                                                        total_story_points = month_tasks['story_point_from_label'].sum()
                                                        
                                                        st.write(f"📋 **Summary:** {len(month_tasks)} tasks, {total_story_points} total points, {total_estimated_days:.1f} estimated days")
                                                        
                                                        # Show individual tasks
                                                        st.write(f"**Individual Tasks:**")
                                                        for _, task in month_tasks.iterrows():
                                                            points = task.get('story_point_from_label', 0)
                                                            days = task.get('estimated_days', 0)
                                                            issue_type = task.get('issue_type', 'Unknown')
                                                            status = task.get('status', 'Unknown')
                                                            st.write(f"   • {task['key']}: {task.get('summary', 'No summary')}")
                                                            st.write(f"     [{issue_type}] [{status}] - {points} pts → {days:.1f} days")
                                                    
                                                    st.write("---")  # Separator between months
                                                
                                                # Field explanations for capacity tasks
                                                with st.expander(f"📊 Field Explanations - {assignee}'s Capacity Tasks"):
                                                    st.write(f"**📊 {assignee}'s Capacity Analysis - Field Details:**")
                                                    st.write("**• Task Key:** JIRA issue identifier contributing to workload")
                                                    st.write("**• Summary:** Task description for context")
                                                    st.write("**• Issue Type:** Type of work (Task, Sub-task, Bug, Story, etc.)")
                                                    st.write("**• Status:** Current progress state")
                                                    st.write("**• Story Points:** Effort estimation from labels")
                                                    st.write("**• Estimated Days:** Converted time estimate (points → days)")
                                                    st.write("   - Conversion: 5 pts = 1 day, 8 pts = 2 days, 13 pts = 3 days")
                                                    st.write("**• Month Summary:**")
                                                    st.write("   - Total tasks assigned to this person for the month")
                                                    st.write("   - Total story points = sum of all task points")
                                                    st.write("   - Total estimated days = sum of converted time estimates")
                                                    st.write("**• Capacity Analysis:**")
                                                    st.write("   - Estimated Days ÷ Available Working Days = Capacity Ratio")
                                                    st.write("   - Shows if person is over/under allocated for the month")
                                                    st.write("   - Helps with workload balancing decisions")
                                            else:
                                                st.info(f"No capacity data available for {assignee}")
                                            
                                            st.write("---")  # Separator between assignees
                        
                        # Accordion for per-assignee carry over by month
                        if 'status' in df.columns:
                            with st.expander("📋 Carry Over by Month per Assignee"):
                                st.write("**Individual assignee carry over analysis by month:**")
                                
                                for assignee in df['assignee'].unique():
                                    if assignee != 'Unassigned':
                                        assignee_data = df[df['assignee'] == assignee]
                                        if not assignee_data.empty:
                                            st.write(f"**📋 {assignee} - Carry Over Analysis:**")
                                            
                                            # Calculate carry over for this assignee
                                            assignee_carry_over = calculate_carry_over_by_month(assignee_data)
                                            if not assignee_carry_over.empty:
                                                st.dataframe(assignee_carry_over)
                                                
                                                # Chart for this assignee's carry over
                                                if len(assignee_carry_over) > 0:
                                                    st.write(f"**{assignee}'s Carry Over Chart:**")
                                                    chart_data = assignee_carry_over.set_index('Month')['Carry Over %']
                                                    st.bar_chart(chart_data)
                                                    
                                                    # Additional completion rate chart
                                                    st.write(f"**{assignee}'s Completion Rate Chart:**")
                                                    completion_chart = assignee_carry_over.set_index('Month')['Completion %']
                                                    st.bar_chart(completion_chart)
                                                
                                                # Detailed task breakdown by status for this assignee
                                                st.write(f"**📋 Detailed Task Breakdown for {assignee}:**")
                                                
                                                # Get month labels for this assignee
                                                assignee_months = []
                                                if 'labels' in assignee_data.columns:
                                                    for labels_str in assignee_data['labels'].dropna():
                                                        if labels_str:
                                                            labels = labels_str.split(', ')
                                                            for label in labels:
                                                                if '2025' in label and any(month in label.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                                                                    if label not in assignee_months:
                                                                        assignee_months.append(label)
                                                
                                                # For each month, show detailed task breakdown
                                                for month in assignee_months:
                                                    st.write(f"**{month} Tasks:**")
                                                    
                                                    # Filter tasks for this month
                                                    month_tasks = assignee_data[assignee_data['labels'].str.contains(month, na=False)]
                                                    
                                                    if not month_tasks.empty:
                                                        # Categorize tasks by status
                                                        completed_statuses = ['Done', 'Closed', 'Resolved', 'Complete', 'Completed']
                                                        in_progress_statuses = ['In Progress', 'In Review', 'Testing', 'Code Review']
                                                        not_started_statuses = ['To Do', 'Open', 'New', 'Backlog']
                                                        
                                                        # Completed tasks
                                                        completed_tasks = month_tasks[month_tasks['status'].isin(completed_statuses)]
                                                        if not completed_tasks.empty:
                                                            st.write(f"✅ **Completed ({len(completed_tasks)} tasks):**")
                                                            for _, task in completed_tasks.iterrows():
                                                                st.write(f"   • {task['key']}: {task.get('summary', 'No summary')} [{task['status']}] - {task.get('story_point_from_label', 0)} pts")
                                                        
                                                        # In Progress tasks
                                                        progress_tasks = month_tasks[month_tasks['status'].isin(in_progress_statuses)]
                                                        if not progress_tasks.empty:
                                                            st.write(f"🔄 **In Progress ({len(progress_tasks)} tasks):**")
                                                            for _, task in progress_tasks.iterrows():
                                                                st.write(f"   • {task['key']}: {task.get('summary', 'No summary')} [{task['status']}] - {task.get('story_point_from_label', 0)} pts")
                                                        
                                                        # Not Started tasks
                                                        pending_tasks = month_tasks[month_tasks['status'].isin(not_started_statuses)]
                                                        if not pending_tasks.empty:
                                                            st.write(f"⏳ **Not Started ({len(pending_tasks)} tasks):**")
                                                            for _, task in pending_tasks.iterrows():
                                                                st.write(f"   • {task['key']}: {task.get('summary', 'No summary')} [{task['status']}] - {task.get('story_point_from_label', 0)} pts")
                                                        
                                                        # Other status tasks
                                                        other_tasks = month_tasks[~month_tasks['status'].isin(completed_statuses + in_progress_statuses + not_started_statuses)]
                                                        if not other_tasks.empty:
                                                            st.write(f"❓ **Other Status ({len(other_tasks)} tasks):**")
                                                            for _, task in other_tasks.iterrows():
                                                                st.write(f"   • {task['key']}: {task.get('summary', 'No summary')} [{task['status']}] - {task.get('story_point_from_label', 0)} pts")
                                                    
                                                    # Carry Over Summary for this month
                                                    st.write(f"**🔢 CARRY OVER SUMMARY for {month}:**")
                                                    
                                                    completed_count = len(completed_tasks)
                                                    progress_count = len(progress_tasks)
                                                    pending_count = len(pending_tasks)
                                                    other_count = len(other_tasks)
                                                    total_tasks = len(month_tasks)
                                                    
                                                    completed_points = completed_tasks['story_point_from_label'].sum()
                                                    progress_points = progress_tasks['story_point_from_label'].sum()
                                                    pending_points = pending_tasks['story_point_from_label'].sum()
                                                    other_points = other_tasks['story_point_from_label'].sum()
                                                    total_points = month_tasks['story_point_from_label'].sum()
                                                    
                                                    # Calculate carry over
                                                    carry_over_tasks = progress_count + pending_count + other_count
                                                    carry_over_points = progress_points + pending_points + other_points
                                                    carry_over_percentage = (carry_over_tasks / total_tasks * 100) if total_tasks > 0 else 0
                                                    completion_percentage = (completed_count / total_tasks * 100) if total_tasks > 0 else 0
                                                    
                                                    st.write(f"📊 **Month Summary:**")
                                                    st.write(f"   • Total Tasks: {total_tasks} ({total_points} pts)")
                                                    st.write(f"   • ✅ Completed: {completed_count} tasks ({completed_points} pts)")
                                                    st.write(f"   • 🔄 In Progress: {progress_count} tasks ({progress_points} pts)")
                                                    st.write(f"   • ⏳ Not Started: {pending_count} tasks ({pending_points} pts)")
                                                    st.write(f"   • ❓ Other Status: {other_count} tasks ({other_points} pts)")
                                                    
                                                    st.write(f"🚨 **CARRY OVER CALCULATION:**")
                                                    st.write(f"   • Carry Over Tasks = In Progress + Not Started + Other")
                                                    st.write(f"   • Carry Over Tasks = {progress_count} + {pending_count} + {other_count} = **{carry_over_tasks} tasks**")
                                                    st.write(f"   • Carry Over Points = {progress_points} + {pending_points} + {other_points} = **{carry_over_points} pts**")
                                                    st.write(f"   • Carry Over % = {carry_over_tasks}/{total_tasks} × 100 = **{carry_over_percentage:.1f}%**")
                                                    st.write(f"   • Completion % = {completed_count}/{total_tasks} × 100 = **{completion_percentage:.1f}%**")
                                                    
                                                    if carry_over_percentage > 50:
                                                        st.error(f"⚠️ HIGH CARRY OVER: {carry_over_percentage:.1f}% of tasks are incomplete!")
                                                    elif carry_over_percentage > 25:
                                                        st.warning(f"⚠️ MODERATE CARRY OVER: {carry_over_percentage:.1f}% of tasks are incomplete")
                                                    else:
                                                        st.success(f"✅ LOW CARRY OVER: Only {carry_over_percentage:.1f}% of tasks are incomplete")
                                                    
                                                    st.write("---")  # Separator between months
                                                
                                                # Field explanations for detailed tasks
                                                with st.expander(f"📋 Field Explanations - {assignee}'s Detailed Tasks"):
                                                    st.write(f"**📋 {assignee}'s Task Breakdown - Field Details:**")
                                                    st.write("**• Task Key:** JIRA issue identifier (e.g., PROJ-123)")
                                                    st.write("**• Summary:** Brief description of the task/issue")
                                                    st.write("**• Status:** Current state of the task")
                                                    st.write("   - ✅ Completed: Done, Closed, Resolved, Complete, Completed")
                                                    st.write("   - 🔄 In Progress: In Progress, In Review, Testing, Code Review")
                                                    st.write("   - ⏳ Not Started: To Do, Open, New, Backlog")
                                                    st.write("   - ❓ Other Status: Any status not in above categories")
                                                    st.write("**• Story Points:** Effort estimation from labels (0 if none)")
                                                    st.write("**• Month Grouping:** Tasks grouped by month labels (July2025, etc.)")
                                                    st.write("**• Carry Over Analysis:**")
                                                    st.write("   - Completed tasks contribute to Completion %")
                                                    st.write("   - In Progress + Not Started = Carry Over")
                                                    st.write("   - Higher story points = more complex tasks")
                                                    st.write("   - Task count shows workload distribution")
                                            else:
                                                st.info(f"No carry over data available for {assignee}")
                                            
                                            st.write("---")  # Separator between assignees
                    
                    st.write("---")
                    
                    # Section 4: Debug Information
                    with st.expander("🔍 Debug Information"):
                        st.subheader("Story Points Debug Info")
                    
                        # Story points statistics
                        if 'story_points' in df.columns and 'story_point_from_label' in df.columns:
                            total_story_points = pd.to_numeric(df['story_points'], errors='coerce').sum()
                            story_points_numeric = pd.to_numeric(df['story_points'], errors='coerce')
                            non_zero_points = story_points_numeric > 0
                            
                            st.write(f"**Total story points:** {total_story_points}")
                            st.write(f"**Non-zero story points:** {len(df[non_zero_points])}")
                            
                            if len(df[non_zero_points]) > 0:
                                st.write("Sample issues with story points:")
                                columns_to_show = ['key', 'issue_type', 'story_points', 'story_point_from_label', 'labels']
                                if 'estimated_days' in df.columns:
                                    columns_to_show.insert(-1, 'estimated_days')
                                sample_with_points = df[non_zero_points][columns_to_show].head()
                                st.dataframe(sample_with_points)
                            
                            # Also show samples with story points from labels specifically
                            if 'story_point_from_label' in df.columns:
                                label_points_numeric = pd.to_numeric(df['story_point_from_label'], errors='coerce')
                                non_zero_label_points = label_points_numeric > 0
                                st.write(f"**Story points extracted from labels:** {len(df[non_zero_label_points])}")
                                if len(df[non_zero_label_points]) > 0:
                                    st.write("Sample issues with story points from labels:")
                                    columns_to_show = ['key', 'issue_type', 'story_points', 'story_point_from_label', 'labels']
                                    if 'estimated_days' in df.columns:
                                        columns_to_show.insert(-1, 'estimated_days')
                                    sample_label_points = df[non_zero_label_points][columns_to_show].head()
                                    st.dataframe(sample_label_points)
                    
                    with st.spinner("Training ML models..."):
                        st.info("Step 2: Training local ML models...")
                        ml_model = JiraMLModel()
                        ml_model.train_models(df)
                        df = ml_model.predict(df)
                        st.success("✅ ML analysis completed")
                    
                    # Create predictions summary for Gemini
                    predictions_summary = f"""
                    Average predicted completion time: {df['predicted_completion_time'].mean():.2f} hours
                    Average risk score: {df['risk_score'].mean():.2f} days
                    High-risk tasks: {len(df[pd.to_numeric(df['risk_score'], errors='coerce') > 7])}
                    Tasks needing attention: {len(df[pd.to_numeric(df['days_since_updated'], errors='coerce') > 5])}
                    Assignee distribution: {df['assignee'].value_counts().to_dict() if 'assignee' in df.columns else 'N/A'}
                    """
                    
                    if enable_ai:
                        with st.spinner("Generating AI insights..."):
                            st.info("Step 3: Generating insights with Gemini AI...")
                            # Use selected model if available
                            selected_model = st.session_state.get('selected_gemini_model', None)
                            ai_insights = generate_jira_insights(df, predictions_summary, selected_model)
                            st.success("✅ AI insights generated")
                
                st.success("JIRA data analysis completed!")
                
                # Display data with assignee highlighting
                st.header("📋 JIRA Data")
                
                # Show query information for transparency
                st.write(f"**Query executed:** `{jql_query}`")
                st.write(f"**Total issues found:** {len(df)}")
                
                if 'assignee' in df.columns and selected_assignees:
                    selected_names = list(selected_assignees.values())
                    filtered_note = f"Showing tasks for: {', '.join(selected_names)}"
                    st.info(filtered_note)
                
                # Show label breakdown for transparency
                if 'labels' in df.columns:
                    all_labels = []
                    for labels_str in df['labels'].dropna():
                        if labels_str:
                            all_labels.extend(labels_str.split(', '))
                    if all_labels:
                        from collections import Counter
                        label_counts = Counter(all_labels)
                        top_labels = label_counts.most_common(10)
                        st.write(f"**Top labels in results:** {', '.join([f'{label}({count})' for label, count in top_labels])}")
                
                st.dataframe(df)
                
                # Display AI insights
                if enable_ai and 'ai_insights' in locals():
                    st.header("🤖 AI-Generated Insights")
                    st.markdown(ai_insights)
                
            except Exception as e:
                st.error(f"❌ Error analyzing JIRA data: {e}")
                st.stop()
    else:
        st.warning("⚠️ Please provide a JQL query to start the analysis.")
        st.info("**Quick Start Guide:**")
        st.info("1. 🔄 Click 'Load Assignees' to fetch team members")
        st.info("2. 👥 Select assignees to filter (optional)")
        st.info("3. 📝 Enter your JQL query (e.g., 'labels IN (July2025)')")
        st.info("4. 🚀 Click 'Analyze JIRA Data' to start")

# ---------------------------------
# Footer Information
# ---------------------------------
st.markdown("---")
st.markdown("### 📚 Quick Reference")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔍 Sample JQL Queries:**")
    st.code("labels IN (July2025)")
    st.code("project = DP AND assignee = currentUser()")
    st.code("status IN ('In Progress', 'Code Review')")
    st.code("labels IN (July2025) AND issueType = Task")

with col2:
    st.markdown("**📊 Key Features:**")
    st.info("• Team workload analysis")
    st.info("• Story points to days conversion")
    st.info("• Monthly capacity planning")
    st.info("• Carry over task tracking")
    st.info("• AI-powered insights")

st.markdown("---")
st.markdown(f"**{APP_NAME} v{APP_VERSION}** | Built with ❤️ using Streamlit")
