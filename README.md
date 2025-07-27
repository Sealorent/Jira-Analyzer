# 📊 JIRA Data Analyzer v1.0.0

Enterprise-grade JIRA project analytics and team performance tracking platform built with Streamlit and powered by AI insights using Google's Gemini AI.

## 🚀 Key Features (v1.0.0)

- **🎯 JIRA Integration**: Advanced JQL query support with automatic enhancement
- **👥 Multi-Assignee Analysis**: Team performance tracking and workload distribution
- **📊 Story Points Analytics**: Automatic extraction and days conversion
- **📅 Monthly Capacity Planning**: Working days calculation with calendar precision
- **📈 Carry Over Tracking**: Task completion monitoring across months
- **🤖 AI-Powered Insights**: Gemini AI for actionable recommendations
- **🔍 Individual Task Breakdown**: Detailed per-assignee task analysis
- **📋 Field Explanations**: Comprehensive documentation for every metric

## 📋 Prerequisites

- Python 3.8 or higher
- Active JIRA account with API access (pre-configured for one-employee.atlassian.net)
- (Optional) Google Gemini API key for AI insights

## 🛠️ Installation

1. **Clone or download the application**
   ```bash
   cd /home/lnk-mis30/streamlit-app
   ```

2. **Create and activate a virtual environment (Required for PEP 668 compliance)**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment (Linux/Mac)
   source venv/bin/activate
   
   # Activate virtual environment (Windows)
   # venv\Scripts\activate
   ```

3. **Install required Python packages**
   ```bash
   # Install all dependencies from requirements.txt
   pip install -r requirements.txt
   
   # Or install individually if needed:
   # pip install streamlit pandas gspread requests google-generativeai scikit-learn numpy openpyxl
   ```

## ⚙️ Configuration

### 1. JIRA Configuration (Already Set Up)

The application is pre-configured with JIRA credentials for `one-employee.atlassian.net`. No additional setup required for JIRA functionality.

### 2. Google Gemini AI (Optional but Recommended)

For AI-powered insights, set up your Gemini API key:

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the environment variable:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```

### 3. Future Integrations (Coming in v2.0.0)

v1.0.0 focuses exclusively on JIRA analytics. Future releases will include:
- **Microsoft Excel Integration**: SharePoint/OneDrive file analysis
- **Google Sheets Integration**: Direct spreadsheet analysis
- **CSV File Upload**: Local file analysis capabilities
- **Confluence Integration**: Documentation and requirements analysis

## 🏃‍♂️ Running the Application

1. **Navigate to the application directory**:
   ```bash
   cd /home/lnk-mis30/streamlit-app
   ```

2. **Set environment variables** (if using optional features):
   ```bash
   # For AI insights
   export GEMINI_API_KEY="your-gemini-api-key"
   
   # For Excel integration (if needed)
   export TENANT_ID="your-azure-tenant-id"
   export CLIENT_ID="your-azure-app-client-id"
   export CLIENT_SECRET="your-azure-app-client-secret"
   ```

3. **Run the Streamlit application**:
   ```bash
   # Make sure virtual environment is activated first
   source venv/bin/activate  # (Linux/Mac) or venv\Scripts\activate (Windows)
   
   # Run the application
   streamlit run app.py
   ```

4. **Access the application**:
   - Local: http://localhost:8501
   - Network: http://your-server-ip:8501

## 📖 Usage Guide (v1.0.0)

### Quick Start Workflow

1. **Load Team Members**:
   - Click "🔄 Load Assignees" to fetch all available team members
   - Select multiple assignees from the dropdown to filter tasks (optional)

2. **Configure Analysis**:
   - Enter your JQL query (e.g., `labels IN (July2025)`)
   - Choose field set: `all`, `summary`, or `minimal`
   - Set maximum results (1-1000)

3. **Enable AI Insights** (Optional):
   - Check "Enable AI Insights (Gemini)" if you have API key configured
   - Select preferred Gemini model from dropdown

4. **Analyze Data**:
   - Click "🚀 Analyze JIRA Data" to start the comprehensive analysis
   - Review enterprise-grade analytics across multiple sections:
     - 🔧 Issue Type Analysis
     - 📅 Monthly Analysis (Capacity & Carry Over)
     - 👥 Team Analysis (Individual Performance)
     - 🤖 AI-Generated Insights

### New v1.0.0 Analysis Sections

#### **📅 Monthly Analysis**
- **Total Effective Working Days**: Multi-month capacity planning
- **Work Capacity by Month**: Utilization percentages and over/under capacity detection
- **Carry Over by Month**: Task completion tracking and delivery performance

#### **👥 Team Analysis**
- **Individual Workload Analysis**: Story points, estimated days, and task distribution
- **Per-Assignee Capacity**: Monthly capacity analysis for each team member
- **Detailed Task Breakdown**: Individual JIRA issues categorized by completion status
- **Carry Over Summary**: Mathematical breakdown of incomplete work per person

#### **🔍 Advanced Features**
- **Field Explanations**: Expandable help for every metric and calculation
- **Task-Level Visibility**: See exact JIRA issues contributing to each metric
- **Calendar-Based Accuracy**: Verified working days calculations (July 2025 = 23 days)

### Sample JQL Queries

```sql
-- All tasks for specific labels
type = Task AND labels IN (May2025)

-- Tasks by status
type = Task AND status IN ("In Progress", "To Do")

-- Recent tasks
type = Task AND updated >= -30d

-- High priority tasks
type = Task AND priority = High

-- Tasks with story points
type = Task AND "Story Points" is not EMPTY
```

## 📊 Features Explained

### Machine Learning Predictions

The application trains local ML models to predict:
- **Completion Time**: Estimated hours to complete tasks
- **Risk Score**: Days since last update (higher = more at risk)

### AI Insights

With Gemini AI enabled, get actionable insights about:
- Team performance analysis
- Risk assessment and recommendations
- Resource allocation suggestions
- Timeline and deadline concerns
- Process improvement recommendations

## 📊 Analysis Functions Documentation

### Story Points to Days Conversion

The application includes a modular function to convert story points to estimated days:

```python
def story_points_to_days(story_points, conversion_mapping=None):
    """
    Convert story points to estimated days of work
    
    Default Fibonacci-based mapping:
    - 1 point = 0.2 days
    - 2 points = 0.4 days
    - 3 points = 0.6 days
    - 5 points = 1.0 days
    - 8 points = 2.0 days
    - 13 points = 3.0 days
    - 21 points = 5.0 days
    - 34 points = 8.0 days
    - 55 points = 13.0 days
    
    Args:
        story_points: Story points value (int or float)
        conversion_mapping: Optional custom mapping dict
    
    Returns:
        Estimated days of work (float)
    """
```

### Working Days Calculation

Calculates actual working days available per month using proper calendar logic:

```python
def calculate_working_days_for_month(month_label, month_configs=None):
    """
    Calculate working days for a specific month using proper calendar logic
    
    Default configurations (verified calendar-based):
    - July2025: 31 total days - 8 weekends - 0 holidays = 23 working days (verified)
    - June2025: 30 total days - 8 weekends - 1 holiday = 21 working days
    - May2025: 31 total days - 10 weekends - 1 holiday = 20 working days
    
    Logic:
    1. Count total days in the month (actual calendar days)
    2. Count weekend days (Saturdays + Sundays)
    3. Count holidays/free days
    4. Calculate: total_days - weekends - holidays
    
    Example July 2025 (verified by research):
    - 31 days total (actual calendar month)
    - 8 weekend days (Saturdays and Sundays)
    - 0 holidays (no public holidays in Indonesia for July 2025)
    - Result: 31 - 8 - 0 = 23 working days
    - Monday to Friday workweek standard
    
    Returns:
        Number of working days for the month
    """
```

### Total Effective Days Calculation

Gets total effective working days across multiple months:

```python
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
    
    Example Usage:
    - get_total_effective_days(["July2025", "June2025"])
    - Returns: {total_days: 42, breakdown: {"July2025": 21, "June2025": 21}}
    
    Business Value:
    - Quarter/semester capacity planning
    - Multi-month resource allocation
    - Long-term project timeline estimation
    """
```

### Monthly Work Capacity Analysis

Analyzes work capacity by comparing estimated workload vs available working days:

```python
def calculate_work_ratio_by_month(df):
    """
    Analyzes work capacity by comparing:
    - Estimated Days: From story points conversion
    - Available Working Days: Calculated from calendar and business rules
    - Capacity Ratio: Estimated days / Available working days
    - Utilization %: Capacity ratio × 100
    
    Logic:
    1. Extract month labels (e.g., "July2025", "June2025") from issue labels
    2. Sum estimated days per month (calculated from story points)
    3. Calculate available working days using calendar configuration
    4. Calculate capacity utilization (>100% = over capacity, <100% = under capacity)
    
    Returns:
        DataFrame with columns: Month, Estimated Days, Available Working Days, Capacity Ratio, Utilization %
    """
```

### Monthly Carry Over Analysis

Tracks work completion and carry over between months:

```python
def calculate_carry_over_by_month(df):
    """
    Analyzes work completion by month:
    - Total Issues: Count of issues per month
    - Completed: Issues with status "Done", "Closed", "Resolved", "Complete"
    - In Progress: Issues with status "In Progress", "In Review", "Testing", "Code Review"
    - Not Started: Remaining issues
    - Carry Over %: Percentage of non-completed work
    - Completion %: Percentage of completed work
    
    Logic:
    1. Extract month labels from each issue
    2. Categorize by status (completed/in-progress/not-started)
    3. Calculate percentages for project management insights
    4. Carry over = In Progress + Not Started
    
    Returns:
        DataFrame with columns: Month, Total Issues, Completed, In Progress, Not Started, Carry Over, Carry Over %, Completion %
    """
```

## 🔍 JIRA Query Enhancement Logic

The application automatically enhances JQL queries to include story point labels while respecting month filtering:

```python
def enhance_jql_with_issue_types(base_jql, assignee_filter=None):
    """
    Enhances user queries with story point labels while preventing cross-month contamination:
    
    Input: labels IN (July2025)
    Output: (labels IN (July2025)) OR (labels IN (1,2,3,5,8,13,21,34,55) AND labels IN (July2025))
    
    This ensures:
    1. Direct month matches are included
    2. Story point issues are included ONLY if they also have the same month label
    3. No cross-month contamination (July2025 query won't return June2025 issues)
    4. Assignee filters are applied to both parts of the OR condition
    
    Key Features:
    - Regex-based month detection with fallback for partial matches
    - Smart month label extraction (handles "July2025" not just "y2025")
    - Modular design for easy customization of story point values
    """
```

## 📈 Advanced Features

### Assignee Workload Analysis
- **Main Summary**: Total story points, estimated days, and task count per assignee
- **Status Breakdown Accordion**: Expandable detailed view of each assignee's task status distribution
- **Visual Charts**: Bar charts showing workload distribution and individual progress

### Monthly Insights
- **Work Ratio Analysis**: Compares planned vs actual effort by month
- **Carry Over Tracking**: Monitors incomplete work across monthly boundaries
- **Efficiency Metrics**: Identifies over/under estimation patterns

### Story Points Extraction
- **Label-Based Detection**: Extracts story points from numeric labels (1, 2, 3, 5, 8, 13, etc.)
- **Mixed Label Support**: Handles labels like ["5", "July2025"] correctly
- **Pattern Recognition**: Supports prefixed formats like "SP5", "Points8"
- **Debug Transparency**: Separate field shows extraction results for verification

### Assignee Analysis

- View task distribution across team members
- Filter by multiple assignees simultaneously
- See workload balance and statistics
- Identify team members with high-risk or overdue tasks

## 🔧 Troubleshooting

### Common Issues

1. **JIRA Connection Failed**:
   - Check internet connectivity
   - Verify JIRA credentials are valid
   - Ensure JQL query syntax is correct

2. **No Assignees Loaded**:
   - Check if there are issues with assignee data in JIRA
   - Try with a simpler JQL query first
   - Verify JIRA permissions

3. **AI Insights Not Working**:
   - Ensure `GEMINI_API_KEY` is set correctly
   - Check API key permissions and quota
   - Verify internet connectivity

4. **Excel Integration Issues**:
   - Verify all Azure environment variables are set
   - Check Azure app permissions
   - Ensure the Excel sharing URL is correct and accessible

### Logs and Debugging

The application provides detailed logging. Check the console output for:
- API response status codes
- Data processing steps
- Error messages with specific details

## 🔒 Security Notes

- API keys and credentials are stored as environment variables
- JIRA credentials are embedded but can be externalized
- Never commit API keys to version control
- Use secure methods to set environment variables in production

## 📈 Performance Tips

- Use specific JQL queries to limit data retrieval
- Set appropriate max results based on your needs
- Monitor API rate limits for external services
- Cache assignee data using session state (already implemented)

## 🤝 Contributing

To extend the application:
1. Add new data source integrations
2. Implement additional ML models
3. Create custom visualizations
4. Enhance AI prompts for better insights

## 📄 License

This project is for internal use. Ensure compliance with API terms of service for external services used.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review application logs
3. Verify API configurations
4. Contact the development team

---

## 🎉 v1.0.0 Release Notes

### What's New in v1.0.0
- **🎯 JIRA-Focused Platform**: Enterprise-grade JIRA analytics platform
- **📊 Advanced Team Analytics**: Individual assignee performance tracking
- **📅 Calendar-Based Accuracy**: Verified working days calculations
- **🔍 Task-Level Transparency**: Detailed breakdown of every metric
- **📋 Comprehensive Documentation**: Field explanations for all calculations
- **🤖 Enhanced AI Integration**: Improved Gemini AI insights
- **🚀 Streamlined UI**: Clean, professional interface design

### Coming in Future Releases
- **v2.0.0**: Microsoft Excel and Google Sheets integration
- **v2.1.0**: CSV file upload capabilities
- **v2.2.0**: Confluence integration
- **v3.0.0**: Advanced reporting and dashboard features

---

**JIRA Data Analyzer v1.0.0** | Built with ❤️ using Streamlit | Enterprise-Ready Analytics 📊✨
