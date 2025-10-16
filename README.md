# 💬 SQL Agent Chatbot

An intelligent chatbot built with **LangChain** and **LangGraph** that allows you to query your database using natural language. Ask questions about your data in plain English and get instant SQL-backed answers with detailed reasoning.

## 🌟 Features

- **Natural Language to SQL**: Ask questions in plain English, get SQL queries automatically
- **Powered by LangChain**: Built on LangChain framework with LangGraph orchestration
- **Smart Reasoning**: See the AI's thought process behind each answer
- **Modern Chat Interface**: Beautiful UI with gradient bubbles and real-time responses
- **Query Transparency**: View the exact SQL queries being executed
- **Fully Customizable**: Easy to adapt to any database schema and use case
- **Free LLM**: Uses Llama 3.3 70B via Groq's free API

## 🚀 Quick Start

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Prepare Your Data

Add your CSV file to the project directory. The app expects a CSV file that will be loaded into SQLite.

**Default file name**: `cur_data.csv`

You can change this in the code by modifying the `output` variable in the `initialize_system()` function.

### 3. Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Add your **Groq API key** in Streamlit Secrets:
   - Go to App Settings → Secrets
   - Add the following:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
6. Deploy!

**Getting a Free Groq API Key:**
- Visit [console.groq.com](https://console.groq.com)
- Sign up for a free account
- Create an API key in the dashboard
- Copy and paste it into Streamlit Secrets

### 4. Start Asking Questions!

Once deployed, you can immediately start querying your data using natural language.

## 🎯 How to Customize for Your Use Case

### Changing the Data Source

**Option 1: Use Your Own CSV**

Replace `cur_data.csv` with your own data file. Update the filename in the code:

```python
output = "your_data.csv"  # Change this line
```

**Option 2: Use Google Drive**

Update the Google Drive URL to point to your file:

```python
url = "https://drive.google.com/uc?id=YOUR_FILE_ID&export=download"
```

### Customizing for Your Database Schema

The chatbot needs to understand your data structure. Modify the system prompt in the `initialize_system()` function:

**1. Update the prompt examples:**

```python
system_message = """You are a SQL expert. Create queries for a SQLite database with this schema:

{table_info}

EXAMPLES of how to think:

Question: "Your example question?"
Thinking: How to approach this query
Query: SELECT your_columns FROM your_table WHERE conditions;

Question: "Another example?"
Thinking: Analysis approach
Query: SELECT other_columns FROM your_table GROUP BY something;

IMPORTANT NOTES:
- Add notes specific to your data (column types, special values, etc.)
- Mention any data quirks or special handling needed
- Specify which columns are most important
- Add formatting preferences for results
"""
```

**2. Update example questions in the sidebar:**

```python
st.markdown("""
**Your Category 1:**
- Example question 1 for your data
- Example question 2 for your data

**Your Category 2:**
- Example question 3 for your data
- Example question 4 for your data
""")
```

### Customizing the UI

**Change Colors:**

Modify the CSS gradient in `st.markdown()`:

```css
.user-message {
    background: linear-gradient(135deg, #your-color-1 0%, #your-color-2 100%);
}
```

**Change Page Title:**

```python
st.set_page_config(
    page_title="Your App Name",
    page_icon="🔍",  # Your emoji
    layout="wide"
)

st.title("🔍 Your App Title")
st.markdown("Your custom description!")
```

## 🏗️ Architecture

### Built With

- **LangChain**: Framework for building LLM applications
- **LangGraph**: Orchestration layer for complex LLM workflows
- **Streamlit**: Web interface framework
- **Groq**: Fast, free LLM inference (Llama 3.3 70B)
- **SQLite**: Lightweight database for querying
- **Pandas**: Data processing and CSV handling

### How It Works

```
User Question
    ↓
LangGraph Workflow
    ↓
1. Write Query Node → Generates SQL from natural language
    ↓
2. Execute Query Node → Runs SQL against database
    ↓
3. Generate Answer Node → Creates human-readable response
    ↓
Answer + Reasoning + SQL Query
```

The LangChain framework handles:
- Prompt engineering and templating
- LLM interactions and structured outputs
- Database connections and query execution
- State management across workflow steps

## 📋 Requirements

Create a `requirements.txt` file:

```txt
streamlit
pandas
sqlalchemy
langchain
langchain-community
langchain-groq
langgraph
gdown
python-dotenv
typing-extensions
```

## 🔧 Advanced Customization

### Using a Different LLM

Replace the Groq model with any LangChain-supported provider:

```python
# OpenAI
llm = init_chat_model("gpt-4", model_provider="openai")

# Anthropic
llm = init_chat_model("claude-3-sonnet", model_provider="anthropic")

# Local model
llm = init_chat_model("local-model-name", model_provider="ollama")
```

Don't forget to update the Streamlit Secrets with the appropriate API key!

### Adding More Graph Nodes

Extend the LangGraph workflow with custom processing:

```python
def custom_node(state: State):
    # Your custom logic
    state["custom_field"] = process_data(state)
    return state

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, custom_node, generate_answer]
)
```

### Using PostgreSQL or MySQL

Replace SQLite with a production database:

```python
# PostgreSQL
engine = create_engine("postgresql://user:password@host:port/database")

# MySQL
engine = create_engine("mysql+pymysql://user:password@host:port/database")
```

Update your Streamlit Secrets with database credentials.

### Adding File Upload

Allow users to upload their own CSV files:

```python
uploaded_file = st.file_uploader("Upload your data", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Process and load into database
```

## 📁 Project Structure

```
your-project/
├── streamlit_cloud.py    # Main application
├── your_data.csv          # Your data file
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

**Important**: Add `*.csv` and `*.db` to your `.gitignore` to avoid uploading large data files.

## 🎨 UI Customization Examples

### Minimal Chat Interface

Remove reasoning boxes and SQL display for a cleaner look by deleting those sections from the chat display loop.

### Add Data Visualizations

Integrate charts using Streamlit:

```python
import plotly.express as px

# After query execution
if result:
    df_result = pd.DataFrame(result)
    fig = px.bar(df_result, x='column', y='value')
    st.plotly_chart(fig)
```

### Dark Mode

Streamlit supports dark mode by default. Users can toggle it in settings.

## 🐛 Common Issues

### "No such table" Error

**Solution**: Verify your CSV file exists and the table name matches in queries.

### Slow Response Times

**Solution**: 
- Use Groq for faster inference (already configured)
- Add query limits: `LIMIT 100`
- Index frequently queried columns

### Empty Results

**Solution**: Check that column names in the prompt examples match your actual data (case-sensitive!).

## 💡 Use Cases

This chatbot can be adapted for:

- **Financial Analysis**: Query transaction data, expenses, revenue
- **Sales Analytics**: Analyze customer data, sales metrics, trends
- **Inventory Management**: Check stock levels, suppliers, orders
- **HR Analytics**: Query employee data, attendance, performance
- **IoT Data**: Analyze sensor readings, device metrics
- **Log Analysis**: Query application logs, error patterns
- **Research Data**: Scientific datasets, experiment results
- **Any CSV Data**: Adapt to any structured dataset!

## 🔐 Security Notes

- Never commit API keys to GitHub
- Use Streamlit Secrets for all sensitive credentials
- The app uses read-only database operations for safety
- LangChain handles SQL injection prevention

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Groq API Documentation](https://console.groq.com/docs)

## 🤝 Contributing

Feel free to:
- Fork this project
- Customize it for your needs
- Share your improvements
- Open issues for bugs

## 📝 Note

- cost_analysis_chatbot.ipynb file is just for reference, use streamlit_cloud.py 

---

**Built with LangChain + LangGraph + Streamlit**

Made with ❤️ for the community
