import os
import streamlit as st
import pandas as pd
import gdown
from sqlalchemy import create_engine
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="SQL Agent Chat",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat bubbles
st.markdown("""
<style>
    /* Chat container */
    .chat-container {
        display: flex;
        margin: 10px 0;
        align-items: flex-start;
    }
    
    /* User message - right aligned */
    .user-container {
        justify-content: flex-end;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Assistant message - left aligned */
    .assistant-container {
        justify-content: flex-start;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        margin-right: auto;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Reasoning box */
    .reasoning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        margin-right: auto;
        border-radius: 8px;
        font-size: 0.9em;
    }
    
    .reasoning-title {
        font-weight: bold;
        color: #856404;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    
    .reasoning-content {
        color: #533f03;
        line-height: 1.6;
    }
    
    /* SQL Query display box */
    .sql-query-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 12px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
        margin: 10px 0;
        border-left: 4px solid #007acc;
        overflow-x: auto;
    }
    
    .sql-query-title {
        color: #4ec9b0;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    /* Query details styling */
    .query-box {
        background: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 10px;
        margin: 8px 0;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85em;
    }
    
    /* Avatar styling */
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        margin: 0 10px;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 SQL Agent - AWS Cost Analysis")
st.markdown("Ask questions about your AWS cost data and get instant answers!")

# -------------------------------
# Initialize Session State
# -------------------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.db = None
    st.session_state.graph = None

# -------------------------------
# Define State Structure
# -------------------------------
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    reasoning: str
    attempts: list

# -------------------------------
# Initialize Database and Graph
# -------------------------------
@st.cache_resource
def initialize_system():
    """Initialize database and LangGraph (cached for performance)"""
    
    output = "cur_data.csv"
    
    # Check if CSV exists in repo (for Streamlit Cloud)
    if not os.path.exists(output):
        # Try downloading from Google Drive (fallback)
        url = "https://drive.google.com/uc?id=1Ebxs4J-P-pRWLEJjWLq-FgJqjnusaSWa&export=download"
        with st.spinner("Downloading data from Google Drive..."):
            try:
                gdown.download(url, output, quiet=False)
            except Exception as e:
                st.error(f"Failed to download CSV: {str(e)}")
                st.info("Please ensure the CSV file is in your repository or the Google Drive link is public.")
                st.stop()
    
    # Load data
    df = pd.read_csv(output)
    
    # Setup database
    engine = create_engine("sqlite:///cur_data.db")
    df.to_sql("cur_data", engine, if_exists="replace", index=False)
    db = SQLDatabase(engine=engine)
    
    # Initialize LLM - Using Llama 3.3 70B via Groq (best free open-source for SQL)
    load_dotenv()
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    
    # Enhanced system prompt with reasoning
    system_message = """You are a SQL expert. Create queries for a SQLite database with this schema:

{table_info}

EXAMPLES of how to think:

Question: "Which resources can be terminated to save cost?"
Thinking: Find resources that are NOT running (IsRunning = 0 or False) and have positive cost
Query: SELECT Resourceid, Costperday, Project, ServiceType, IsRunning FROM cur_data WHERE (IsRunning = 0 OR IsRunning = 'False' OR IsRunning IS NULL) AND Costperday > 0 ORDER BY Costperday DESC LIMIT 15;

Question: "Show me expensive idle resources"
Thinking: Idle means not running, expensive means high cost per day
Query: SELECT Resourceid, Costperday, Project, ServiceType, team, IsRunning FROM cur_data WHERE (IsRunning = 0 OR IsRunning = 'False') AND Costperday > 5 ORDER BY Costperday DESC LIMIT 10;

Question: "Which team has the highest usage?"
Thinking: Sum costs by team
Query: SELECT team, SUM(Costperday) as total_cost, COUNT(*) as resource_count FROM cur_data WHERE team IS NOT NULL GROUP BY team ORDER BY total_cost DESC LIMIT 10;

Question: "Top 5 services by cost"
Thinking: Group by service type and sum costs
Query: SELECT ServiceType, SUM(Costperday) as total_cost FROM cur_data GROUP BY ServiceType ORDER BY total_cost DESC LIMIT 5;

IMPORTANT NOTES:
- IsRunning can be 0, 1, 'False', 'True', or NULL - handle all cases
- For "idle" or "not running" resources: use (IsRunning = 0 OR IsRunning = 'False' OR IsRunning IS NULL)
- For "running" resources: use (IsRunning = 1 OR IsRunning = 'True')
- Always filter Costperday > 0 when looking for cost-saving opportunities
- Use LIMIT {top_k} unless user specifies otherwise
- Column names are CASE SENSITIVE - use exact names from schema
- Order results DESC for costs (show highest first)

Create a valid SQLite query."""
    
    user_prompt = "Question: {input}"
    query_prompt_template = ChatPromptTemplate(
        [("system", system_message), ("user", user_prompt)]
    )
    
    # Define query functions
    class QueryOutput(TypedDict):
        query: str
    
    def write_query(state: State):
        """Generate SQL query from question"""
        prompt_input = {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state.get("question", ""),
        }
        
        prompt = query_prompt_template.invoke(prompt_input)
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        state["query"] = result["query"]
        if "attempts" not in state:
            state["attempts"] = []
        return state
    
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    
    def execute_query(state: State):
        """Execute the SQL query"""
        query = state.get("query", "")
        attempts = state.get("attempts", [])
        
        try:
            result = execute_query_tool.invoke(query)
            
            # More lenient result checking
            result_str = str(result).strip() if result else ""
            is_success = bool(result_str and result_str != "[]" and result_str != "()")
            
            attempts.append({
                "query": query,
                "result": result_str[:500],
                "success": is_success
            })
            state["attempts"] = attempts
            state["result"] = result_str if result_str else "No matching records found"
            return state
            
        except Exception as e:
            error_msg = str(e)
            attempts.append({
                "query": query,
                "result": f"ERROR: {error_msg}",
                "success": False
            })
            state["attempts"] = attempts
            state["result"] = f"Query failed: {error_msg}"
            return state
    
    def generate_answer(state: State):
        """Generate natural language answer with reasoning"""
        result = state.get("result", "")
        question = state.get("question", "")
        query = state.get("query", "")
        
        # Handle actual query errors
        if "Query failed:" in result and "ERROR:" in result:
            return {
                "answer": f"I encountered an error executing the query. Please try rephrasing your question.",
                "reasoning": f"The SQL query failed with an error. This might be due to incorrect column names or syntax issues."
            }
        
        # Handle empty results
        if not result or result == "No matching records found" or result == "[]" or result == "()":
            return {
                "answer": f"I couldn't find any data matching your query about: '{question}'.",
                "reasoning": "The query executed successfully but returned no results. This could mean:\n- No resources match the specified criteria\n- The column values might be in a different format\n- Try broadening your search criteria"
            }
        
        # Generate answer with reasoning
        prompt = f"""Based on this database query result, provide a comprehensive analysis with detailed insights.

Question: {question}
SQL Query Used: {query}
Query Result: {result}

Provide your response in this EXACT format:

ANSWER:
[Provide a detailed, comprehensive answer. Include:
- Specific resource IDs, costs, and relevant details from the data
- Clear breakdown of findings with numbers and percentages where applicable
- Actionable recommendations based on the data
- Total savings or costs calculated from the results
- Format the answer in a readable way with bullet points, numbered lists, or tables as needed
- Make it thorough and informative - at least 3-4 paragraphs or equivalent detail]

REASONING:
[Provide detailed reasoning explaining:
- Why you queried these specific columns and conditions
- What patterns or insights you discovered in the data
- How you interpreted the results to reach your conclusion
- What makes these findings significant for cost optimization or analysis
- Any additional context that helps understand the answer
- Make this 2-3 paragraphs explaining your analytical process]"""
        
        response = llm.invoke(prompt)
        content = response.content
        
        # Parse answer and reasoning
        answer = ""
        reasoning = ""
        
        if "ANSWER:" in content and "REASONING:" in content:
            parts = content.split("REASONING:")
            answer = parts[0].replace("ANSWER:", "").strip()
            reasoning = parts[1].strip()
        else:
            answer = content
            reasoning = "Analysis based on the query results above."
        
        return {
            "answer": answer,
            "reasoning": reasoning
        }
    
    # Build LangGraph
    graph_builder = StateGraph(State).add_sequence(
        [write_query, execute_query, generate_answer]
    )
    graph_builder.add_edge(START, "write_query")
    graph = graph_builder.compile()
    
    return db, graph, df

# -------------------------------
# Initialize on First Run
# -------------------------------
if not st.session_state.initialized:
    try:
        with st.spinner("Initializing system... This may take a moment."):
            db, graph, df = initialize_system()
            st.session_state.db = db
            st.session_state.graph = graph
            st.session_state.df = df
            st.session_state.initialized = True
        st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.stop()

# -------------------------------
# Sidebar - Database Info
# -------------------------------
with st.sidebar:
    st.header("📊 Database Info")
    if st.session_state.initialized:
        st.metric("Total Records", len(st.session_state.df))
        st.metric("Columns", len(st.session_state.df.columns))
        
        with st.expander("View Column Names"):
            st.code("\n".join(st.session_state.df.columns.tolist()))
        
        with st.expander("Sample Data"):
            st.dataframe(st.session_state.df.head(5))
    
    st.divider()
    st.header("💡 Example Questions")
    st.markdown("""
    **Cost Optimization:**
    - Which resources can be terminated to save cost?
    - Show me expensive idle resources
    - What's the total cost of non-running resources?
    
    **Usage Analysis:**
    - Which team has the highest usage?
    - Which project spends the most?
    - Top 5 cloud services by cost
    
    **Resource Management:**
    - Show running resources by cost
    - Which resources have no project?
    - Total cost by environment
    """)

# -------------------------------
# Main Chat Interface
# -------------------------------

# Display chat messages with custom styling
for message in st.session_state.messages:
    if message["role"] == "user":
        # User message - right aligned
        st.markdown(f"""
        <div class="chat-container user-container">
            <div class="user-message">
                {message["content"]}
            </div>
            <div class="avatar user-avatar">👤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant message - left aligned
        st.markdown(f"""
        <div class="chat-container assistant-container">
            <div class="avatar assistant-avatar">🤖</div>
            <div class="assistant-message">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show reasoning if available
        if "reasoning" in message and message["reasoning"]:
            st.markdown(f"""
            <div style="max-width: 70%; margin-left: 46px;">
                <div class="reasoning-box">
                    <div class="reasoning-title">
                        💡 My Reasoning
                    </div>
                    <div class="reasoning-content">
                        {message["reasoning"]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show SQL query used
        if "attempts" in message and message["attempts"]:
            for attempt in message["attempts"]:
                if attempt["success"]:
                    st.markdown(f"""
                    <div style="max-width: 70%; margin-left: 46px;">
                        <div class="sql-query-box">
                            <div class="sql-query-title">🔍 SQL Query Used:</div>
                            <code>{attempt["query"]}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    break  # Only show the successful query

# Chat input
if prompt := st.chat_input("Ask a question about your AWS costs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    st.markdown(f"""
    <div class="chat-container user-container">
        <div class="user-message">
            {prompt}
        </div>
        <div class="avatar user-avatar">👤</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate response
    status_placeholder = st.empty()
    result_container = st.container()
    
    try:
        status_placeholder.info("🤔 Analyzing your question...")
        
        # Run the graph
        final_state = {}
        for state in st.session_state.graph.stream(
            {"question": prompt, "attempts": []}, 
            stream_mode="values"
        ):
            final_state = state
            
            if "query" in state and not final_state.get("result"):
                status_placeholder.info("⚙️ Executing SQL query...")
            elif "result" in state and not final_state.get("answer"):
                status_placeholder.info("📝 Generating answer...")
        
        status_placeholder.empty()
        
        # Extract answer and reasoning
        if "answer" in final_state:
            answer = final_state["answer"]
            reasoning = final_state.get("reasoning", "")
            attempts = final_state.get("attempts", [])
            
            # Display assistant response
            with result_container:
                st.markdown(f"""
                <div class="chat-container assistant-container">
                    <div class="avatar assistant-avatar">🤖</div>
                    <div class="assistant-message">
                        {answer}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show reasoning
                if reasoning:
                    st.markdown(f"""
                    <div style="max-width: 70%; margin-left: 46px;">
                        <div class="reasoning-box">
                            <div class="reasoning-title">
                                💡 My Reasoning
                            </div>
                            <div class="reasoning-content">
                                {reasoning}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show SQL query used
                if attempts:
                    for attempt in attempts:
                        if attempt["success"]:
                            st.markdown(f"""
                            <div style="max-width: 70%; margin-left: 46px;">
                                <div class="sql-query-box">
                                    <div class="sql-query-title">🔍 SQL Query Used:</div>
                                    <code>{attempt["query"]}</code>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            break  # Only show the successful query
            
            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "reasoning": reasoning,
                "attempts": attempts
            })
        else:
            result_container.error("Unable to generate response. Please try again.")
            
    except Exception as e:
        status_placeholder.empty()
        result_container.error(f"Error: {str(e)}")
        st.exception(e)

# -------------------------------
# Clear Chat Button
# -------------------------------
if st.session_state.messages:
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()