import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import StructuredTool
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load dataset
df = pd.read_csv("BFS_Share_Price.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%y")

# ------------------- Monthly Stats Tool -------------------
class MonthYearInput(BaseModel):
    month: str = Field(..., description="Month like June or Jun or 06")
    year: int = Field(..., description="Year like 2022")

def get_monthly_stats(month: str, year: int) -> str:
    try:
        month = month[:3].capitalize()
        month_num = datetime.strptime(month, "%b").month
        filtered = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month_num)]
    except:
        return "Invalid month format. Try June or Jun or 06."

    if filtered.empty:
        return f"No data found for {month}-{year}"

    avg = round(filtered['Close Price'].mean(), 2)
    min_ = round(filtered['Close Price'].min(), 2)
    max_ = round(filtered['Close Price'].max(), 2)

    return f"{month}-{year} Stats:\n• Avg: ₹{avg}\n• Min: ₹{min_}\n• Max: ₹{max_}"

monthly_stats_tool = StructuredTool.from_function(
    func=get_monthly_stats,
    name="GetMonthlyStats",
    description="Get max, min, average price for a given month and year",
    args_schema=MonthYearInput
)

# ------------------- Yearly Stats Tool -------------------
class YearInput(BaseModel):
    year: int = Field(..., description="Year like 2023")

def get_yearly_stats(year: int) -> str:
    filtered = df[df['Date'].dt.year == year]
    if filtered.empty:
        return f"No data for year {year}"

    avg = round(filtered['Close Price'].mean(), 2)
    min_ = round(filtered['Close Price'].min(), 2)
    max_ = round(filtered['Close Price'].max(), 2)

    return f"{year} Stats:\n• Avg: ₹{avg}\n• Min: ₹{min_}\n• Max: ₹{max_}"

yearly_stats_tool = StructuredTool.from_function(
    func=get_yearly_stats,
    name="GetYearlyStats",
    description="Get max, min, average price for a given year",
    args_schema=YearInput
)

# ------------------- Custom Date Range Tool -------------------
class DateRangeInput(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

def get_custom_range_stats(start_date: str, end_date: str) -> str:
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except:
        return "Invalid date format. Use YYYY-MM-DD."

    filtered = df[(df['Date'] >= start) & (df['Date'] <= end)]
    if filtered.empty:
        return "No data for the given range"

    avg = round(filtered['Close Price'].mean(), 2)
    min_ = round(filtered['Close Price'].min(), 2)
    max_ = round(filtered['Close Price'].max(), 2)

    return f"{start.date()} to {end.date()} Stats:\n• Avg: ₹{avg}\n• Min: ₹{min_}\n• Max: ₹{max_}"

custom_range_tool = StructuredTool.from_function(
    func=get_custom_range_stats,
    name="GetCustomRangeStats",
    description="Get max, min, average price for a custom date range",
    args_schema=DateRangeInput
)

# ------------------- Trend Insight Tool -------------------
class TrendInput(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

def get_trend_insight(start_date: str, end_date: str) -> str:
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except:
        return "Invalid date format."

    filtered = df[(df['Date'] >= start) & (df['Date'] <= end)]
    if filtered.empty:
        return "No data for that range."

    change = filtered['Close Price'].iloc[-1] - filtered['Close Price'].iloc[0]
    pct_change = round((change / filtered['Close Price'].iloc[0]) * 100, 2)
    trend = "Increasing 📈" if change > 0 else "Decreasing 📉"
    return f"From {start.date()} to {end.date()}, the stock is {trend} by ₹{round(change,2)} ({pct_change}%)."

trend_tool = StructuredTool.from_function(
    func=get_trend_insight,
    name="GetTrendInsight",
    description="Get upward or downward trend and percentage change for any date range",
    args_schema=TrendInput
)

# ------------------- Transcript Semantic Search Tool -------------------
# Loading and combining all PDF transcripts
pdf_files = [
    "Earnings Call Transcript Q1 - FY25.pdf",
    "Earnings Call Transcript Q2 - FY25.pdf",
    "Earnings Call Transcript Q3 - FY25.pdf",
    "Earnings Call Transcript Q4 - FY25.pdf"
]

all_pages = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    pages = loader.load()
    all_pages.extend(pages)

# Split and embed
docs = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(all_pages)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedder)

def search_transcripts(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n---\n\n".join([doc.page_content for doc in results])

transcript_search_tool = Tool(
    name="SearchTranscripts",
    func=search_transcripts,
    description="Search quarterly transcripts for information like partnerships, market strategy, or commentary."
)

# ------------------- LLM and Agent -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

tools = [monthly_stats_tool, yearly_stats_tool, custom_range_tool, trend_tool, transcript_search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "system_message": """
You are a financial analyst assistant. You can answer questions about:
- Bajaj Finserv share prices using tools
- Quarterly investor call transcripts using the SearchTranscripts tool

When using tools:
- Avoid dumping raw numbers unless asked
- Summarize the answer smartly and clearly
- Use CFO tone for commentary-type queries
- Use the given tools to answer the queries if that particular tool is available
"""
    }
)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="FinSight AI", layout="centered")
st.title("FinSight AI: Bajaj Finserv Chatbot")
st.markdown("Ask anything about stock prices or investor discussions.")

with st.expander("💡 Try example queries"):
    st.markdown("""
    - What was the average price in May 2023?
    - Compare Bajaj Finserv from 2023-01-01 to 2023-03-31
    - Why is BAGIC facing headwinds in Motor insurance?
    - What's the rationale behind Hero partnership?
    - Draft CFO commentary for upcoming call
    """)

query = st.text_input("Your question:", placeholder="e.g. What is the trend from 2023-01-01 to 2023-03-31?")
if query:
    with st.spinner("Analyzing..."):
        try:
            response = agent.run(query)
            st.success(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.subheader("📈 Stock Price Chart")
date_range = st.date_input("Select a date range", [])
if len(date_range) == 2:
    chart_data = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]
    st.line_chart(chart_data.set_index("Date")["Close Price"])
