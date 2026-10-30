
import streamlit as st
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# 1. Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Create a simple tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# 3. Create AI Agent
agent = create_react_agent(
    llm,
    [add_numbers]
)

# 4. Streamlit UI
st.title("My First AI Agent")

question = st.chat_input("Ask something...")

if question:

    st.write("You:", question)

    # 5. Execute Agent
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    # 6. Display response
    answer = result["messages"][-1].content

    st.write("AI Agent:", answer)
