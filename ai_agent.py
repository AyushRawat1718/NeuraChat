# ----------------------------
# Step 1: Setup API Keys
# ----------------------------
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ----------------------------
# Step 2: Setup LLM & Tools
# ----------------------------
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Initialize language models
openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")

# Initialize search tool
search_tool = TavilySearch(max_results=2)

# ----------------------------
# Step 3: Setup AI Agent with Search tool functionality
# ----------------------------
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# Define system prompt / persona
system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    # ----------------------------
    # Select LLM based on provider
    # ----------------------------
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    else:
        return "Invalid provider"

    # ----------------------------
    # Setup tools
    # ----------------------------
    tools = [TavilySearch(max_results=2)] if allow_search else []

    # ----------------------------
    # Create agent
    # ----------------------------
    from langgraph.prebuilt.chat_agent_executor import ReactAgentConfig
    agent_config = ReactAgentConfig(persona=system_prompt)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        agent_config=agent_config
    )

    # ----------------------------
    # Format user messages correctly
    # ----------------------------
    state = {"messages": [{"role": "user", "content": "\n".join(query)}]}

    # ----------------------------
    # Invoke agent
    # ----------------------------
    response = agent.invoke(state)
    messages = response.get("messages", [])

    # ----------------------------
    # Extract AI messages
    # ----------------------------
    ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
    raw_answer = ai_messages[-1] if ai_messages else "No response"

    # ----------------------------
    # Format the response nicely
    # ----------------------------
    import re

    # Remove empty lines
    formatted_answer = "\n".join([line.strip() for line in raw_answer.splitlines() if line.strip()])

    # Convert numbered list to bullets
    formatted_answer = re.sub(r"\d+\.\s", "- ", formatted_answer)

    return {"answer": formatted_answer}
