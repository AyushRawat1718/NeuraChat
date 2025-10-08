# Step 1: Setup UI with Streamlit
import streamlit as slt

slt.set_page_config(page_title="LangGraph Agent UI", layout="centered")
slt.title("AI Chatbot Agents")
slt.write("Create and Interact with AI Agents")

system_prompt = slt.text_area(
    "Define your AI Agent: ", height=70, placeholder="Type your system prompt here..."
)

MODEL_NAMES_GROQ = ["mixtral-8x7b-32768", "llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = slt.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = slt.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = slt.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search = slt.checkbox("Allow Web Search")

user_query = slt.text_area("Enter your query: ", height=150, placeholder="Ask Anything !")

API_URL = "http://127.0.0.1:9999/chat"

# Step 2: Send request to backend
if slt.button("Ask Agent !") and user_query.strip():
    import requests  # Section-specific import

    payload = {
        "model_name": selected_model,
        "model_provider": provider,
        "system_prompt": system_prompt,
        "messages": [user_query],
        "allow_search": allow_web_search
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        if "error" in response_data:
            slt.error(response_data["error"])
        else:
            slt.subheader("Agent Response")
            # Show the actual response from backend
            slt.markdown(f"**Final Response:** {response_data}")
    else:
        slt.error(f"Backend Error: {response.status_code}")
