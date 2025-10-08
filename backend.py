# Step 1 : Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# Step 2 : Setup AI agent from Frontend Request
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

ALLOWED_MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]

app = FastAPI(title="LangGraph AI Agent")

@app.post("/chat")
def chat_endpoint(request: RequestState):
    try:
        if request.model_name not in ALLOWED_MODEL_NAMES:
            return {"error": "Invalid Model Name"}

        state_messages = request.messages  # List of strings

        # Call AI agent
        response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=state_messages,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            provider=request.model_provider
        )
        return {"answer": response}

    except Exception as e:
        return {"error": str(e)}


# Step 3 : Run app & explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
