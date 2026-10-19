from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import LLM_MODEL   # also loads .env

print("Testing model:", LLM_MODEL)

llm = ChatNVIDIA(
    model=LLM_MODEL,
    temperature=0,
)

response = llm.invoke("Hello")

print(response.content)
