from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA

for m in ChatNVIDIA.get_available_models():
    print(m.id)
