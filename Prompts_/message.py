from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

messages = [
    SystemMessage(content = "You are a helpful assistant."),
    HumanMessage(content = "Tell me about langchain.")
]

result = model.invoke(messages)

messages.append(AIMessage(result.content))

print(messages)