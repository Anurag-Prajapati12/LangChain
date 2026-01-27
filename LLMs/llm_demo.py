from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
result=llm.invoke("What is the capital of India")

print(result.content)