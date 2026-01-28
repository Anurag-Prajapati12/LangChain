from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

prompt = PromptTemplate(
        template = 'answer the following question \n {question} from the following text - \n {text}',
        input_variables = ['question','text']

)

parser = StrOutputParser()

url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"
loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser
print(chain.invoke({'question':'what is the peak brightnes of this product','text':docs[0].page_content}))