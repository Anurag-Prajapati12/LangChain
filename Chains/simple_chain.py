from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

prompt = PromptTemplate(
    template = 'Generate five interesting dacts about {topic}',
    input_variables = ['topic'],

)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

# Chain visualization
chain.get_graph().print_ascii()