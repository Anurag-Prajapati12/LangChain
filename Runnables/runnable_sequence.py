from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'Explain th following joke - {text}',
    input_variables = ['text']
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))
