from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

prompt1 = PromptTemplate(
    template = 'Generate a tweet about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a linkedin post about {topic}',
    input_variables = ['topic']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin_post': RunnableSequence(prompt2, model, parser)
})
parallel_chain.invoke({'topic':'Artificial Intelligence'})
result = parallel_chain.invoke({'topic':'Artificial Intelligence'})

print(result)
print(result['linkedin_post'])