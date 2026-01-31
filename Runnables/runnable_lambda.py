from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda,RunnableSequence, RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

def word_count(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count) # we can also give lambda fubction here
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic':'AI'})
print("""{} \n Word count -{}""".format(result['joke'],result['word_count']))