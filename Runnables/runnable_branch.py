from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda,RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableBranch
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

prompt1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarize the following text \n {text}',
    input_variable = ['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough() 
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukarine'}))
