from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm=HuggingFacePipeline.from_model_id(
     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    ),
    model_kwargs={
        "low_cpu_mem_usage": True,
        "torch_dtype": "float16",
        "device_map": None
    }
)

model = ChatHuggingFace(llm=llm)


#1 prompt -> detailed report
template1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variables = ['topic']    
)



#2 prompt -> summary
template2 = PromptTemplate(
    template = "write a 5 line summary summary on following text. /n {text}",
    input_variables = ['text']    
)

prompt1=template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content })

result1 = model.invoke(prompt2)

print(result.content)

