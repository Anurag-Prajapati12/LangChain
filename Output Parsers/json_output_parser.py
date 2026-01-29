from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me a name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()} # it is partial_variable because it is not fill by user, it is fill in runtime
)
prompt = template.format()

result = model.invoke(prompt)
print(result.content)
final_result = parser.parse(result.content)

print(final_result)