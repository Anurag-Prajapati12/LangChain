from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


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

class person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18,description='Age of the person')
    city: str = Field(description='Name of the city the person belongs too')

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template = 'Generate the name, age and city of a fictional {place} person\n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'place':'indian'})

result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
