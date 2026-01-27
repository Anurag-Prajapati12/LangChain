from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

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

result = model.invoke("summarize the attention is all you need paper in 5 lines")
print(result.content)

