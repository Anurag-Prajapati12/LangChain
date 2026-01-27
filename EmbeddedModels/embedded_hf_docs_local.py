from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents=[
    "Delhi is the capital of india",
    "Koklata is the capital of west bengal",
    "Paris is the capital of france"
]

vector=embedding.embed_documents(documents)

print(str(vector))