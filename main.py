import json

import chromadb
from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

loader = JSONLoader(
    file_path="./datasets/train.json",
    jq_schema='.data[] | {recipe_id: .recipe_id, steps: (.context | map(.body) | join(" \n "))}',
    text_content=False,
)
data = loader.load()

# RecursiveJsonSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",
        "\uff0c",
        "\u3001",
        "\uff0e",
        "\u3002",
        "",
    ],
)

texts = []
metadatas = []
for doc in data[:100]:
    doc.page_content = json.loads(doc.page_content)
    recipe_id = doc.page_content["recipe_id"]
    steps = recipe_id + " " + doc.page_content["steps"]

    # Combine all steps into one text (or split per-step)
    chunks = splitter.split_text(steps)

    # Store chunks + metadata
    texts.extend(chunks)
    metadatas.extend([{"recipe_id": recipe_id}] * len(chunks))

print(len(data))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts)

recipe = chromadb.Client()
collection = recipe.get_or_create_collection("test")
ids = [f"chunk_{i}" for i in range(len(texts))]
collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

# Testing the chromadb
results = collection.query(
    query_texts="halal vanilla extract",
    n_results=2,
)
# Print the results
print("Top matching recipe chunks:")
print(results)
for i, doc in enumerate(results["documents"][0]):
    recipe_name = results["metadatas"][0][i]["recipe_id"]
    print(f"\nMatch #{i + 1}: {recipe_name}")
    print(doc)


chroma_db = Chroma(
    client=recipe, collection_name="test", embedding_function="all-MiniLM-L6-v2"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="llama2"),
    chain_type="stuff",
    retriever=chroma_db.as_retriever(),
)

# query = "How to make vegan pasta?"
# print(qa_chain.run(query))
