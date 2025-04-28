import json

import chromadb
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Step 1: Load JSON data and extract recipe steps
loader = JSONLoader(
    file_path="./datasets/train.json",
    jq_schema='.data[] | {recipe_id: .recipe_id, steps: (.context | map(.body) | join(" \n "))}',
    text_content=False,
)
data = loader.load()
print("Data loading done!")

# Step 2: Split text into chunks for better retrieval
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

texts, metadatas = [], []

# Step 3: Process data (first 100 recipes)
for doc in data[:100]:
    doc.page_content = json.loads(doc.page_content)
    recipe_id = doc.page_content["recipe_id"]
    steps = recipe_id + " " + doc.page_content["steps"]
    chunks = splitter.split_text(steps)
    texts.extend(chunks)
    metadatas.extend([{"recipe_id": recipe_id}] * len(chunks))

print("Data splitting done!")

# Step 4: Embed text chunks for semantic search
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts)
print("Text embedding done!")

# Step 5: Initialize ChromaDB client and add documents
recipe = chromadb.Client()
collection = recipe.get_or_create_collection("test")
ids = [f"chunk_{i}" for i in range(len(texts))]
collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
print("ChromaDB collection updated!")


# Step 6: Test function for ChromaDB query
def testing_function(collection):
    results = collection.query(
        query_texts="halal vanilla extract",
        n_results=2,
    )
    print("Top matching recipe chunks:")
    for i, doc in enumerate(results["documents"][0]):
        recipe_name = results["metadatas"][0][i]["recipe_id"]
        print(f"\nMatch #{i + 1}: {recipe_name}")
        print(doc)


# testing_function(collection)

# Step 7: Set up LangChain HuggingFace embeddings
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("HuggingFace embeddings set up!")  # Step 7 done

# Step 8: Initialize Chroma for LangChain
chroma_db = Chroma(
    client=recipe, collection_name="test", embedding_function=hf_embeddings
)

# Step 9: Create QA chain for querying
qa_chain = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="tinyllama"),
    chain_type="stuff",
    retriever=chroma_db.as_retriever(),
)
print("QA chain created!")

# Query example
# query = "How to make vegan pasta?"
# print(qa_chain.run(query))
