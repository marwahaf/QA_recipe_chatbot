import json

import chromadb
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Set up JSON loader with specific extraction pattern
# jq_schema defines how to extract data from each record in the JSON file
loader = JSONLoader(
    file_path="./datasets/train.json",
    jq_schema='.data[] | {recipe_id: .recipe_id, steps: (.context | map(.body) | join(" \n "))}',
    text_content=False,
)  # Load the JSON data
data = loader.load()
print("Data loading done! ")


# Create a text splitter to break down recipe text into manageable chunks
# This helps with more precise retrieval and better context management
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

# Initialize lists to store processed text chunks and their metadata
texts = []
metadatas = []

# Process only the first 100 recipes for demonstration purposes
for doc in data[:100]:
    doc.page_content = json.loads(doc.page_content)
    recipe_id = doc.page_content["recipe_id"]
    # Combine recipe name with steps to ensure context is preserved
    steps = recipe_id + " " + doc.page_content["steps"]

    # Split the recipe text into chunks
    chunks = splitter.split_text(steps)

    # Store chunks + metadata
    texts.extend(chunks)
    metadatas.extend([{"recipe_id": recipe_id}] * len(chunks))

print("Data splitting done ")


# Initialize the embedding model from Sentence Transformers
# This model converts text into vector embeddings for semantic search
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# Initialize ChromaDB client
recipe = chromadb.Client()
collection = recipe.get_or_create_collection("test")

# Generate unique IDs for each chunk
ids = [f"chunk_{i}" for i in range(len(texts))]

# Add documents, embeddings, and metadata to the collection
collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def testing_function(collection):
    """Test the ChromaDB with a sample query"""
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


# testing_function(collection)

# Set up HuggingFace embeddings for LangChain
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Chroma for LangChain using the existing ChromaDB client
chroma_db = Chroma(
    client=recipe, collection_name="test", embedding_function=hf_embeddings
)

# Create a question-answering chain
# This combines the retriever with an LLM to generate answers based on retrieved context
qa_chain = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="tinyllama"),
    chain_type="stuff",
    retriever=chroma_db.as_retriever(),
)

# query = "How to make vegan pasta?"
# print(qa_chain.run(query))
