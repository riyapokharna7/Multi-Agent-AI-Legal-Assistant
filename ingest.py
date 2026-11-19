import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

print("1. Embedding model load ho raha hai...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

data_folder = "./data"
txt_files = []

for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".txt"):
            txt_files.append(os.path.join(root, file))

if not txt_files:
    print("ERROR: No .txt files found in the data folder. Please check the folder path and ensure that it contains .txt files.")
else:
    print(f"Found {len(txt_files)} .txt files in the data folder. Processing files...")
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for index, file_path in enumerate(txt_files, start=1):
        filename = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))

        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["category"] = folder_name

        all_chunks.extend(chunks)

        if index % 50 == 0 or index == len(txt_files):
            print(f"   -> {index}/{len(txt_files)} files processed...")

    print(f"Storing a total of {len(all_chunks)} chunks in Qdrant Vector Database...")
    vector_store = QdrantVectorStore.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        path="./qdrant_db",
        collection_name="legal_contracts",
        force_recreate=True
    )
    print("SUCCESS! Chunks has been stored in Qdrant Vector Database.")