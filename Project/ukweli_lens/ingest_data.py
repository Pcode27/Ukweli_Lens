# ingest_data.py
import os
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---
DOCS_PATH = "./source_documents"
CHROMA_PATH = "./chroma_db"
MODEL_NAME = "multi-qa-mpnet-base-dot-v1"

# --- 2. LOAD DOCUMENTS ---
print("Loading documents...")
documents = []
for filename in os.listdir(DOCS_PATH):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(DOCS_PATH, filename)
        loader = PyPDFLoader(pdf_path)
        # Load and split pages
        pages = loader.load_and_split()
        documents.extend(pages)
        print(f"Loaded {len(pages)} pages from {filename}")

if not documents:
    print("\n❌ FAILURE: NO DOCUMENTS LOADED.")
    print(f"Directory checked: {os.path.abspath(DOCS_PATH)}")
    
    # Check if the directory exists
    if not os.path.exists(DOCS_PATH):
        print("Reason 1: The 'source_documents' folder does not exist.")
    else:
        # Check if files exist but were skipped
        files_in_dir = os.listdir(DOCS_PATH)
        pdf_count = len([f for f in files_in_dir if f.endswith('.pdf')])
        
        if pdf_count == 0:
            print("Reason 2: The 'source_documents' folder exists, but contains NO PDF files.")
        else:
            print("Reason 3: PDFs were found but failed to load (e.g., encrypted or corrupt).")
    
    exit()

# --- 3. CHUNK DOCUMENTS ---
print("Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} pages into {len(chunks)} chunks.")

# Prepare data for ChromaDB (texts and metadata)
texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]

# --- 4. INITIALIZE MODEL & DB ---
print(f"Loading embedding model: {MODEL_NAME}")
# This will download the model (approx. 420MB) the first time
model = SentenceTransformer(MODEL_NAME)

print("Initializing ChromaDB (persistent)...")
# Use PersistentClient to save the DB to disk
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection_name = "uhakiki_docs"

# Get or create the collection
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"} # Use cosine similarity
)

# --- 5. GENERATE EMBEDDINGS & INDEX ---
print("Generating embeddings (this may take a while)...")
# Note: model.encode() is smart and can batch process
embeddings = model.encode(texts, show_progress_bar=True)
print(f"Generated {len(embeddings)} embeddings.")

print("Adding documents to ChromaDB collection...")
# Add in batches (good for large datasets)
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    batch_metadatas = metadatas[i:i+batch_size]
    # Create unique IDs for each chunk
    batch_ids = [f"chunk_{j}" for j in range(i, i + len(batch_texts))]
    
    collection.add(
        embeddings=batch_embeddings,
        documents=batch_texts,
        metadatas=batch_metadatas,
        ids=batch_ids
    )

print(f"\n--- Phase 1 Ingestion Complete ---")
print(f"Total chunks indexed: {collection.count()}")
print(f"ChromaDB data saved to: {CHROMA_PATH}")