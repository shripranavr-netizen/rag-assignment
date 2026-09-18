import os
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "./chroma_db"

def ingest_and_index(file_path: str):
    """Loads document, chunks text, creates embeddings, and stores in ChromaDB."""
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        sys.exit(1)

    print(f"\n[Step 1: Document Processing] Loading: {file_path}")
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()
    print(f"-> Successfully loaded {len(docs)} document page(s)/section(s).")

    print("\n[Step 2: Chunking]")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"-> Split into {len(chunks)} text chunks (Size: 1000 chars, Overlap: 200 chars).")

    print("\n[Step 3 & 4: Embeddings & Vector Database Indexing]")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"-> Chunks embedded and indexed into ChromaDB at '{CHROMA_PATH}'.")
    return vector_db, embeddings

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_qa_loop(vector_db):
    """Sets up LLM, retrieval, and runs an interactive Q&A loop."""
    print("\n[Step 5 & 6: LLM & Similarity Retrieval Setup]")
    llm = ChatOllama(model="llama3.1", temperature=0)

    system_prompt = (
        "You are an assistant for document question-answering tasks.\n"
        "Answer the question using ONLY the provided context.\n"
        "If the answer cannot be found in the context, clearly state: "
        "'The requested information is not available in the document.'\n"
        "Do not extrapolate or speculate.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Modern LCEL Chain replacing the deprecated modules
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "="*50)
    print(" RAG SYSTEM READY: Type your question or 'exit' to quit.")
    print("="*50)

    while True:
        try:
            query = input("\nAsk a question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting RAG system.")
                break

            print("\n--- Performing Similarity Search ---")
            docs_and_scores = vector_db.similarity_search_with_score(query, k=3)
            
            for rank, (doc, score) in enumerate(docs_and_scores, start=1):
                page = doc.metadata.get("page", "N/A")
                print(f"[Chunk {rank}] (Distance Score: {score:.4f} | Page: {page})")
                print(f"{doc.page_content[:180].strip()}...\n")

            print("--- Generating Grounded Response ---")
            answer = rag_chain.invoke(query)
            print("\nFinal Answer:")
            print(answer)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    doc_path = input("Enter path to your document (e.g., sample.pdf): ").strip()
    db, _ = ingest_and_index(doc_path)
    run_qa_loop(db)
