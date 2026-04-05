import sys
print(f"Python: {sys.version}")

try:
    import langchain
    print(f"LangChain: {langchain.__version__}")
except:
    print("LangChain not found/version unknown")

print("\n--- Attempting Imports ---")

# Check ParentDocumentRetriever
try:
    from langchain.retrievers import ParentDocumentRetriever
    print("SUCCESS: from langchain.retrievers import ParentDocumentRetriever")
except ImportError as e:
    print(f"FAIL: from langchain.retrievers import ParentDocumentRetriever ({e})")

try:
    from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
    print("SUCCESS: from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever")
except ImportError as e:
    print(f"FAIL: from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever ({e})")

try:
    from langchain.retrievers.multi_vector import MultiVectorRetriever
    print("SUCCESS: from langchain.retrievers.multi_vector import MultiVectorRetriever")
except ImportError as e:
    print(f"FAIL: from langchain.retrievers.multi_vector import MultiVectorRetriever ({e})")
