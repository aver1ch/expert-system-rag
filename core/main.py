from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import glob

# 1. LLM
llm = Ollama(model="deepseek-r1:7b")

# 2. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

# 3. Загрузка всех PDF
all_docs = []
for pdf_file in glob.glob("docs/*.pdf"):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    for doc in docs:
        doc.metadata['source'] = pdf_file
    all_docs.extend(docs)

# 4. Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
documents = splitter.split_documents(all_docs)

# 5. Persistent FAISS
db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_db")

# 6. RAG
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

# 7. Запрос
query = "Что говорит ГОСТ 123 о безопасности?"
res = qa(query)
print(res["result"])
print([d.metadata['source'] for d in res["source_documents"]])
