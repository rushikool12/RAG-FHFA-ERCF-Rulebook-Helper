import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from the .env file
load_dotenv()

print ("Loading PDF file...")
loader = PyPDFLoader("./FHFA_ERCF_Framework.pdf")
document = loader.load()
# print(document[0].page_content)

print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(document)
print(f"Total pages: {len(document)}")
print(f"Total chunks: {len(chunks)}")
# print(chunks[0].page_content)

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
if os.path.exists("./chroma_db"):
    print("Loading existing vector database...")
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
else:
    print("Creating vector database...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

# Configure the database to act as a document retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Define the hidden prompt structure for the LLM
template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Initialize the free Gemini model tier
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# Helper function to stitch retrieved chunks into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect everything together using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# user_question = "What days can I work from home?"
# print(f"\nQuestion: {user_question}")

# response = rag_chain.invoke(user_question)
# print(f"Answer: {response.content}")

# Clean up the output if Gemini returns a list of content blocks
# if isinstance(response.content, list):
#     clean_answer = response.content[0]['text']
# else:
#     clean_answer = response.content

# print(f"Answer: {clean_answer}")

# Chat with your PDF in a continuous loop
print("\n--- PDF Chatbot Initialized ---")
print("Type 'exit' or 'quit' to stop.")

while True:
    # 1. Wait for the user to type a question
    user_question = input("\nYour Question: ")

    # 2. Allow the user to break the loop and close the program
    if user_question.lower() in ['exit', 'quit']:
        print("Shutting down chatbot. Goodbye!")
        break

    # 3. Send the question through our RAG chain
    response = rag_chain.invoke(user_question)

    # 4. Clean up the output format
    if isinstance(response.content, list):
        clean_answer = response.content[0]['text']
    else:
        clean_answer = response.content

    # 5. Print the final answer to the console
    print(f"Answer: {clean_answer}")