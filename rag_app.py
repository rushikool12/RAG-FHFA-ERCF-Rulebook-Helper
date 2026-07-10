import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from the .env file
load_dotenv()

print ("Loading PDF file...")
loader = PyPDFLoader("./TechCorp_Official_Employee_Handbook.pdf")
document = loader.load()
print(document[0].page_content)
