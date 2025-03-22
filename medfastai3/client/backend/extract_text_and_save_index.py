# extract_text_and_save_index.py

import pandas as pd
from tqdm import tqdm
import pdfplumber
from langchain_community.document_loaders import DirectoryLoader  # Updated import (if needed)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import NLTKTextSplitter  # optional alternative

# For prompts and chains (if needed)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# For vector stores and embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# ---------------------------
# Step 1: Extract Text from PDF
# ---------------------------
# Correct the file path using either raw string or forward slashes.
pdf_path = r"C:\MedFast-main\medfastai3\client\backend\Brain_Tumors.pdf"
# Alternatively: pdf_path = "C:/MedFast-main/medfastai3/client/backend/Brain_Tumors.pdf"

pdf_reader = pdfplumber.open(pdf_path)

# Define the page intervals you want to include
included_pages_intervals = [
    [14, 52],
    [57, 82],
    [87, 155],
    [159, 188],
    [192, 225],
    [229, 264],
    [270, 306],
    [311, 348],
    [351, 365],
    [368, 393]
]

included_pages = []
for interval in included_pages_intervals:
    included_pages.extend(range(interval[0], interval[1] + 1))

def include_page(page_number):
    # pdfplumber pages are 0-indexed; add 1 for comparison
    return (page_number + 1) in included_pages

def include_text(obj):
    # Only include text objects with size >= 10 (to filter out small artifacts)
    return obj.get('size', 0) >= 10

def extract_single_page(page):
    filtered_page = page.filter(include_text)
    text = filtered_page.extract_text()
    tables = page.find_tables()
    table_text = ''
    for table in tables:
        table_df = pd.DataFrame.from_records(table.extract())
        # Check if table is empty: if all values are empty strings or null
        if (table_df == '').values.sum() + table_df.isnull().values.sum() == table_df.shape[0]*table_df.shape[1]:
            continue  # Table is empty
        else:
            table_text += '\n\n' + table_df.to_html(header=False, index=False)
    return text + '\n\n' + table_text

def extract_pages(pdf_reader, source):
    documents = []
    for page_number, page in tqdm(enumerate(pdf_reader.pages), total=len(pdf_reader.pages)):
        if include_page(page_number):
            doc = Document(
                page_content=extract_single_page(page),
                metadata={"source": source, "page": page_number + 1}
            )
            documents.append(doc)
    return documents

documents = extract_pages(pdf_reader, "Brain_Tumors.pdf")
print(f"Pages extracted: {len(documents)}")

# ---------------------------
# Step 2: Split the Text into Chunks
# ---------------------------
# Using RecursiveCharacterTextSplitter to split documents into chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Adjust chunk size as needed (in characters)
    chunk_overlap=50  # Overlap between chunks
)
split_docs = text_splitter.split_documents(documents)
print(f"Created {len(split_docs)} chunks from the textbook.")

# ---------------------------
# Step 3: Compute Embeddings and Build FAISS Index
# ---------------------------
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectordb = FAISS.from_documents(split_docs, embedding=embeddings)
print("FAISS index built.")

# ---------------------------
# Step 4: Save the FAISS Index Locally
# ---------------------------
vectordb.save_local("faiss_index_hp")
print("FAISS index saved to 'faiss_index_hp'.")
