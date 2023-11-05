import os
import pandas as pd    
import PyPDF2
from paperscraper.pdf import save_pdf
from paperscraper.get_dumps import biorxiv

import openai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import PyPDF2
from publication import Publication
from VectorDatabase import Latern
from fragment import Fragment

# OpenAI Setup
OPEN_API_KEY = "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"
# openai.api_key = os.getenv(openai_api_key)
os.environ['OPENAI_API_KEY'] = OPEN_API_KEY

def scrapeBiorxiv(start, end, out_file):
    filepath=out_file
    biorxiv(begin_date=start, end_date=end, save_path=out_file)
    retreiveTextFromPdf(filepath)

def get_embeddings(fname):
    """
    """
    loader = TextLoader(fname)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator = ".",chunk_size = 1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    emb = OpenAIEmbeddings()
    input_texts = [d.page_content for d in docs]

    input_embeddings = emb.embed_documents(input_texts)
    text_embeddings = list(zip(input_texts, input_embeddings))
    return text_embeddings, emb

def retreiveTextFromPdf(inp_file):
    

    json = pd.read_json(path_or_buf=inp_file, lines=True)
    print(len(json['doi']))
    
    for n, doi in enumerate(json['doi']):
        print(n, doi)
        
#         if latern.publication_exists(doi):
#             continue
        
        paper_data = {'doi': doi}
        doi = doi.replace("/", "-")
        pdf_dir = './papers/'
        if not os.path.exists(pdf_dir):
            os.mkdir(pdf_dir)

        pdfsavefile='./papers/' + doi +'.pdf'
        print(paper_data, pdfsavefile)
        save_pdf(paper_data, filepath=pdfsavefile)
        
        # creating a pdf reader object
        reader = PyPDF2.PdfReader(pdfsavefile)
        save_txt_path = 'scrapped_txts/'
        if not os.path.exists(save_txt_path):
            os.mkdir(save_txt_path)
        extract_text = ''
        for page in reader.pages:
            extract_text+=page.extract_text()

        txt_file = str('{}.txt'.format(doi))
        with open(save_txt_path+txt_file, 'w') as file:
            file.write(extract_text)
        
        
        txt_embs, emb = get_embeddings(save_txt_path+txt_file)
            
        for txt,embs in txt_embs:

        fragments = []
        for txt, embs in txt_embs:
            fragment = Fragment(doi, 'methods', txt, embs)
            fragments.append(fragment)
            
        title = ""
        pmc = ""
        pubmed = ""

        publication = Publication(doi, title, pmc, pubmed, doi)
        
        latern = Latern()
        latern.insertEmbeddings(fragments)
        latern.insertPublication(publication)
        
        os.remove(pdfsavefile)

start_date = "2023-10-30"
end_date = "2023-10-31"
out_file = "bio.jsonl"

scrapeBiorxiv(start_date, end_date, out_file)
