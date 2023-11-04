

import os
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

import re
import requests
import xml.etree.ElementTree as ET


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate


# OPENAI SETUP

openai_api_key = "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"
os.environ['OPENAI_API_KEY'] = openai_api_key


def get_pmc_paper(pmcid):
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
    req = requests.get(url)
    res = req.text
    return res

def extract_methods_from_pmc_paper(paper):
    tree = ET.fromstring(paper)

    mtext = []
    for sec in tree.iter('sec'):
        for title in sec.iter('title'):
            if isinstance(title.text, str):
                if re.search('methods', title.text, re.IGNORECASE):
                    mtext.extend(list(sec.itertext()))

    return " ".join(mtext)

def run_test(pmcid: str, queries: [str]):
    ## Write to file
    # pmcid = 'PMC9935389' 
    #pmcid = 'PMC10081221'
    text = get_pmc_paper(pmcid)
    methods_text = extract_methods_from_pmc_paper(text)
    with open('input_file.txt', 'w') as file:
        file.write(methods_text)


    loader = TextLoader("./input_file.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
    current_document = "input_doc"
    faissIndex.save_local(current_document)


    chatbot = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0, model_name="gpt-3.5-turbo", max_tokens=50
        ), 
        chain_type="stuff", 
        retriever=FAISS.load_local(current_document, OpenAIEmbeddings())
            .as_retriever(search_type="similarity", search_kwargs={"k":1})
    )

    for q in queries:

        template = """ {query}? """

        prompt = PromptTemplate(
            input_variables=["query"],
            template=template,
        )

        print(chatbot.run(
            prompt.format(query="Does the paper report a new structure of a biomolecule or biomolecular complex modeled using experimental data")
        ))


    # function that takes context, prompts, and returns answers 

    # comparison of answers to known answers

    # 


def main():
    #silly example
    queries = ["how many", "how much"]
    run_test(pmcid='PMC23402394802394', queries=queries)
