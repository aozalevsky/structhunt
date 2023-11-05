

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
import timeout_decorator


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


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

def preprocess(input_text):
    processed_data = input_text.replace("\n","")
    return processed_data

def embed_article(pmcid):
    text = get_pmc_paper(pmcid)
    methods_text = preprocess(extract_methods_from_pmc_paper(text))
    filepath = f'./data/{pmcid}'
    txtfilepath = filepath + '.txt'
    with open(txtfilepath, 'w', encoding="utf-8") as file:
        file.write(methods_text)

    print('text copied')

    loader = TextLoader(txtfilepath, autodetect_encoding=True)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator = ".", chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
    faissIndex.save_local(filepath)
    return FAISS.load_local(filepath, OpenAIEmbeddings())


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
    #embeddings have been saved

    chatbot = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0, model_name="gpt-3.5-turbo", max_tokens=50, timeout = 9, max_retries=0
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

    print('finished queries')


    # function that takes context, prompts, and returns answers 

    # comparison of answers to known answers

    # 

def fetch_embedding(pmcid):
    try:
        filepath = f'./data/{pmcid}'
        return FAISS.load_local(filepath, OpenAIEmbeddings())
    except e:
        print(e)
        print('failure')
        return embed_article(pmcid)


def compare_against_known():
    # only asking "are there more methodologies beyond "
    queries = ["Are there experimental techniques beyond using Cryo-Em incorporated in the paper? Answer with Yes or No followed by the experimental technique."]
    pmc_ids_false = ['PMC8536336', 'PMC7417587', 'PMC5957504', 'PMC7492086', 'PMC9293004']
    pmc_ids_true = ['PMC7854634', 'PMC5648754', 'PMC8022279', 'PMC8655018', 'PMC8916737']

    for pmc in pmc_ids_false:
        #run_test(pmc, )
        pass

def embed_all():
    pmc_ids = ['PMC8536336', 'PMC7417587', 'PMC5957504', 'PMC7492086', 'PMC9293004', 
    'PMC7854634', 'PMC5648754', 'PMC8022279', 'PMC8655018', 'PMC8916737']
    for pmcid in pmc_ids:
        embedding = fetch_embedding(pmcid)
        result = evaluate_query(embedding, queries)
        print(result)


queries = ["Are there experimental techniques beyond using Cryo-Em incorporated in the paper? Answer with Yes or No followed by the experimental technique."]

def evaluate_query(embedding, queries):
    chatbot = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0, model_name="gpt-3.5-turbo", max_tokens=50, request_timeout = 9, max_retries=0
        ), 
        chain_type="stuff", 
        retriever=embedding.as_retriever(search_type="similarity", search_kwargs={"k":1})

    )
    
    template = """ {query}? """
    for q in queries:
        prompt = PromptTemplate(
            input_variables=["query"],
            template=template,
        )

        return(chatbot.run(
            prompt.format(query=q)
        ))




    


embed_all()

def main():
    #silly example
    queries = ["Are there experimental techniques beyond using Cryo-Em incorporated in the paper? Answer with Yes or No followed by the experimental technique."]
    run_test(pmcid='PMC23402394802394', queries=queries)


print(fetch_embedding('PMC8536336'))