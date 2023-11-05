from fragment import Fragment
from VectorDatabase import Latern
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import torch
import re
import requests
import xml.etree.ElementTree as ET
import string
import random


vd = Latern()

def get_pmc_paper(pmcid):
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
    req = requests.get(url)
    res = req.text
    return res

def get_sentence_from_text(text):
    return text.split(".")

def random_string():       
    N=7
    return ''.join(random.choices(string.ascii_letters, k=N))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print(f"You are using {device}. This is much slower than using "
          "a CUDA-enabled GPU. If on Colab you can change this by "
          "clicking Runtime > Change runtime type > GPU.")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
model

pmcid = 'PMC10081221'
text = get_pmc_paper(pmcid)
sentences = get_sentence_from_text(text)


fragments = []
for i in tqdm(range(0, len(sentences))):
    content = sentences[i]
    vector = [float(x) for x in model.encode(sentences[i])]
    id = random_string()
    fragments.append(Fragment(id, content[0], content, vector))

vd.insertEmbeddings(fragments)



