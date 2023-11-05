import os
import pandas as pd    
import PyPDF2
from paperscraper.pdf import save_pdf
from paperscraper.get_dumps import biorxiv

def scrapeBiorxiv(start, end, out_file):
    filepath=out_file
    biorxiv(begin_date=start, end_date=end, save_path=out_file)
    retreiveTextFromPdf(filepath)

def retreiveTextFromPdf(inp_file):
    json = pd.read_json(path_or_buf=inp_file, lines=True)
    print(len(json['doi']))
    
    for n, doi in enumerate(json['doi']):
        print(n, doi)
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

        os.remove(pdfsavefile)
    
start_date = "2023-10-30"
end_date = "2023-10-31"
out_file = "bio.jsonl"

scrapeBiorxiv(start_date, end_date, out_file)
