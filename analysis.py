
from VectorDatabase import Lantern
from database_entities import Publication, Fragment
from google_sheets import SheetsApiClient

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

class DocumentAnalyzer:
    """sfdaf
    """
    
    keywords_groups = {
        'CX-MS': ['cross-link', 'crosslink', 'XL-MS', 'CX-MS', 'CL-MS', 'XLMS', 'CXMS', 'CLMS', "chemical crosslinking mass spectrometry", 'photo-crosslinking', 'crosslinking restraints', 'crosslinking-derived restraints', 'chemical crosslinking', 'in vivo crosslinking', 'crosslinking data'],
        'HDX': ['Hydrogen–deuterium exchange mass spectrometry', 'Hydrogen/deuterium exchange mass spectrometry' 'HDX', 'HDXMS', 'HDX-MS'],
        'EPR': ['electron paramagnetic resonance spectroscopy', 'EPR', 'DEER', "Double electron electron resonance spectroscopy"],
        'FRET': ['FRET',  "forster resonance energy transfer", "fluorescence resonance energy transfer"],
        'AFM': ['AFM',  "atomic force microscopy" ],
        'SAS': ['SAS', 'SAXS', 'SANS', "Small angle solution scattering", "solution scattering", "SEC-SAXS", "SEC-SAS", "SASBDB", "Small angle X-ray scattering", "Small angle neutron scattering"],
        '3DGENOME': ['HiC', 'Hi-C', "chromosome conformation capture"],
        'Y2H': ['Y2H', "yeast two-hybrid"],
        'DNA_FOOTPRINTING': ["DNA Footprinting", "hydroxyl radical footprinting"],
        'XRAY_TOMOGRAPHY': ["soft x-ray tomography"],
        'FTIR': ["FTIR", "Infrared spectroscopy", "Fourier-transform infrared spectroscopy"],
        'FLUORESCENCE': ["Fluorescence imaging", "fluorescence microscopy", "TIRF"],
        'EVOLUTION': ['coevolution', "evolutionary covariance"],
        'PREDICTED': ["predicted contacts"],
        'INTEGRATIVE': ["integrative structure", "hybrid structure", "integrative modeling", "hybrid modeling"],
        'SHAPE': ['Hydroxyl Acylation analyzed by Primer Extension']
    }    
    
    def __init__(self):
        # self.lantern = Lantern()
        self.sheets = SheetsApiClient()
    

    def process_publications(self, publications: [Publication]):
        """takes a list of publications, applies retrievalQA and processes responses
        NOTE: completely untested, just refactored code from hackathon

        Args:
            publications ([]): list of publications 
        """
        query = [f"You are reading a materials and methods section of a scientific paper. Here is the list of structural biology methods {methods_string}.\n\n Did the authors use any methods from the list? \n\n Answer with Yes or No followed by the names of the methods."]

        rows = []
        hits = 0
        for pub in publications:
            text_embeddings = self.lantern.get_embeddings_for_pub(pub.id)
            classification, response = 0, ''
            if self.paper_about_cryoem(text_embeddings):    
                classification, response = self.analyze_publication(text_embeddings)
                hits += classification
            else:
                #print('paper not about cryo-em')
                pass
            rows.append([pub.doi, pub.title, "11-2-2023", "11-5-2023", "", int(classification), response, ""])

        self.update_spreadsheet(rows, hits)
        
    def update_spreadsheet(rows: [], hits: int, notify=True):
        """pushes a list of rows to the spreadsheet and notifies via email

        Args:
            rows ([]): rows of data to be uploaded to sheet 
            hits (int): number of positive classifications in the rows
            notify (bool): notify via email if True
        """

        if hits > len(rows):
            raise ValueError(f"Number of hits ({hits}) is greater than the number of entries ({len(rows)}), sus")
        
        #print(rows)
        self.sheets.append_rows(rows)
        msg = f"""
            This batch of paper analysis has concluded. 
            {len(rows)} papers were analyzed in total over the date range 11/2 - 11/3
            {hits} {"were" if ((hits>0) or (hits == 0)) else was} classified as having multi-method structural data"""

        if notify:
            sheets.notify_arthur(message=msg)
        

    def analyze_publication(self, publication: Publication):
        """leaving this blank for now because i think the way these are stored is changing

        Args:
            publication (Publication): publication to be analyzed
        
        Returns:
            bool: classification of response to query as positive (True) or negative (False) 
            str: response from chatGPT
        """
        #faissIndex = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=open_ai_emb)
        #result = llm.evaluate_queries(faissIndex, query)
        response = None
        return self.classify_response(response), response

    @staticmethod
    def classify_response(response: str):
        """converting text response from GPT into boolean

        Args:
            response (str): response from ChatGPT to the query

        Returns:
            bool: True if answer to question is "yes" 
        """
        if result == None:
            return False
        # this was used to filter out cases where ChatGPT said "Yes, Cryo-EM was used..." which is wrong because we asked it about 
        # inclusion of non-cryo-em stuff 
        #if "cryo" in response.lower():
        #    return (False, None)
        return response.lower().startswith('yes')
    
    @staticmethod
    def paper_about_cryoem(text_embeddings: []):
        """checks if the string "cryoem" or "cryo-em" is present in the text

        Args:
            text_embeddings [(text, embedding)]: text and embeddings of a publication

        Returns:
            bool: True if the text mentions cryo-em 
        """
        return any(re.search("cryo-?em", text, re.IGNORECASE) for text, _ in embeddings)

    @staticmethod
    def methods_string():
        methods_string = ''
        for i, (k, v) in enumerate(DocumentAnalyzer.keywords_groups.items()):
            if i > 0:
                methods_string += ' or '
            methods_string += f'{k} ({", ".join(v)})'
        return methods_string


class LlmHandler:
    """pulled this straight from the hackathon code, should work though
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", ".", ","], chunk_size=300, chunk_overlap=100)
        self.llm=ChatOpenAI(
                temperature=0, model_name="gpt-4", max_tokens=300, request_timeout = 30, max_retries=3
            )
        
        
    def evaluate_queries(self, embedding, queries):
        chatbot = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=embedding.as_retriever(search_type="similarity", search_kwargs={"k":3})
        )
        
        template = """ {query}? """
        response = []
        for q in queries:
            prompt = PromptTemplate(
                input_variables=["query"],
                template=template,
            )

            response.append(chatbot.run(
                prompt.format(query=q)
            ))
        return response




def main():
    x = DocumentAnalyzer()
    l = LlmHandler()

if __name__ == '__main__':
    main()