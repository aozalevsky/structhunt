
from VectorDatabase import Lantern, Publication, Fragment
from google_sheets import SheetsApiClient
from prompts import get_qbi_hackathon_prompt

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from datetime import date


class DocumentAnalyzer:
    """Takes in a list of publications to analyze, then prompts the chatbot, processes the response, aggregates the results,
    and reports the results to the spreadsheet
    """  
    
    def __init__(self):
        self.lantern = Lantern()
        self.sheets = SheetsApiClient()
        self.llm = LlmHandler()
    
    def analyze_all_unread(self):
        """pulls all new files from Lantern database, evaluates them, and publishes results to google sheets
        """
        publications = lantern.getUnreadPublications()
        self.process_publications(publications)

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
            # add date if it's added 
            rows.append([pub.doi, pub.title, "", str(date.today()), "", int(classification), response, ""])

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
            {hits} {"were" if (hits != 1) else "was"} classified as having multi-method structural data"""

        if notify:
            sheets.notify_arthur(message=msg)
        

    def analyze_publication(self, text_embeddings: []):
        """poses a question about the document, processes the result and returns it
        NOTE: for now, only uses the hackathon question, might add more later

        Args:
            text_embeddings ([]): list of (embedding, text) pairs from document to be analyzed
        
        Returns:
            bool: classification of response to query as positive (True) or negative (False) 
            str: response from chatGPT
        """
        # NOTE: These very likely need to change
        open_ai_emb = OpenAIEmbeddings()
        query = get_qbi_hackathon_prompt()
        faissIndex = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=open_ai_emb)
        response = self.llm.evaluate_queries(faissIndex, query)[0]
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


class LlmHandler:
    """pulled this straight from the hackathon code, should work though
    """

    def __init__(self):
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
        responses = []
        for q in queries:
            prompt = PromptTemplate(
                input_variables=["query"],
                template=template,
            )

            responses.append(chatbot.run(
                prompt.format(query=q)
            ))
        return responses




def main():
    document_analyzer = DocumentAnalyzer()
    document_analyzer.analyze_all_unread() #analyzes all new files in lantern db

if __name__ == '__main__':
    main()