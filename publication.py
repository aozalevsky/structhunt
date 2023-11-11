# Class to represent a publication with attributes id, title, pmc, pubmed, and doi
class Publication:

    id = ""
    title = ""
    pmc = ""
    pubmed = ""
    doi = ""

    def __init__(self, id, title, pmc, pubmed, doi):
        self.id = id # (DOI) Unique identifier for the publication
        self.title = title    # Title of the publication
        self.pmc = pmc        # PubMed Central (PMC) Link
        self.pubmed = pubmed  # PubMed Link
        self.doi = doi # Digital Object Identifier (DOI) Link for the publication
