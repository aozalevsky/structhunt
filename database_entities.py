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

# Class to represent a fragment of a publication with attributes id, header, content, and vector
class Fragment:


    # Class variables to store default values for attributes
    id = ""        
    header = ""    
    content = ""   
    vector = ""    

    def __init__(self, id, header, content, vector):
        # Constructor to initialize the attributes of the Fragment object

        # Set the attributes of the object with the values provided during instantiation
        self.id = id          # (DOI) Unique identifier for the fragment
        self.header = header  # Header or title of the fragment
        self.content = content # Content or text of the fragment
        self.vector = vector  # Vector representation of the fragment
