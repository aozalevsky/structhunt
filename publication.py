
class Publication:

    id = ""
    title = ""
    pmc = ""
    pubmed = ""
    doi = ""

    def __init__(self, id, title, pmc, pubmed, doi):
        self.id = id
        self.title = title
        self.pmc = pmc
        self.pubmed = pubmed
        self.doi = doi