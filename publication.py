
class Publication:

    pdbid = ""
    title = ""
    pmc = ""
    pubmed = ""
    doi = ""

    def __init__(self, pdbid, title, pmc, pubmed, doi):
        self.pdbid = pdbid
        self.title = title
        self.pmc = pmc
        self.pubmed = pubmed
        self.doi = doi