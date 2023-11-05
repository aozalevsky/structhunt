
class Publication:

    id = ""
    title = ""
    pmc = ""
    pubmed = ""
    doi = ""
    date_added = ""

    def __init__(self, id, title, pmc, pubmed, doi, date_added):
        self.id = id
        self.title = title
        self.pmc = pmc
        self.pubmed = pubmed
        self.doi = doi
        self.date_added = date_added