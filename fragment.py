class Fragment:
    pdbid = ""
    header = ""
    content = ""
    vector = ""

    def __init__(self, pdbid, header, content, vector):
        self.pdbid = pdbid
        self.header = header
        self.content = content
        self.vector = vector