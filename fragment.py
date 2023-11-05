class Fragment:

    VECTOR_LENGTH=384

    pdbid = ""
    header = ""
    content = ""
    vector = ""

    def __init__(self, pdbid, header, content, vector):
        self.pdbid = pdbid
        self.header = header
        self.content = content
        self.vector = vector