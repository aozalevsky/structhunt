import psycopg2
import Fragment, Publication

class Latern: 
    conn = ""

    def __init__(self, database="structdb"):
        self.conn = self.connect()
        self.createTable()


    def connect(self, database="structdb"):
        # We use the dbname, user, and password that we specified above
        conn = psycopg2.connect(
            dbname="testdb",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432" # default port for Postgres
        )

        cursor = conn.cursor()
        # Execute the query to load the Lantern extension in
        cursor.execute("CREATE EXTENSION IF NOT EXISTS lantern;")

        conn.commit()
        cursor.close()

        return conn
    

    def createTables(self):
        self.createFragmentTable()
        self.createPublicationTable()

    def createFragmentTable(self):
        conn = self.conn
        # Create the table
        cursor = conn.cursor()

        create_table_query = "CREATE TABLE IF NOT EXISTS fragments (pdbid text FORIEGN key, header text, content text, vector real[]);"

        cursor.execute(create_table_query)

        conn.commit()
        cursor.close()

    def createPublicationTable(self):
        conn = self.conn
        cursor = conn.cursor()

        create_table_query = "CREATE TABLE IF NOT EXISTS publications (pdbid text PRIMARY KEY, title text, pmc text, pubmed text, doi text)"

        cursor.execute(create_table_query)

        conn.commit()
        cursor.close()

    def insertEmbedding(self, fragment: Fragment):
        conn = self.conn
        cursor = conn.cursor()

        cursor.execute("INSERT INTO fragments (pdbid, header, content, vector) VALUES (%s, %s, %s, %s);", (fragment.pdbid, fragment.text, fragment.content, fragment.vector))
        cursor.execute("CREATE INDEX ON fragments USING hnsw (vector dist_cos_ops) WITH (dim=" + fragment.VECTOR_LENGTH + ");")

        conn.commit()
        cursor.close()

    def insertEmbeddings(self, fragments: list):
        for fragment in fragments:
            self.insertEmbedding(fragment)

    def insertPublication(self, publication):
        conn = self.conn
        cursor = conn.cursor()
        p = publication

        cursor.execute("INSERT INTO publications (pdbid, title, pmc, pubmed, doi) VALUES (%s, %s, %s, %s, %s);", (p.pdbid, p.title, p.pmc. p.pubmed, p.doi))

        conn.commit()
        cursor.close()

    

    