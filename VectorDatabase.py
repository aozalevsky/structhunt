from fragment import Fragment
from publication import Publication

import psycopg2

class Latern: 
    conn = ""

    def __init__(self, database="structdb"):
        self.conn = self.connect(database)
        self.createTables()


    def connect(self, database="structdb"):
        # We use the dbname, user, and password that we specified above
        conn = psycopg2.connect(
            dbname=database,
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

        create_table_query = "CREATE TABLE IF NOT EXISTS fragments (id text, header text, content text, vector real[]);"

        cursor.execute(create_table_query)

        conn.commit()
        cursor.close()

    def createPublicationTable(self):
        conn = self.conn
        cursor = conn.cursor()

        create_table_query = "CREATE TABLE IF NOT EXISTS publications (id text PRIMARY KEY, title text, pmc text, pubmed text, doi text)"

        cursor.execute(create_table_query)

        conn.commit()
        cursor.close()

    def insertEmbedding(self, fragment: Fragment):
        conn = self.conn
        cursor = conn.cursor()

        cursor.execute("INSERT INTO fragments (id, header, content, vector) VALUES (%s, %s, %s, %s);", (fragment.id, fragment.header, fragment.content, fragment.vector))
        cursor.execute("CREATE INDEX ON fragments USING hnsw (vector dist_cos_ops) WITH (dim=" + str(fragment.VECTOR_LENGTH) + ");")

        conn.commit()
        cursor.close()

    def insertEmbeddings(self, fragments: list):
        if (len(fragments) < 1):
            print("Empty List")
            return
        conn = self.conn
        cursor = conn.cursor()

        queries=[]
        for fragment in fragments:
            queries.append((fragment.id, fragment.header, fragment.content, fragment.vector))
        
        cursor.executemany("INSERT INTO fragments (id, header, content, vector) VALUES (%s, %s, %s, %s);", queries)
        cursor.execute("CREATE INDEX ON fragments USING hnsw (vector dist_cos_ops) WITH (dim=" + str(len(fragments[0].vector)) + ");")
        conn.commit()
        cursor.close()

    def insertPublication(self, publication):
        conn = self.conn
        cursor = conn.cursor()
        p = publication

        cursor.execute("INSERT INTO publications (id, title, pmc, pubmed, doi) VALUES (%s, %s, %s, %s, %s);", (p.id, p.title, p.pmc. p.pubmed, p.doi))

        conn.commit()
        cursor.close()

    def getAllFragmentsOfPublication(self, id):
        conn = self.conn
        cursor = conn.cursor()

        query='SELECT * FROM fragments WHERE id=\'{:s}\';'.format(id)
        cursor.execute(query)
        fragments = cursor.fetchall()
        conn.commit()
        cursor.close()

        fragmentObjects = []
        for fragment in fragments:
            fragmentObjects.append(Fragment(id, fragment[1], fragment[2], fragment[3]))
        
        return fragmentObjects

    