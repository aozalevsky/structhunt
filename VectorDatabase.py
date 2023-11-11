import psycopg2
from database_entities import Fragment, Publication

# Lantern class that exposes functionality of database to application
class Lantern:
    conn = ""

    def __init__(self, database="structdb"):
        self.conn = self.connect(database)  # Connect to database
        self.createTables()  # Create tables if necessary

    def connect(self, database="structdb"):
        # We use the dbname, user, and password
        # user and password are established in initialize_database.sh
        # local version of postgres/lantern
        conn = psycopg2.connect(
            dbname=database,
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"  # default port for Postgres
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
        self.createUnreadTable()

    """
    Creates a table named 'fragments' in the connected database to store fragment information.
    Parameters:
        - None (uses the database connection stored in self.conn
    Returns:
        - None
    Notes:
        - The 'fragments' table will only be created if it does not already exist.
    """

    def createFragmentTable(self):
        conn = self.conn
        cursor = conn.cursor()

        create_table_query = "CREATE TABLE IF NOT EXISTS fragments (id text, header text, content text, vector real[]);"

        cursor.execute(create_table_query)

        conn.commit()
        cursor.close()

    """
    Creates a table named 'publications' in the connected database to store publication information.
    Parameters:
        - None (uses the database connection stored in self.conn)
    Returns:
        - None
    Notes:
        - The 'publications' table will only be created if it does not already exist.
    """

    def createPublicationTable(self):
        conn = self.conn
        cursor = conn.cursor()

        create_table_query = "CREATE TABLE IF NOT EXISTS publications (id text PRIMARY KEY, title text, pmc text, pubmed text, doi text);"

        cursor.execute(create_table_query)

        conn.commit()
        cursor.close()

    """
    Creates a table named 'unread' in the connected database to store unread publication identifiers.
    Parameters:
        - None (uses the database connection stored in self.conn)
    Returns:
        - None
    Notes:
        - The 'unread' table will only be created if it does not already exist.
    """

    def createUnreadTable(self):
        conn = self.conn
        cursor = conn.cursor()

        create_table_query = "CREATE TABLE IF NOT EXISTS unread (id text PRIMARY KEY);"
        cursor.execute(create_table_query)

        conn.commit()
        cursor.close()

    """
    Inserts a fragment and its embedding into the 'fragments' table in the connected database.
    Parameters:
        - fragment: Fragment, the fragment object containing information to be inserted.
    Returns:
        - None
    Notes:
        - Inserts the provided fragment's id, header, content, and vector into the 'fragments' table.
        - Creates an HNSW index on the 'vector' column for efficient cosine distance operations.
    """

    def insertEmbedding(self, fragment: Fragment):
        conn = self.conn
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO fragments (id, header, content, vector) VALUES (%s, %s, %s, %s);",
            (fragment.id,
             fragment.header,
             fragment.content,
             fragment.vector))
        cursor.execute(
            "CREATE INDEX ON fragments USING hnsw (vector dist_cos_ops) WITH (dim=" + str(len(fragment.vector)) + ");")

        conn.commit()
        cursor.close()

    """
    Inserts a list of fragments and their embeddings into the 'fragments' table.
    Parameters:
        - fragments: List[Fragment], a list of fragment objects containing information to be inserted.
    Returns:
        - None
    Notes:
        - If the provided list is empty, prints a message and returns without performing insertion.
        - Inserts each fragment's id, header, content, and vector into the 'fragments' table using executemany.
        - Creates an HNSW index on the 'vector' column for efficient cosine distance operations.
        - Prints an error message if there is an exception during insertion.
    """

    def insertEmbeddings(self, fragments: list):
        if (len(fragments) < 1):
            print("Empty List")
            return
        conn = self.conn
        cursor = conn.cursor()

        queries = []
        for fragment in fragments:
            queries.append(
                (fragment.id,
                 fragment.header,
                 fragment.content,
                 fragment.vector))

        try:
            cursor.executemany(
                "INSERT INTO fragments (id, header, content, vector) VALUES (%s, %s, %s, %s);",
                queries)
        except Exception:
            print("Error with insertion")
        cursor.execute("CREATE INDEX ON fragments USING hnsw (vector dist_cos_ops) WITH (dim=" +
                       str(len(fragments[0].vector)) + ");")
        conn.commit()
        cursor.close()

    """
    Inserts a publication and its information into the 'publications' and 'unread' tables.
    Parameters:
        - p: Publication, the publication object containing information to be inserted.
    Returns:
        - None
    Notes:
        - If a publication with the same id already exists, returns without performing insertion.
        - Inserts the provided publication's id, title, pmc, pubmed, and doi into the 'publications' table.
        - Inserts the id of the publication into the 'unread' table to mark it as unread.
    """

    def insertPublication(self, p):
        if self.publicationExists(p.id):
            return

        conn = self.conn
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO publications (id, title, pmc, pubmed, doi) VALUES (%s, %s, %s, %s, %s);",
            (p.id,
             p.title,
             p.pmc,
             p.pubmed,
             p.doi))

        query = 'INSERT INTO unread (id) VALUES (\'{:s}\');'.format(p.id)
        cursor.execute(query)
        conn.commit()
        cursor.close()

    """
    Retrieves all fragments of a publication from the 'fragments' table.
    Parameters:
        - id: Text, the unique identifier of the publication.
    Returns:
        - List[Fragment], a list of Fragment objects representing the fragments of the specified publication.
    Notes:
        - Queries the 'fragments' table to retrieve all fragments with the provided publication id.
        - Constructs a list of Fragment objects from the retrieved data.
    """

    def getAllFragmentsOfPublication(self, id):
        conn = self.conn
        cursor = conn.cursor()

        query = 'SELECT * FROM fragments WHERE id=\'{:s}\';'.format(id)
        cursor.execute(query)
        fragments = cursor.fetchall()
        conn.commit()
        cursor.close()

        fragmentObjects = []
        for fragment in fragments:
            fragmentObjects.append(
                Fragment(
                    id,
                    fragment[1],
                    fragment[2],
                    fragment[3]))

        return fragmentObjects

    """
    Retrieves unread publications from the 'publications' table.
    Parameters:
        - delete_unread_entries: bool, decides if entries are deleted from the "unread" table
    Returns:
        - List[Publication], a list of Publication objects representing the unread publications.
    Notes:
        - Performs a left join between 'publications' and 'unread' tables to retrieve unread publications.
        - Clears the 'unread' table after retrieving the unread publications.
    """

    def getUnreadPublications(self, delete_unread_entries=True):
        conn = self.conn
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM publications AS p LEFT JOIN unread AS u ON u.id=p.id;')

        publications = cursor.fetchall()

        if delete_unread_entries:
            cursor.execute('DELETE FROM unread;')
            
        conn.commit()
        cursor.close()

        publicationObjects = []
        for p in publications:
            publicationObjects.append(
                Publication(p[0], p[1], p[2], p[3], p[4]))

        return publicationObjects

    """
    Checks if a publication with the given id exists in the 'publications' table.
    Parameters:
        - id: Text, the unique identifier of the publication.
    Returns:
        - bool, True if a publication with the provided id exists, False otherwise.
    Notes:
        - Queries the 'publications' table to count the occurrences of the provided id.
        - Returns True if the count is equal to 1 (indicating existence), False otherwise.
    """

    def publicationExists(self, id):
        conn = self.conn
        cursor = conn.cursor()

        query = 'SELECT COUNT(*) FROM publications WHERE id=\'{:s}\''.format(
            id)
        cursor.execute(query)
        count = cursor.fetchone()
        conn.commit()
        cursor.close()

        return count[0] == 1
