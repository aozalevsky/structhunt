# Latern Vector Database 

## Installation

Run `initialize_database.sh` to:
1. Setup Postgres
2. Create databases
3. Install dependancies

## Classes
Fragment and Publication classes which contain a Python representation of datarow from table. 

## Database Structure

Latern creates the following two tables in the database:

1. `fragments` table:
   - Columns: id (text), header (text), content (text), vector (real[])
   - Used to store information about molecular fragments, including their ID(DOI), header, content, and associated vector data.

2. `publications` table:
   - Columns: id (text, primary key), title (text), pmc (text), pubmed (text), doi (text)
   - Used to store information about publications related to the fragments, including their ID(DOI), title, and links to PMC, PubMed, and DOI.

## Usage

VectorDatabase file, which has class Lantern, provides the main functionality for the vector database. For example, you can insert an embedding with the insertEmbedding().