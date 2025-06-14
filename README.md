# Semantic Data Management - Lab 1

# Requires the following libraries:
- neo4j-python-driver
- faker
- keybert

# Contents
- PartA.2(1)_MeneghiniSpisso.py: 
    This is the first part of the solution to Task A2, where we download the data from the Semantic Scholar API and
    save them in CSV files, ready for bulk load in Neo4j database in the appropriate import folder. Finally, we create 
    synthetic keywords, corresponding authors and reviewers for each paper, as well as conference locations, which could 
    not be downloaded from the Semantic Scholar API.
    
    Usage: python "PartA.2(1)_MeneghiniSpisso.py" -v SIGMOD VLDB ICDE AAAS Nature -n_paper 1000 -l_cited 20 -n path_to_import_directory


- PartA.2(2)_MeneghiniSpisso.py:
    This is the second part of Task A2, where we execute the Cypher queries to bulk import the data created and saved in
    the import folder of the database.

    Usage: python "PartA.2(2)_MeneghiniSpisso.py" --user your_username --password your_password --database your_database_name   


- PartA.3_MeneghiniSpisso.py:
    This is the solution to Task A3, where we update the database with the reviewer nodes, deliting previously existing
    WROTE_REVIEW edges, and add authors' affiliations, without using any bulk load functionality to access and manipulate 
    the database.

    Usage: python "PartA.3_MeneghiniSpisso.py" --user your_username --password your_password --database your_database_name


- PartB_MeneghiniSpisso.py:
    This is the solution to Task B.

    Usage: python "PartB_MeneghiniSpisso.py" --user your_username --password your_password --database your_database_name


- PartC_MeneghiniSpisso.py:
    This is the solution to Task C.

    Usage: python "PartC_MeneghiniSpisso.py" --user your_username --password your_password --database your_database_name -c database 
            -k  "graph,database"


- PartD_MeneghiniSpisso.py:
    This is the solution to Task D, using the paper citation graph and running the Pagerank and Louvain algorithms.

    Usage: "PartD_MeneghiniSpisso.py" --user your_username --password your_password --database your_database_name


- PartD_alternative_MeneghiniSpisso.py:
    This is an alternative solution to Task D, creating and using the author citation graph and running the Pagerank and Louvain algorithms.

    Usage: "PartD_alternative_MeneghiniSpisso.py" --user your_username --password your_password --database your_database_name

- DOC_MeneghiniSpisso.pdf:
    The file containing the Lab report.