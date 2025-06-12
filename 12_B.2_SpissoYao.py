# %%
import pandas as pd
from rdflib import Graph, Namespace, RDF, RDFS, XSD, URIRef, Literal
import json
import numpy as np
import ast
import urllib.parse
import os

pd.set_option('display.max_columns', None)

tbox = Graph()
tbox.parse("12_B.1_SpissoYao_tbox.ttl", format="turtle")  # or 'xml', 'n3', etc.

EX = Namespace("http://example.org/research_paper/")

research_paper = pd.read_csv("research_papers.csv")


classes = set(tbox.subjects(RDF.type, RDFS.Class))

properties = {}
for prop in tbox.subjects(RDF.type, RDF.Property):
    domain = tbox.value(prop, RDFS.domain)
    range_ = tbox.value(prop, RDFS.range)
    properties[prop] = (domain, range_)


print("Classes in TBox:")
for c in classes:
    print(f"  {c}")

print("Properties in TBox and their domains/ranges:")
for p, (d, r) in properties.items():
    print(f"  {p} domain={d} range={r}")

print("Subclasses in TBox:")
for c in classes:
    subclasses = list(tbox.subjects(predicate=RDFS.subClassOf, object=c))
    if subclasses:
        print(f"  {c} subclasses={subclasses}")

abox = Graph()
abox.bind("ex", EX)
abox.bind("rdf", RDF)
abox.bind("rdfs", RDFS)

def urlib_parse(value):
    return urllib.parse.quote(value, safe='')

# Function to create a property instance
def create_property_instance(graph, namespace, subject_uri_value, property_name, object_value, is_literal=False):
    subject_uri = URIRef(namespace[urlib_parse(subject_uri_value)])
    predicate = namespace[property_name]

    if is_literal:
        graph.add((subject_uri, predicate, Literal(object_value, datatype=XSD.string)))
    else:
        object_uri = URIRef(namespace[urlib_parse(object_value)])
        graph.add((subject_uri, predicate, object_uri))

owners_set = set(research_paper['author'].dropna().unique())

reviewers_set = set()
for reviewers_list_str in research_paper['reviewers'].dropna():
    try:
        reviewers_list = ast.literal_eval(reviewers_list_str)
        if isinstance(reviewers_list, list):
            reviewers_set.update(reviewers_list)
    except (ValueError, SyntaxError):
        # Handle cases where ast.literal_eval might fail if the string is not a valid list literal
        print(f"Warning: Could not parse reviewers list: {reviewers_list_str}")

unique_authors = set()

for authors_str in research_paper['authors']:
    authors = ast.literal_eval(authors_str)
    unique_authors.update(authors)

author_ids = {author: f"AUTHOR_{i}" for i, author in enumerate(sorted(unique_authors))}

unique_topics = set()

for topics_str in research_paper['topic']:
    topics = ast.literal_eval(topics_str)
    unique_topics.update(topics)

topic_ids = {topic: f"TOPIC_{i}" for i, topic in enumerate(sorted(unique_topics))}

np.random.seed(42)  # For reproducibility
unique_conferences = research_paper['conference'].dropna().unique()
conference_ids = {name: f"CONFERENCE_{idx}" for idx, name in enumerate(unique_conferences, start=1)}
conference_chair = {}

grouped = research_paper.groupby('conference')

for conference, group in grouped:
    conference_people = set()

    # Collect all authors and reviewers in this conference group
    for authors_str in group['authors']:
        conference_people.update(ast.literal_eval(authors_str))

    for reviewers_str in group['reviewers']:
        conference_people.update(ast.literal_eval(reviewers_str))

    # Choose a chair from authors not involved in this conference
    eligible_chairs = list(unique_authors - conference_people)

    conference_chair[conference] = (
        np.random.choice(eligible_chairs) if eligible_chairs else None
    )


np.random.seed(42)  # For reproducibility

unique_journals = research_paper['journal'].dropna().unique()
journal_ids = {name: f"JOURNAL_{idx}" for idx, name in enumerate(unique_journals, start=1)}

journal_editor = {}

grouped = research_paper.groupby('journal')

for journal, group in grouped:
    journal_people = set()

    # Collect all authors and reviewers in this conference group
    for authors_str in group['authors']:
        journal_people.update(ast.literal_eval(authors_str))

    for reviewers_str in group['reviewers']:
        journal_people.update(ast.literal_eval(reviewers_str))

    # Choose a chair from authors not involved in this conference
    eligible_editors = list(unique_authors - journal_people)

    journal_editor[journal] = (
        np.random.choice(eligible_editors) if eligible_editors else None
    )

global_review_id_counter, proceeding_id_counter = 0, 1
proceeding_ids = {}

print(f"Initial ABOX size: {len(abox)} triples.")

for index, row in research_paper.iterrows():

    isJournal = pd.notna(row['journal'])
    if isJournal:
        journal_name = row['journal']
        journal_volume_id = urlib_parse(journal_name) + "_" + "volume_" + str(row['journal_volume'])
        journal_volume_label = str(row['journal_volume'])
        publication_year = str(row['journal_year'])
        editor = journal_editor[journal_name]
    else:
        conference_name = row['conference']
        conference_city = row['conference_city']
        publication_year = str(row['conference_year'])
        conference_edition_id = urlib_parse(conference_name) + "_" + "edition_" + str(row['conference_edition'])
        conference_edition_label = str(row['conference_edition'])
        chair = conference_chair[conference_name]
        proceeding_name = f'Proceeding of {conference_name} at edition {conference_edition_label}'
        if not proceeding_name in proceeding_ids:
            proceeding_ids[proceeding_name] = f'PROCEEDING_{proceeding_id_counter}'
            proceeding_id_counter += 1

    
    paper_id = row['id']
    paper_title = row['title']
    abstract = str(row.get('abstract', ''))

    try:
        authors_list = ast.literal_eval(row['authors']) if pd.notna(row['authors']) else []
        references_list = ast.literal_eval(row['references']) if pd.notna(row['references']) else []
        topics_list = ast.literal_eval(row['topic']) if pd.notna(row['topic']) else []
        reviewers_list = ast.literal_eval(row['reviewers']) if pd.notna(row['reviewers']) else []
        reviews_list = ast.literal_eval(row['reviews']) if pd.notna(row['reviews']) else []

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing list data for paper {paper_id}: {e}. Skipping this paper.")
        continue

    paper_owner_name = row['author']
        
        
    current_paper_review_instance_ids = []
    if len(reviewers_list) == len(reviews_list):
        for i in range(len(reviews_list)):
            review_text = reviews_list[i]
            review_instance_id = f'REVIEW_{global_review_id_counter + i + 1}'
            current_paper_review_instance_ids.append(review_instance_id)
    else:
        print(f"Warning: the length of reviewers and reviews are not equal")
        break

    # Create Property instances
    create_property_instance(abox, EX, paper_id, "has_title", paper_title, is_literal=True)
    create_property_instance(abox, EX, paper_id, "has_abstract", abstract, is_literal=True)
    create_property_instance(abox, EX, paper_id, "corresponds_to_author", author_ids[paper_owner_name])
    create_property_instance(abox, EX, author_ids[paper_owner_name], "has_name", paper_owner_name, is_literal=True)

    for topic in topics_list:
        topic_id = topic_ids[str(topic)]
        create_property_instance(abox, EX, paper_id, "has_topic", topic_id)
        create_property_instance(abox, EX, topic_id, "has_name", str(topic), is_literal=True)

    for cited_paper_id in references_list:
        create_property_instance(abox, EX, paper_id, "cites", str(cited_paper_id))
        
    for author in authors_list:
        author_id = author_ids[str(author)]
        create_property_instance(abox, EX, paper_id, "written_by", author_id)
        create_property_instance(abox, EX, author_id, "has_name", str(author), is_literal=True)

    for i in range(len(reviewers_list)):
        review_instance_id = current_paper_review_instance_ids[i]
        reviewer_name = str(reviewers_list[i])
        create_property_instance(abox, EX, paper_id, "has_review", review_instance_id)
        create_property_instance(abox, EX, review_instance_id, "noted_by_reviewer", author_ids[reviewer_name])
        create_property_instance(abox, EX, review_instance_id, "has_content", reviews_list[i], is_literal=True)
        create_property_instance(abox, EX, review_instance_id, "review_assigned_by", author_ids[editor] if isJournal else author_ids[chair])

    global_review_id_counter += len(current_paper_review_instance_ids)

    if isJournal:
        if pd.notna(row['published_paper']) and row['published_paper']:
            create_property_instance(abox, EX, paper_id, "published_in", journal_volume_id)
        create_property_instance(abox, EX, journal_volume_id, "has_name", f"Volume {journal_volume_label}", is_literal=True)
        create_property_instance(abox, EX, journal_volume_id, "has_date", publication_year, is_literal=True)
        create_property_instance(abox, EX, journal_volume_id, "is_volume_of", journal_ids[journal_name])
        create_property_instance(abox, EX, journal_ids[journal_name], "has_name", journal_name, is_literal=True)
        create_property_instance(abox, EX, journal_ids[journal_name], "has_editor", author_ids[editor])
        create_property_instance(abox, EX, author_ids[editor], "has_name", editor, is_literal=True)

    else:
        if pd.notna(row['published_paper']) and row['published_paper']:
            create_property_instance(abox, EX, paper_id, "published_in", conference_edition_id)
        create_property_instance(abox, EX, paper_id, "presented_in_proceedings", proceeding_ids[proceeding_name])
        create_property_instance(abox, EX, proceeding_ids[proceeding_name], "has_name", proceeding_name, is_literal=True)
        create_property_instance(abox, EX, conference_edition_id, "has_proceedings_record", proceeding_ids[proceeding_name])
        create_property_instance(abox, EX, conference_edition_id, "has_name", f"Edition {conference_edition_label}", is_literal=True)
        create_property_instance(abox, EX, conference_edition_id, "has_date", publication_year, is_literal=True)
        create_property_instance(abox, EX, conference_edition_id, "held_in_city", conference_city, is_literal=True)
        create_property_instance(abox, EX, conference_edition_id, "is_edition_of", conference_ids[conference_name])
        create_property_instance(abox, EX, conference_ids[conference_name], "has_name", conference_name, is_literal=True)
        create_property_instance(abox, EX, conference_ids[conference_name], "has_chair", author_ids[chair])
        create_property_instance(abox, EX, author_ids[chair], "has_name", chair, is_literal=True)


print(f"Finished ABOX generation. Total triples in abox: {len(abox)}")

# Use current working directory instead of script path
script_dir = os.getcwd()  # Works in Jupyter
rdf_file_path = os.path.join(script_dir, "12_B.2_SpissoYao_abox.ttl")

# Save the graph
abox.serialize(destination=rdf_file_path, format="turtle")

print(f"RDF saved to: {rdf_file_path}")


