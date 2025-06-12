from rdflib import Graph, Namespace, RDF, RDFS, XSD, Literal
import os

# Create a Graph
EX = Namespace("http://example.org/research_paper/")

# Create a graph
g = Graph()

# Bind common prefixes (for readability)
g.bind("ex", EX)
g.bind("rdfs", RDFS)
g.bind("rdf", RDF)
g.bind("xsd", XSD)

def create_property(g, namespace, property_name, domain, range_):
    g.add((namespace[property_name], RDF.type, RDF.Property))
    g.add((namespace[property_name], RDFS.label, Literal(property_name.replace("_", " "), lang="en")))
    g.add((namespace[property_name], RDFS.domain, namespace[domain]))
    if range_ != "string" and range_ != "date":
        g.add((namespace[property_name], RDFS.range, namespace[range_]))
    else:
        if range_ == "date":
            g.add((namespace[property_name], RDFS.range, XSD.date))
        elif range_ == "string":
            g.add((namespace[property_name], RDFS.range, XSD.string))


def create_class(g, namespace, class_name, label=None):
    g.add((namespace[class_name], RDF.type, RDFS.Class))
    if label:
        g.add((namespace[class_name], RDFS.label, Literal(label, lang="en")))
    else:
        g.add((namespace[class_name], RDFS.label, Literal(class_name.replace("_", " "), lang="en")))

def create_subclass(g, namespace, subclass, superclass):
    g.add((namespace[subclass], RDFS.subClassOf, namespace[superclass]))


print(f"Initial TBOX size: {len(g)} triples.")

create_class(g, EX, "ResearchEntity")

# Core Classes
create_class(g, EX, "Paper")
create_class(g, EX, "Author")
create_class(g, EX, "Topic", label="Research Topic")
create_class(g, EX, "Review")
create_class(g, EX, "Publication", label="Publication Venue")
create_class(g, EX, "Journal")
create_class(g, EX, "Conference")
create_class(g, EX, "Proceeding", label="Conference Proceedings")
create_class(g, EX, "ConferenceEdition", label="Conference Edition")
create_class(g, EX, "JournalVolume", label="Journal Volume")

# Role Classes
create_class(g, EX, "Owner", label="Corresponding Author")
create_class(g, EX, "Reviewer")
create_class(g, EX, "ConferenceChair", label="Conference Chair")
create_class(g, EX, "JournalEditor", label="Journal Editor")

# Author Roles
create_subclass(g, EX, "Owner", "Author")
create_subclass(g, EX, "Reviewer", "Author")
create_subclass(g, EX, "ConferenceChair", "Author")
create_subclass(g, EX, "JournalEditor", "Author")

# Publication Types
create_subclass(g, EX, "ConferenceEdition", "Publication")
create_subclass(g, EX, "JournalVolume", "Publication")
create_subclass(g, EX, "Proceeding", "Publication")

create_subclass(g, EX, "Paper", "ResearchEntity")
create_subclass(g, EX, "Author", "ResearchEntity")
create_subclass(g, EX, "Topic", "ResearchEntity")
create_subclass(g, EX, "Conference", "ResearchEntity")
create_subclass(g, EX, "Journal", "ResearchEntity")
create_subclass(g, EX, "Publication", "ResearchEntity")


# General Properties
create_property(g, EX, "has_name", "ResearchEntity", "string")

# Paper properties
create_property(g, EX, "has_title", "Paper", "string")
create_property(g, EX, "written_by", "Paper", "Author")
create_property(g, EX, "cites", "Paper", "Paper")
create_property(g, EX, "has_abstract", "Paper", "string")
create_property(g, EX, "has_topic", "Paper", "Topic")
create_property(g, EX, "corresponds_to_author", "Paper", "Owner")
create_property(g, EX, "has_review", "Paper", "Review")
create_property(g, EX, "published_in", "Paper", "Publication")
create_property(g, EX, "presented_in_proceedings", "Paper", "Proceeding")

# Review properties
create_property(g, EX, "noted_by_reviewer", "Review", "Reviewer")
create_property(g, EX, "has_content", "Review", "string")
create_property(g, EX, "review_assigned_by", "Review", "Author") # will be Chair or Editor
# create_property(g, EX, "review_assigned_by_chair", "Review", "Chair")
# create_property(g, EX, "review_assigned_by_editor", "Review", "Editor")

# Publication properties
create_property(g, EX, "has_date", "Publication", "string")

# Journal and Volume properties
create_property(g, EX, "is_volume_of", "JournalVolume", "Journal")

# Conference and Edition properties
create_property(g, EX, "has_proceedings_record", "ConferenceEdition", "Proceeding")
create_property(g, EX, "held_in_city", "ConferenceEdition", "string")
create_property(g, EX, "is_edition_of", "ConferenceEdition", "Conference")

# Roles and Assignments
create_property(g, EX, "has_chair", "Conference", "ConferenceChair")
create_property(g, EX, "has_editor", "Journal", "JournalEditor")

print(f"Finished TBOX generation. Total triples in TBOX: {len(g)}.")


# Use current working directory instead of script path
script_dir = os.getcwd()  # Works in Jupyter
rdf_file_path = os.path.join(script_dir, "12_B.1_SpissoYao_tbox.ttl")

# Save the graph
g.serialize(destination=rdf_file_path, format="turtle")

print(f"RDF saved to: {rdf_file_path}")


