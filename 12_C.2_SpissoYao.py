import torch
import pykeen
import pandas as pd
from pykeen import predict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np

file_path = 'C.2_query.tsv'

tf = TriplesFactory.from_path(file_path, delimiter="\t")
training, testing = tf.split([0.8, 0.2], random_state=2025)

## The Most Basic Model
device = "cuda" if torch.cuda.is_available() else "cpu"

resultTransE = pipeline(
    training=training,
    testing=testing,
    model="TransE",
    model_kwargs=dict(
        embedding_dim=128,
    ),
    training_kwargs=dict(
        num_epochs=50
    ),
    negative_sampler_kwargs=dict(
        num_negs_per_pos=5,
    ),
    random_seed=2025,
    device = device
)


# We choose the paper with id: "4ab60d22-67d1-4683-8648-f1f2601d2ce0"
paper = '4ab60d22-67d1-4683-8648-f1f2601d2ce0'

cites = 'cites'
written_by = 'written_by'

entity_embeddings = resultTransE.model.entity_representations[0](indices= None).detach()
relation_embeddings = resultTransE.model.relation_representations[0](indices=None).detach()

citing_paper_id = resultTransE.training.entity_to_id[paper]
cites_id = resultTransE.training.relation_to_id[cites]

citing_paper_embedding = entity_embeddings[citing_paper_id]
cites_embedding = relation_embeddings[cites_id]


# Compute the estimated embedding of the cited paper
cited_paper_embedding = citing_paper_embedding + cites_embedding

# Compute Euclidean distances to all entities
distances = torch.norm(entity_embeddings - cited_paper_embedding.unsqueeze(0), p=2, dim=1)
sorted_distances, sorted_indices = torch.sort(distances)

# Find the closest entity that is not the citing paper itself
top_cited_paper_id = -1
for index in sorted_indices:
    if index != citing_paper_id:
        top_cited_paper_id = index.item()
        break

# Output the result
if top_cited_paper_id != -1:
    print('A likely cited paper was found.')
    print(f'The cited paper is: {resultTransE.training.entity_id_to_label[top_cited_paper_id]}')
    print(f'Most likely cited paper embedding vector:\n{cited_paper_embedding}\n')
else:
    print('No likely cited paper found.')

written_by_id = resultTransE.training.relation_to_id[written_by]
written_by_embedding = relation_embeddings[written_by_id]

author_embedding = cited_paper_embedding + written_by_embedding

print(f'The most likely author embedding vector: \n {author_embedding}')

distances = torch.norm(entity_embeddings - author_embedding.unsqueeze(0), p=2, dim=1)  # Euclidean Distance
sorted_distances, sorted_indices = torch.sort(distances)

top_author_id = -1
for index in sorted_indices:
    if index != citing_paper_id:
        top_author_id = index.item()
        break
if top_author_id != -1:
    top_author = resultTransE.training.entity_id_to_label[top_author_id]

    print(f'The most likely author is: {top_author}')


