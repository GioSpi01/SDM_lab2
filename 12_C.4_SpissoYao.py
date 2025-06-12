import torch
import json
import pandas as pd
from pykeen.pipeline import pipeline_from_config
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np

# Load best pipeline configuration
with open('./rotate_hpo_result/best_pipeline/pipeline_config.json') as f:
    config = json.load(f)

# Load triples
file_path = './C.1_query.tsv'
tf = TriplesFactory.from_path(file_path, delimiter="\t")

# Split triples for training
training, testing = tf.split([0.85, 0.15], random_state=2025)
training, validation = training.split([0.8, 0.2], random_state=2025)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Inject splits into config
config['pipeline']['training'] = training
config['pipeline']['testing'] = testing
config['pipeline']['validation'] = validation   
config['pipeline']['device'] = device
config['pipeline']['random_seed'] = 2025

# Train model
pipeline_result = pipeline_from_config(config)
model = pipeline_result.model

# Extract entity IDs
entity_to_id = pipeline_result.training.entity_to_id
id_to_entity = {v: k for k, v in entity_to_id.items()}

# Identify papers using UUID pattern
uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

paper_entities = [
    (idx, uri) for idx, uri in id_to_entity.items()
    if uuid_pattern.match(uri)
]

# Create dataframe of embeddings
data = []
for idx, uri in paper_entities:
    embedding_complex = model.entity_representations[0](
        indices=torch.LongTensor([idx])
    ).detach().cpu().numpy().flatten()

    # Split into real and imaginary parts
    real_part = np.real(embedding_complex)
    imag_part = np.imag(embedding_complex)
    embedding_real = np.concatenate([real_part, imag_part])

    data.append([uri] + embedding_real.tolist())

embedding_df = pd.DataFrame(data)
embedding_df.rename(columns={0: 'paper_id'}, inplace=True)

# Load research papers file
papers_df = pd.read_csv('./research_papers.csv')

# Build the target label: journal or conference name
def extract_label(row):
    if pd.notna(row['journal']):
        return row['journal']
    elif pd.notna(row['conference']):
        return row['conference']
    else:
        return None

papers_df['publication_name'] = papers_df.apply(extract_label, axis=1)

# Merge embeddings with labels
merged_df = pd.merge(embedding_df, papers_df, left_on='paper_id', right_on='id')

# Keep only rows with a valid label
merged_df = merged_df.dropna(subset=['publication_name'])

# Prepare data for training
X = merged_df.drop(columns=['paper_id', 'id', 'publication_name', 'abstract', 'authors', 'n_citation',
                             'references', 'title', 'year', 'publication', 'author', 'topic',
                             'journal', 'conference', 'reviewers', 'reviews', 'conference_year',
                             'conference_city', 'journal_year', 'journal_volume', 'maxreview',
                             'affiliation', 'reviewers_suggested_decision', 'published_paper',
                             'conference_edition']).values

y = merged_df['publication_name'].values



# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(128, 64,), random_state=42, learning_rate='adaptive')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


