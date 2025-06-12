import torch
import pykeen
import pandas as pd
from pykeen import predict
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
import numpy as np
from pykeen.utils import set_random_seed


file_path = 'C.1_query.tsv'
tf = TriplesFactory.from_path(file_path, delimiter="\t")
training, testing = tf.split([0.85, 0.15], random_state=2025)
training, validation = training.split([0.8, 0.2], random_state=2025)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
print(f"Total triples: {len(tf.mapped_triples)}")
print(f"Training triples: {len(training.mapped_triples)}")
print(f"Validation triples: {len(validation.mapped_triples)}")
print(f"Testing triples: {len(testing.mapped_triples)}")

# Model Hyperparameter Tuning

def run_pipeline(model_name, model_kwargs_range, study_name, random_seed=2025):
    # Set the random seed for reproducibility
    set_random_seed(random_seed)

    # Validate input
    if model_name is None or model_kwargs_range is None:
        raise ValueError("Both model name and model parameter ranges must be provided.")

    # Run the hyperparameter optimization pipeline
    result = hpo_pipeline(
        training=training,
        testing=testing,
        validation=validation,

        # Hyperparameter optimization configuration
        sampler="tpe",
        n_trials=30,

        study_name=study_name,

        # Training parameter ranges
        training_kwargs_ranges=dict(
            num_epochs=dict(type="int", low=50, high=150, step=50)
        ),

        # Model and its parameter search space
        model=model_name,
        model_kwargs=dict(
            random_seed=random_seed
        ),
        model_kwargs_ranges=model_kwargs_range,

        # Optimizer configuration
        optimizer="adam",
        optimizer_kwargs_ranges=dict(
            lr=dict(type="float", low=0.0001, high=0.001, log=True),
            weight_decay=dict(type="float", low=1e-4, high=1e-3, log=True)
        ),

        # Negative sampling configuration
        negative_sampler="basic",
        negative_sampler_kwargs_ranges=dict(
            num_negs_per_pos=dict(type="int", low=1, high=10, step=3)
        ),

        # Early stopping settings
        stopper="early",
        stopper_kwargs=dict(
            patience=10,
            frequency=5,
            metric="hits@5",
            relative_delta=0.002,
            larger_is_better=True,
        ),

        device=device,
    )

    return result

# TransH Model
model_name = "TransH"
model_kwargs_range =dict(
            embedding_dim=dict(type="int", low=128, high=256, step=64)
        )
# Run the HPO pipeline
resultTransH = run_pipeline(model_name, model_kwargs_range, 'TransH_HPO_Experiment')
resultTransH.save_to_directory('transh_hpo_result')

# RotatE Model
model_name = "RotatE"
model_kwargs_range =dict(
            embedding_dim=dict(type="int", low=128, high=256, step=64),
        )
# Run the HPO pipeline
resultRotatE = run_pipeline(model_name, model_kwargs_range, 'RotatE_HPO_Experiment')
resultRotatE.save_to_directory('rotate_hpo_result')

# ComplEx Model
model_name = "ComplEx"
model_kwargs_range =dict(
            embedding_dim=dict(type="int", low=128, high=256, step=64),
        )
# Run the HPO pipeline
resultComplEx = run_pipeline(model_name, model_kwargs_range, 'ComplEx_HPO_Experiment')
resultComplEx.save_to_directory('complex_hpo_result')

# ConvKB Model
model_name = "ConvKB"
model_kwargs_range =dict(
            embedding_dim=dict(type="int", low=128, high=256, step=64),
            hidden_dropout_rate=dict(type="float", low=0.3, high=0.5, step=0.1),
            num_filters=dict(type="int", low=64, high=128, step=32),
        )
# Run the HPO pipeline
resultConvKB = run_pipeline(model_name, model_kwargs_range, 'ConvKB_HPO_Experiment')
resultConvKB.save_to_directory('convkb_hpo_result')


