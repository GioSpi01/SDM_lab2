import torch
import pykeen
import pandas as pd
from pykeen import predict
from pykeen.pipeline import pipeline_from_config
from pykeen.triples import TriplesFactory
import numpy as np
from pykeen.utils import set_random_seed
from matplotlib import pyplot as plt
import json


file_path = 'C.1_query.tsv'
tf = TriplesFactory.from_path(file_path, delimiter="\t")
training, testing = tf.split([0.85, 0.15], random_state=2025)
training, validation = training.split([0.8, 0.2], random_state=2025)
device = "cuda" if torch.cuda.is_available() else "cpu"

pd.set_option('display.max_columns', None)


def graph_display(x, y, title):
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(x, y, '-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Objective Metric Value')
    plt.title(title)
    plt.grid(True)  # Optional: add grid for readability
    plt.xticks(x)
    plt.show()


## Choose the best hyperparameter configuration for each model
transHResults = pd.read_csv('transh_hpo_result/trials.tsv', sep='\t')
rotatEResults = pd.read_csv('rotate_hpo_result/trials.tsv', sep='\t')
complExResults = pd.read_csv('complex_hpo_result/trials.tsv', sep='\t')
convKBResults = pd.read_csv('convkb_hpo_result/trials.tsv', sep='\t')

transHResults = transHResults[['number', 'value', 'params_loss.margin', 'params_model.embedding_dim', 'params_model.scoring_fct_norm',
                'params_negative_sampler.num_negs_per_pos', 'params_optimizer.lr', 'params_optimizer.weight_decay', 
                'params_regularizer.weight', 'params_training.batch_size', 'params_training.num_epochs', 'state']]
transHResults = transHResults[transHResults.state=='COMPLETE']
transHResults
graph_display(transHResults.number, transHResults.value, 'TransH Models Evaluation')

transHResults.rename(columns=lambda x: x.replace("params_", "") if x.startswith("params_") else x, inplace=True)
transHResults.sort_values('value', ascending=False)

rotatEResults = rotatEResults[['number', 'value', 'params_loss.margin', 'params_model.embedding_dim', 
                               'params_negative_sampler.num_negs_per_pos', 'params_optimizer.lr', 'params_optimizer.weight_decay', 
                               'params_training.batch_size', 'params_training.num_epochs', 'state']]
rotatEResults = rotatEResults[rotatEResults.state=='COMPLETE']
graph_display(rotatEResults.number, rotatEResults.value, 'RotatE Models Evaluation')

rotatEResults.rename(columns=lambda x: x.replace("params_", "") if x.startswith("params_") else x, inplace=True)
rotatEResults.sort_values('value', ascending=False)

complExResults = complExResults[['number', 'value', 'params_model.embedding_dim', 
                               'params_negative_sampler.num_negs_per_pos', 'params_optimizer.lr', 'params_optimizer.weight_decay', 
                               'params_training.batch_size', 'params_training.num_epochs', 'state']]
complExResults = complExResults[complExResults.state=='COMPLETE']
graph_display(complExResults.number, complExResults.value, 'ComplEx Models Evaluation')

complExResults.rename(columns=lambda x: x.replace("params_", "") if x.startswith("params_") else x, inplace=True)
complExResults.sort_values('value', ascending=False)

convKBResults = convKBResults[['number', 'value', 'params_loss.margin', 'params_model.embedding_dim', 'params_model.hidden_dropout_rate',
                               'params_model.num_filters', 'params_negative_sampler.num_negs_per_pos', 'params_optimizer.lr',
                               'params_optimizer.weight_decay', 'params_regularizer.weight', 'params_training.batch_size', 
                               'params_training.num_epochs', 'state']]
convKBResults = convKBResults[convKBResults.state=='COMPLETE']
graph_display(convKBResults.number, convKBResults.value, 'ConvKB Models Evaluation')

convKBResults.rename(columns=lambda x: x.replace("params_", "") if x.startswith("params_") else x, inplace=True)
convKBResults.sort_values('value', ascending=False)

## Evaluate the best model
def get_best_hyperparameter(best_params_path):
    with open(best_params_path) as f:
        config = json.load(f)
    config['pipeline']['training'] = training
    config['pipeline']['testing'] = testing
    config['pipeline']['validation'] = validation   
    config['pipeline']['device'] = device
    config['pipeline']['random_seed'] = 2025

    return pipeline_from_config(config=config)


def get_results(pipeline_result, model_name, metric = ['inverse_harmonic_mean_rank', 'variance', 'arithmetic_mean_rank', 'hits_at_5']):
    results = pipeline_result.metric_results.to_df()
    results = results[(results.Side == 'both') & (results.Rank_type == 'realistic') & (results.Metric.isin(metric))]

    results = results[['Metric', 'Value']].set_index('Metric')
    results.index.name = 'Model'
    results = results.T
    results.rename(index={'Value': model_name}, inplace=True)
    return results

best_rotate = get_best_hyperparameter('rotate_hpo_result/best_pipeline/pipeline_config.json')

best_rotate_results = get_results(best_rotate, 'RotatE')
print(best_rotate_results)


