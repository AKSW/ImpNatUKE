import pandas as pd
import networkx as nx
import numpy as np
import logging

from bertopic import BERTopic
from tqdm import tqdm
from copy import deepcopy
from sentence_transformers import SentenceTransformer, LoggingHandler

path = 'path-to-data-repository'

df_file = pd.read_parquet('{}file-name_query03-05.parquet'.format(path))
del df_file['Unnamed: 0']

df_topics = pd.read_parquet('{}topics03-05.parquet'.format(path))
del df_topics['text']
del df_topics['phrases']
df_topics['txt_file_name'] = df_topics['file_name'].str.replace('.pdf', '.txt')
del df_topics['file_name']

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

def model_encode(model):
    embeddings = []
    for _, file_name in tqdm(df_topics['txt_file_name'].iteritems()):
        embedding = np.zeros(512)
        try:
            with open(f'{path}processed_txts/{file_name}', 'r') as f:
                data = f.read()
            embedding += model.encode(data, show_progress_bar=False)
        except:
            embeddings.append(embedding)
            continue
        embeddings.append(embedding)
    return embeddings

df_topics['embedding'] = model_encode(model)
df_topics

df_file = df_file.join(df_topics.set_index('doi'), on='doi')

df_file = df_file.fillna('?')
df_topics = df_topics.fillna('?')

def make_networkx(df, texts_df, topic_model,
                    id_feature='doi', special_attributes={'name': 'smile'}, list_features=[
                        'bioActivity', 'molType', 'collectionSpecie', 'collectionSite', 'collectionType', 'molecularMass', 'monoisotropicMass', 'cLogP', 'tpsa', 
                        'numberOfLipinskiViolations', 'numberOfH_bondAcceptors', 'numberOfH_bondDonors', 'numberOfRotableBonds', 'molecularVolume', 'name'
                    ],
):
    def create_edge(value1, value2, group1, group2, node_from):            
        if value1 != '?' and value2 != '?':
            G.add_edge(value1, value2, edge_group=f'{group1}_{group2}')
            G.nodes[value1]['group'] = group1
            G.nodes[value2]['group'] = group2
            G.nodes[value2]['node_from'] = node_from
    
    def create_attribute(attribute_feature, feature_value, attribute_value):
        if attribute_value != '?':
            try:
                G.nodes[feature_value][attribute_feature] = attribute_value
            except:
                print('attribute_feature: {}, feature_value: {}, attribute_value: {}'. format(attribute_feature, feature_value, attribute_value))

    G = nx.Graph()
    for _, row in tqdm(df.iterrows()):
        for feature in list_features:
            create_edge(row[id_feature], row[feature], id_feature, feature, 'nubbe')
            if feature in special_attributes:
                create_attribute(special_attributes[feature], row[feature], row[special_attributes[feature]])
    for _, row in texts_df.iterrows():
        for topic in row['topics']:
            create_edge(row[id_feature], topic_model.get_topic_info(topic)['Name'].iloc[0], id_feature, 'topic', 'pdf')
        try:
            if row['embedding'] != '?':
                G.nodes[row[id_feature]]['embedding'] = row['embedding']
            else:
                G.nodes[row[id_feature]]['embedding'] = np.array([0.0]*row['embedding'].shape[0])
                print(f"doi {row[id_feature]} is NaN")
        except:
            print('doi {} has no connections'.format(row[id_feature]))

    return G

topic_model = BERTopic.load('{}topic_model03-05'.format(path))

G = make_networkx(df_file, df_topics, topic_model)

nx.write_gpickle(G, f"{path}hin_grobid_bert-18-09.gpickle")

## splits

def disturbed_hin(G, split=0.6, random_state=None, extra_cut_from='nubbe', edge_group='doi_bioActivity', node_from_feature='node_from', type_feature='edge_group', group_feature='group'):
    """
    G: hin;
    split: percentage to be cut from the hin;
    random_state: ;
    extra_cut_from: edges from the origin that needs to be cut but not restored;
    edge_group: string of type of edge to be added for restoration;
    type_feature: feature name of edge_type on your hin.
    """
    def keep_left(x, G):
        edge_split = x['type'].split('_')
        if G.nodes[x['node']][group_feature] != edge_split[0]:
            x['node'], x['neighbor'] = x['neighbor'], x['node']
        return x
    # prepare data for type counting
    edges = list(G.edges)
    edge_types = [G[edge[0]][edge[1]][type_feature] for edge in edges]
    
    edges = pd.DataFrame(edges)
    edges = edges.rename(columns={0: 'node', 1: 'neighbor'})
    edges['type'] = edge_types
    edges = edges.apply(keep_left, G=G, axis=1)
    edges_group = edges.groupby(by=['type'], as_index=False).count().reset_index(drop=True)

    # preparar arestas para eliminar
    edges = edges.sample(frac=1, random_state=random_state).reset_index(drop=True)
    edges_group = edges_group.rename(columns={'node': 'count', 'neighbor': 'to_cut_count'})
    edges_group['to_cut_count'] = edges_group['to_cut_count'].apply(lambda x:round(x * split))
    train, test = {}, {}
    for _, row in edges_group.iterrows():
        if row['type'] == edge_group:
            train[row['type']] = edges[edges['type'] == row['type']].reset_index(drop=True).loc[row['to_cut_count']:].reset_index(drop=True)
            test[row['type']] = edges[edges['type'] == row['type']].reset_index(drop=True).loc[:row['to_cut_count']-1].reset_index(drop=True)
                    
    G_disturbed = deepcopy(G)
    hidden = {'node': [], 'neighbor_group': []}
    for tc_df in test.values():
        for _, row in tc_df.iterrows():
            neighbors_list = list(G_disturbed.neighbors(row['node']))
            neighbors_hidden = []
            has_cut = False
            for neighbor in neighbors_list:
                if G_disturbed.nodes[neighbor][node_from_feature] == extra_cut_from:
                    has_cut = True
                    neighbors_hidden.append({'neighbor': neighbor, 'edge_group': G_disturbed[row['node']][neighbor][type_feature]})
                    G_disturbed.remove_edge(row['node'],neighbor)
            if has_cut:
                hidden['node'].append(row['node'])
                hidden['neighbor_group'].append(neighbors_hidden)
    return G_disturbed, train, test, pd.DataFrame(hidden)

def true_restore(G, hidden, train, test, percentual=1.0, edge_group='doi_bioActivity', node_feature='node', neighbor_group_feature='neighbor_group', neighbor_feature='neighbor', edge_group_feature='edge_group'):
    G_found = deepcopy(G)
    adding_df = hidden.loc[0:round(hidden.shape[0] * percentual)-1]
    remaining_df = hidden.loc[round(hidden.shape[0] * percentual):hidden.shape[0]-1]
    df_train, df_test = train[edge_group], test[edge_group]
    for _, row in adding_df.iterrows():
        df_train = pd.concat([df_train, df_test[df_test[node_feature] == row[node_feature]]])
        df_test = df_test.drop(df_test[df_test[node_feature] == row[node_feature]].index)
        for to_add in row[neighbor_group_feature]:
            G_found.add_edge(row[node_feature], to_add[neighbor_feature], edge_type=to_add[edge_group_feature])
    
    train[edge_group], test[edge_group] = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    return G_found, remaining_df.reset_index(drop=True), train, test

def execution(G, iteration, edge_group, percentual_to_time):
    G_disturbed, train, test, hidden = disturbed_hin(G, split=0.8, random_state=(1 + iteration), edge_group=edge_group)
    G_found, hidden, train, test = true_restore(G_disturbed, hidden, train, test, percentual=0.0, edge_group=edge_group)

    for key, value in percentual_to_time.items():
        train[edge_group].to_csv(f"{path}grobid_bert_splits/train_{edge_group}_{iteration}_{key}.csv", index=False)
        test[edge_group].to_csv(f"{path}grobid_bert_splits/test_{edge_group}_{iteration}_{key}.csv", index=False)
        nx.write_gpickle(G_found, f"{path}grobid_bert_splits/kg_{edge_group}_{iteration}_{key}.gpickle")
        G_found, hidden, train, test = true_restore(G_found, hidden, train, test, percentual=value, edge_group=edge_group)

edge_groups = ['doi_name', 'doi_bioActivity', 'doi_collectionSpecie', 'doi_collectionSite', 'doi_collectionType']
percentual_to_time = {'1st': 0.3, '2nd': 0.32, '3rd': 0.5, '4th': 0.0}

for edge_group in tqdm(edge_groups):
    for iteration in range(10):
        execution(G, iteration, edge_group, percentual_to_time)