import argparse
from tqdm import tqdm

import pandas as pd
import scipy as sp

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns


def load_interactome(path):
    '''
    Loads an interactome .tsv file generated with grexome-TIMC-interactome,
    returns a networkx interactome graph.
    '''
    interactome_df = pd.read_csv(path, sep='\t', header=None)
    G = nx.from_pandas_edgelist(interactome_df, 0, 1, edge_attr=True)
    return G


def load_causal_genes(path):
    '''
    Loads a list (.p pickle file) of genes causal for the given phenotype,
    returns a list of causal genes.
    '''
    causal_genes = pd.read_pickle(path)
    causal_genes_list = list(set([c for c in causal_genes if c in G.nodes()]))

    return causal_genes_list

def get_other_genes(G, causal_genes):
    '''
    Loads the networkx interactome graphs,
    returns a list of all non-causal genes.
    '''
    other_genes = [n for n in G.nodes() if n not in causal_genes] 
    return other_genes

def calculate_adjacency_matrix_powers(G):
    '''
    Calculates the adjacency matrices to the powers up to longest distance between MMAF and non-MMAF genes for scoring,
    returns a dictionary to store matrices with structure:
    
    {power : A^power,
    ...}
    
    '''
    dict_adjacency = {}

    # get A to the power of 1
    A = nx.adjacency_matrix(G)
    A_sparse = sp.sparse.csc_matrix(A.todense(), shape=A.shape)
    dict_adjacency[1] = A_sparse.todense()

    # get A to the power of 2
    res = A_sparse.dot(A_sparse)
    res_sparse = sp.sparse.csc_matrix(res.astype(bool).todense().astype(int))
    dict_adjacency[2] = res_sparse.todense()

    # get A to the powers of up to 4
    for power in tqdm(range(3, 5)):
        res = res.dot(A_sparse)
        res_sparse = sp.sparse.csc_matrix(res.astype(bool).todense().astype(int))
        dict_adjacency[power] = res_sparse.todense()

    return dict_adjacency

def calculate_scores(G, causal_genes, other_genes, alpha=0.5):
    '''
    Calculates the new centrality score for every non-MMAF gene in interactome based on the proximity to causal genes,
    returns a dictionary with structure:
    
    {gene (non causal) : score,
    ...}

    '''
    dict_scores = {}

    gene_list = list(G.nodes())

    # iterate over non-causal genes
    for node in tqdm(other_genes):
        node_idx = gene_list.index(node)

        score = 0
        
        # iterate over causal genes
        for causal_gene in causal_genes:
            causal_gene_idx = gene_list.index(causal_gene)
            
            try:
                # if there's no path connecting node to candidate then there's no contribution to node score
                d = nx.shortest_path_length(G, node, causal_gene)
                
                if d > 4:
                    continue
                
                else:
                    # calculate score
                    A = dict_adjacency.get(d)
                    score += alpha ** d * A[node_idx, causal_gene_idx]
            
            except:
                continue

        dict_scores[node] = score
    
    # sort results from the highest to the lowest scores
    dict_scores_sorted = dict(sorted(dict_scores.items(), key=lambda v: v[1], reverse=True))

    return dict_scores_sorted

def get_gene_info(G, dict_scores_sorted, dict_distances):
    '''
    Gets more info about each node (degree, candidates at distances).

    As input, takes the dictionary with results from calculate_scores(),
    returns a dataframe where each row corresponds to a non-causal gene.
    '''
    for n, score in dict_scores_sorted.items():
        dict_scores_sorted[n] = [round(score, 7), 
                                G.degree(n), 
                                causal_genes_at_distance(dict_distances, n, 1), 
                                causal_genes_at_distance(dict_distances, n, 2),
                                causal_genes_at_distance(dict_distances, n, 3),
                                causal_genes_at_distance(dict_distances, n, 4)]

    df = pd.DataFrame.from_dict(dict_scores_sorted, 
                                orient='index', 
                                columns=['score', 'degree', 'candidates at d=1', 'candidates at d=2', 'candidates at d=3', 'candidates at d=4'])

    return df

def get_distances(G, causal_genes, other_genes):
    '''
    Gets distances between all MMAF and non-MMAF genes.
    
    As input, takes interactome, list of causal genes and list of non-causal genes,
    returns a dictionary with structure:

    {non-causal gene: {causal gene: distance,
                        causal gene: distance,
                        ...}
    ...}

    '''
    
    dict_distances = {}

    # iterate over non-causal genes
    for source_node in tqdm(other_genes):
        dict_tmp = {}

        # iterate over causal genes
        for target_node in causal_genes:
            try:
                # get distance
                distance = nx.shortest_path_length(G, source_node, target_node)

                dict_tmp[target_node] = distance
            except:
                continue

        dict_distances[source_node] = dict_tmp

    return dict_distances

def causal_genes_at_distance(dict_distances, node, d):
    '''
    Helper function for get_gene_info() to calculate the number of causal genes at a distance d from the given gene.
    '''
    return len([dist for dist in dict_distances.get(node).values() if dist == d])

def score_new_candidates(G, results_df, candidates_list, canonical_genes_df):
    '''
    As input, takes interactome, dataframe with scoring results from get_gene_info(), list of new candidates and dataframe with canonical genes.

    Finds candidates in the dataframe with scoring results from get_gene_info() and retrieves their information,
    returns a dataframe where each row corresponds to a candidate.
    '''
    dict_new_candidates = dict([(gene, canonical_genes_df[canonical_genes_df['GENE'] == gene]['ENSG'].values[0]) for gene in candidates_list])

    # check what percentage of new candidates is in interactome
    percentage_in_interactome = len([nc for nc in dict_new_candidates.values() if nc in G.nodes()]) / len(dict_new_candidates.values())
    print(f"{percentage_in_interactome}% of candidates are in the interactome")

    # merge with canonical_genes_df based on ENSG to get gene names for each candidate
    results_df = results_df.merge(canonical_genes_df, left_index=True, right_on='ENSG')

    # sort results from the highest to the lowest scores 
    results_df.sort_values(by='score', inplace=True, ascending=False)
    results_df.reset_index(inplace=True, drop=True)

    # finds rows in the results dataframe corresponding to candidates
    df_new_candidates = results_df[results_df['ENSG'].isin(dict_new_candidates.values())]

    return df_new_candidates

def plot_results_new_candidates(results_df_new_candidates, phenotype, out_path):
    '''
    Plots the results (a violin plot of scores for all non-causal genes, as well as scores for new candidates),
    saves a .png in the given path.
    '''

    # plot scores for all non-causal genes
    sns.violinplot(data=results_df_new_candidates, y='score')
    plt.title("Scores of new candidates")

    # plot scores for new candidates
    for idx, row in results_df_new_candidates.iterrows():
        gene = row['GENE']
        score = row['score']
        plt.plot(score, 'or')
        plt.text(0, score, s=f"{gene}, {score}")
    
    # save plot to png
    plt.savefig(out_path + f"/scores_{phenotype}_new_candidate_genes.png")

if __name__ == "__main__":

    # set phenotype
    phenotype = "MMAF"

    # set alpha parameter
    alpha = 0.5

    # set output path
    out_path = "/home/kubicaj/calc/newCentrality"

    # load data -- to be modified in the future to improve usability
    path_to_interactome = "/home/kubicaj/calc/data/Interactome_human.tsv"
    path_to_causal_genes = f"/home/kubicaj/calc/data/candidateGenesList_{phenotype}.p"
    path_to_canonical_genes = "/home/kubicaj/calc/data/canonicalGenes.tsv"

    # new candidates
    new_candidates = ['CLHC1', 'PHF20', 'NUSAP1', 'CDC20B', 'FAM221A', 'GALR3', 'LRRC9', 'KIF27', 'ZNF208', 'C6orf118', 'CCDC66', 'CCNA1', 'DDX43', 'FSCB', 'FHAD1', 'LRGUK', 'MYCBPAP', 'MYH7B', 'PCDHB15', 'SAMD15', 'SPACA9', 'SPATA24', 'SPATA6', 'TSSK4', 'TTLL2']

    # load interactome to networkx graph
    G = load_interactome(path_to_interactome)

    # get causal and non-causal genes
    causal_genes = load_causal_genes(path_to_causal_genes)
    other_genes = get_other_genes(G, causal_genes)

    print(f"Interactome size: {len(G.nodes())}, number of {phenotype} genes in interactome: {len(causal_genes)}")

    # calculate adjacency matrices to the powers between MMAF and non-MMAF genes for scoring
    print("Calculating adjacency matrix powers for scoring")
    dict_adjacency = calculate_adjacency_matrix_powers(G)

    # calculate new centrality for every non-MMAF gene
    print("Calculating scores")
    dict_scores_sorted = calculate_scores(G, causal_genes, other_genes, alpha)

    # calculate distances between each causal and non-causal genePOF
    print("Calculating distances between causal and non-causal genes")
    dict_distances = get_distances(G, causal_genes, other_genes)

    # get more info about each non-causal gene (degree, causal genes at distance d) and save in pickle format
    result_df = get_gene_info(G, dict_scores_sorted, dict_distances)
    result_df.to_csv(f"{out_path}/scores_{phenotype}_genes.csv", sep='\t', header=True, index=True)

    # first, load canonical genes to map gene names to ENSG in score_new_candidates()
    canonical_genes_df = pd.read_csv(path_to_canonical_genes, sep='\t')
    # score new candidates
    results_df_new_candidates = score_new_candidates(G, result_df, new_candidates, canonical_genes_df)
    # save results in a .csv file
    results_df_new_candidates.to_csv(f"{out_path}/scores_{phenotype}_new_candidates.csv", sep='\t', header=True, index=False)

    # plot scores of new candidates and save figure
    plot_results_new_candidates(results_df_new_candidates, phenotype, out_path=out_path)