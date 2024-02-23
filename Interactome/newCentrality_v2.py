import argparse
import pathlib

import tqdm

import pandas
import scipy
import numpy

import networkx

import matplotlib.pyplot
import seaborn


def load_interactome(path):
    '''
    Loads an interactome .tsv file generated with grexome-TIMC-interactome,
    returns a networkx interactome graph.
    '''
    interactome_df = pandas.read_csv(path, sep='\t', header=None)
    G = networkx.from_pandas_edgelist(interactome_df, 0, 1, edge_attr=True)
    return G


def load_causal_genes(path):
    '''
    Loads a list (.p pickle file) of genes causal for the given phenotype,
    returns a list of causal genes.
    '''
    causal_genes = pandas.read_pickle(path)
    causal_genes_list = list(set([c for c in causal_genes if c in G.nodes()]))

    return causal_genes_list

def get_nonCausal_genes(G, causal_genes):
    '''
    Loads the networkx interactome graph,
    returns a list of all non-causal genes.
    '''
    nonCausal_genes = [n for n in G.nodes() if n not in causal_genes] 
    return nonCausal_genes

def calculate_adjacency_matrix_powers(G):
    '''
    Calculates the adjacency matrices to the powers up to longest distance between causal and non-causal genes for scoring,
    returns a dictionary to store matrices with structure:
    
    {power : A^power,
    ...}
    
    '''
    dict_adjacency = {}

    # get A to the power of 1
    A = networkx.adjacency_matrix(G)
    A_sparse = scipy.sparse.csc_matrix(A, dtype=bool)
    dict_adjacency[1] = A_sparse

    # get A to the power of 2
    res = A_sparse.dot(A_sparse)
    res_sparse = scipy.sparse.csc_matrix(res, dtype=bool)
    dict_adjacency[2] = res_sparse

    # get A to the powers of up to 4
    for power in tqdm.tqdm(range(3, 5)):
        res = res.dot(A_sparse)
        res_sparse = scipy.sparse.csc_matrix(res, dtype=bool)
        dict_adjacency[power] = res_sparse

    return dict_adjacency

def calculate_scores(G, causal_genes, alpha=0.5):
    '''
    Calculates the new centrality score for every non-causal gene in interactome based on the proximity to causal genes,
    returns a dictionary with structure:
    
    {gene : score,
    ...}

    '''
    # initialize a vector of elements corresponding to each node in interactome being a causal gene (1) or not (0)
    causal_genes_array = numpy.array([1/(alpha * len(causal_genes)) if n in causal_genes else 0 for n in G.nodes()]).reshape(-1, 1)

    scores = numpy.zeros((len(causal_genes_array))).reshape(-1, 1)
    norm_factors = numpy.zeros((len(causal_genes_array))).reshape(-1, 1)

    # iterate over distances
    for d in range(1, 5):
        # get the adjacancy matrix^d
        A = dict_adjacency.get(d)

        # calculate scores for each node
        scores += alpha ** d * A.dot(causal_genes_array)

        # calculate elements of normalization vector
        norm_factors += (1 / (alpha * len(G.nodes())) * alpha ** d) * numpy.sum(A, axis=1)

    # normalize scores
    scores_normalized = numpy.squeeze(scores / norm_factors)

    # create a dictionary sorted by scores with structure:
    # {gene (non-causal) : score,}
    dict_scores = dict(zip(G.nodes(), scores_normalized))
    dict_scores_sorted = dict(sorted(dict_scores.items(), key=lambda v: v[1], reverse=True))
    dict_scores_sorted = {k: v for k, v in dict_scores_sorted.items() if k not in causal_genes}

    return dict_scores_sorted

def get_gene_info(G, dict_scores_sorted, canonical_genes_df):
    '''
    Gets more info about each node (degree, candidates at distances).

    As input, takes the dictionary with results from calculate_scores(),
    returns a dataframe where each row corresponds to a non-causal gene.
    '''

    dict_distances = get_distances(G, causal_genes, nonCausal_genes)

    for n, score in dict_scores_sorted.items():
        dict_scores_sorted[n] = [score, 
                                G.degree(n), 
                                causal_genes_at_distance(dict_distances, n, 1), 
                                causal_genes_at_distance(dict_distances, n, 2),
                                causal_genes_at_distance(dict_distances, n, 3),
                                causal_genes_at_distance(dict_distances, n, 4)
                                ]

    df = pandas.DataFrame.from_dict(dict_scores_sorted, 
                                orient='index', 
                                columns=['score', 'degree', 'candidates at d=1', 'candidates at d=2', 'candidates at d=3', 'candidates at d=4'])

    # merge with canonical_genes_df based on ENSG to get gene names for each candidate
    results_df = canonical_genes_df.merge(df, right_index=True, left_on='ENSG')

    # sort results from the highest to the lowest scores 
    results_df.sort_values(by='score', inplace=True, ascending=False)
    results_df.reset_index(inplace=True, drop=True)
    
    return results_df

def score_new_candidates(G, results_df, candidates_list):
    '''
    As input, takes interactome, dataframe with scoring results from get_gene_info(), list of new candidates and dataframe with canonical genes.

    Finds candidates in the dataframe with scoring results from get_gene_info() and retrieves their information,
    returns a dataframe where each row corresponds to a candidate.
    '''
    
    # finds rows in the results dataframe corresponding to candidates
    df_new_candidates = results_df[results_df['GENE'].isin(candidates_list)]

    # check what percentage of new candidates is in interactome
    percentage_in_interactome = len(df_new_candidates.index) / len(candidates_list)
    print(f"{percentage_in_interactome}% of candidates in the interactome")

    return df_new_candidates

def plot_results_new_candidates(results_df_new_candidates, phenotype, out_path):
    '''
    Plots the results (a violin plot of scores for all non-causal genes, as well as scores for new candidates),
    saves a .png in the given path.
    '''

    # plot scores for all non-causal genes
    seaborn.violinplot(data=results_df_new_candidates, y='score')
    matplotlib.pyplot.title("Scores of new candidates")

    # plot scores for new candidates
    for idx, row in results_df_new_candidates.iterrows():
        gene = row['GENE']
        score = row['score']
        matplotlib.pyplot.plot(score, 'or')
        matplotlib.pyplot.text(0, score, s=f"{gene}, {score}")
    
    # save plot to png
    file_name = f"scores_{phenotype}_new_candidate_genes.png"
    matplotlib.pyplot.savefig(pathlib.Path(out_path, file_name))


def get_distances(G, causal_genes, nonCausal_genes):
    '''
    Helper function for get_gene_info() to get distances between all causal and non-causal genes.
    
    As input, takes interactome, list of causal genes and list of non-causal genes,
    returns a dictionary with structure:

    {non-causal gene: {causal gene: distance,
                        causal gene: distance,
                        ...}
    ...}

    '''
    
    dict_distances = {}

    print("Calculating distances between causal and non-causal genes")

    # iterate over non-causal genes
    for source_node in tqdm.tqdm(nonCausal_genes):
        dict_tmp = {}

        # iterate over causal genes
        for target_node in causal_genes:
            try:
                # get distance
                distance = networkx.shortest_path_length(G, source_node, target_node)

                dict_tmp[target_node] = distance
            except:
                continue

        dict_distances[source_node] = dict_tmp

    return dict_distances

def causal_genes_at_distance(dict_distances, node, d):
    '''
    Helper function for get_gene_info() to calculate the number of causal genes at a distance d from the given gene.
    '''
    try:
        return len([dist for dist in dict_distances.get(node).values() if dist == d])
    except:
        return 0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="newCentrality.py",
        description="Calculate new centrality for new candidates of infertility based on the guilt-by-association approach."
    )

    parser.add_argument('--interactome', type=pathlib.Path)
    parser.add_argument('--causal_genes', type=pathlib.Path)
    parser.add_argument('--canonical_genes', type=pathlib.Path)
    parser.add_argument('--new_candidates', type=str, nargs='+')
    parser.add_argument('--phenotype', type=str)
    parser.add_argument('-o', '--output_path', type=pathlib.Path)

    args = parser.parse_args()

    ## set phenotype
    # phenotype = "MMAF"

    # set phenotype
    phenotype = args.phenotype.upper()

    # set alpha parameter
    alpha = 0.5

    ## set output path
    # out_path = "/home/kubicaj/calc/newCentrality"
    out_path = args.output_path

    ## load data -- to be modified in the future to improve usability
    # path_to_interactome = "/home/kubicaj/calc/data/Interactome_human.tsv"
    # path_to_causal_genes = f"/home/kubicaj/calc/data/candidateGenesList_{phenotype}.p"
    # path_to_canonical_genes = "/home/kubicaj/calc/data/canonicalGenes.tsv"
    
    # load data
    path_to_interactome = args.interactome
    path_to_causal_genes = args.causal_genes
    path_to_canonical_genes = args.canonical_genes

    ## new candidates
    new_candidates = ['CLHC1', 'PHF20', 'NUSAP1', 'CDC20B', 'FAM221A', 'GALR3', 'LRRC9', 'KIF27', 'ZNF208', 'C6orf118', 'CCDC66', 'CCNA1', 'DDX43', 'FSCB', 'FHAD1', 'LRGUK', 'MYCBPAP', 'MYH7B', 'PCDHB15', 'SAMD15', 'SPACA9', 'SPATA24', 'SPATA6', 'TSSK4', 'TTLL2']
    # new_candidates = args.new_candidates

    # load interactome to networkx graph
    G = load_interactome(path_to_interactome)

    # get causal and non-causal genes
    causal_genes = load_causal_genes(path_to_causal_genes)
    nonCausal_genes = get_nonCausal_genes(G, causal_genes)

    print(f"Interactome size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges \n Number of {phenotype} genes in interactome: {len(causal_genes)}")

    # calculate adjacency matrices to the powers between causal and non-causal genes for scoring
    print("Calculating adjacency matrix powers for scoring")
    dict_adjacency = calculate_adjacency_matrix_powers(G)

    # calculate new centrality for every non-causal gene
    print("Calculating scores")
    dict_scores_sorted = calculate_scores(G, causal_genes, alpha)

    # get more info about each non-causal gene (degree, causal genes at distance d) and save in pickle format
    # first, load canonical genes to map gene names to ENSG in score_new_candidates()
    canonical_genes_df = pandas.read_csv(path_to_canonical_genes, sep='\t')

    result_df = get_gene_info(G, dict_scores_sorted, canonical_genes_df)
    file_name = f"scores_{phenotype}_genes.csv"
    result_df.to_csv(pathlib.Path(out_path, file_name), sep='\t', header=True, index=False)

    # score new candidates
    results_df_new_candidates = score_new_candidates(G, result_df, new_candidates)
    # save results in a .csv file
    file_name = f"scores_{phenotype}_new_candidates.csv"
    results_df_new_candidates.to_csv(pathlib.Path(out_path, file_name), sep='\t', header=True, index=False)

    # plot scores of new candidates and save figure
    plot_results_new_candidates(results_df_new_candidates, phenotype, out_path=out_path)