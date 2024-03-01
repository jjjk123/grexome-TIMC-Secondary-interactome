import logging
import sys
import os

import argparse
import pathlib

import tqdm

import pandas
import scipy
import numpy

import networkx

import matplotlib.pyplot
import seaborn

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


def parse_interactome(interactome_file) -> tuple[networkx.Graph, dict]:
    '''
    Creates a networkx.Graph interactome.

    arguments:
    - interactome_file: path to interactome SIF file, type=pathlib.Path
      with 3 columns: gene1 pp gene2
    
    returns:
    - interactome: type=networkx.Graph
    - genes: dict with key=gene value=0
    '''
    interactome = networkx.Graph()
    genes = {}

    try:
        f = open(interactome_file, 'r')
    except Exception as e:
        logger.error("Opening provided SIF interactome file %s: %s", interactome_file, e)
        raise Exception("cannot open provided interactome file")

    for line in f:
        line_splitted = line.rstrip().split('\t')
        if len(line_splitted) != 3:
            logger.error("SIF file %s has bad line (not 3 tab-separated fields): %s", interactome_file, line)
            raise Exception("Bad line in the interactome file")

        gene1, pp, gene2 = line_splitted

        # exclude self-interactions
        if gene1 == gene2:
            continue
        # else: ppopulate structures
        interactome.add_edge(gene1, gene2)
        genes[gene1] = 0
        genes[gene2] = 0

    return (interactome, genes)


def parse_causal_genes(causal_genes_file, canonical_genes_file, genes) -> dict:
    '''
    Creates a dictionary of all genes in the interactome, key=gene, value=1 if causal, 0 otherwise.

    arguments:
    - causal_genes_file: path to known causal genes CSV file, type=pathlib.Path
      with 2 columns: gene, pathology
    - canonical_genes_file: path to canonical genes, type=pathlib.Path
      with 2 columns: gene_name, ENSG
    - pathology: phenotype for which to get the causal genes, type=str
    - genes: dict with key=gene, value=0
    
    returns:
    - causal_genes: dict with key=gene, value=1 if causal, 0 otherwise
    '''
    causal_genes = genes.copy()
    canonical_genes = {}

    # first, parse canonical genes
    try:
        f_canonical = open(canonical_genes_file, 'r')
    except Exception as e:
        logger.error("Opening provided canonical genes file %s: %s", canonical_genes_file, e)
        raise Exception("cannot open provided canonical genes file")

    # skip header
    next(f_canonical)

    for line in f_canonical:
        line_splitted = line.rstrip().split('\t')

        gene_name, ENSG = line_splitted
        
        # populate canonical genes dictionary
        canonical_genes[gene_name] = ENSG

    f_canonical.close()

    # second, parse causal genes
    try:
        f_causal = open(causal_genes_file, 'r')
    except Exception as e:
        logger.error("Opening provided causal genes file %s: %s", causal_genes_file, e)
        raise Exception("cannot open provided causal genes file")

    for line in f_causal:
        line_splitted = line.rstrip().split('\t')

        gene_name, pathology = line_splitted

        # map gene names to ENSGs
        if gene_name in canonical_genes.keys():
            ENSG = canonical_genes.get(gene_name)
        else:
            continue
        
        # populate structures
        # NOTE: hardcoded pathology - to remove
        if pathology == "MMAF":
            causal_genes[ENSG] = 1

    f_causal.close()

    return causal_genes


def calculate_scores(interactome, causal_genes, alpha=0.5):
    '''
    Calculates scores for every gene in the interactome based on the proximity to causal genes.

    arguments:
    - interactome: type=networkx.Graph
    - causal_genes: dict with key=gene, value=1 if causal, 0 otherwise

    returns:
    - scores: dict with key=gene, value=score
    '''
    # 1D numpy array for genes in the interactome: 1 if causal gene, 0 otherwise
    causal_genes_array = numpy.array([1 if causal_genes.get(n) == 1 else 0 for n in interactome.nodes()])

    scores_array = numpy.zeros((len(causal_genes_array)))
    norm_factors_array = numpy.zeros((len(causal_genes_array)))

    # initiate dict key=power, value=adjacency_matrix**power
    adjacency_matrices = {}

    A = networkx.to_scipy_sparse_array(interactome) # returns scipy.sparse._csr.csr_array
    A.setdiag(0)
    adjacency_matrices[1] = A

    # @ - matrix multiplication
    res = A @ A
    res.setdiag(0)
    adjacency_matrices[2] = res

    for power in range(2, 4):
        res = res @ A
        res.setdiag(0)
        adjacency_matrices[power] = res

    # calculate normalized scores
    for d in range(1, 3):
        A = adjacency_matrices.get(d)

        # numpy.dot is not aware of sparse arrays, todense() should be used
        scores_array += alpha ** d * numpy.dot(A.todense(), causal_genes_array)

        norm_factors_array += alpha ** d * A.sum(axis=0)

    scores_array_normalized = numpy.squeeze(scores_array / norm_factors_array)

    scores = dict(zip(interactome.nodes(), scores_array_normalized))

    return scores


# def get_gene_info(G, dict_scores_sorted, canonical_genes_df):
#     '''
#     Gets more info about each node (degree, candidates at distances).

#     As input, takes the dictionary with results from calculate_scores(),
#     returns a dataframe where each row corresponds to a non-causal gene.
#     '''

#     dict_distances = get_distances(G, causal_genes, nonCausal_genes)

#     for n, score in dict_scores_sorted.items():
#         dict_scores_sorted[n] = [score, 
#                                 G.degree(n), 
#                                 causal_genes_at_distance(dict_distances, n, 1), 
#                                 causal_genes_at_distance(dict_distances, n, 2),
#                                 causal_genes_at_distance(dict_distances, n, 3),
#                                 causal_genes_at_distance(dict_distances, n, 4)
#                                 ]

#     df = pandas.DataFrame.from_dict(dict_scores_sorted, 
#                                 orient='index', 
#                                 columns=['score', 'degree', 'candidates at d=1', 'candidates at d=2', 'candidates at d=3', 'candidates at d=4'])

#     # merge with canonical_genes_df based on ENSG to get gene names for each candidate
#     results_df = canonical_genes_df.merge(df, right_index=True, left_on='ENSG')

#     # sort results from the highest to the lowest scores 
#     results_df.sort_values(by='score', inplace=True, ascending=False)
#     results_df.reset_index(inplace=True, drop=True)
    
#     return results_df

# def score_new_candidates(G, results_df, candidates_list):
#     '''
#     As input, takes interactome, dataframe with scoring results from get_gene_info(), list of new candidates and dataframe with canonical genes.

#     Finds candidates in the dataframe with scoring results from get_gene_info() and retrieves their information,
#     returns a dataframe where each row corresponds to a candidate.
#     '''
    
#     # finds rows in the results dataframe corresponding to candidates
#     df_new_candidates = results_df[results_df['GENE'].isin(candidates_list)]

#     # check what percentage of new candidates is in interactome
#     percentage_in_interactome = len(df_new_candidates.index) / len(candidates_list)
#     print(f"{percentage_in_interactome}% of candidates in the interactome")

#     return df_new_candidates

# def plot_results_new_candidates(results_df_new_candidates, phenotype, out_path):
#     '''
#     Plots the results (a violin plot of scores for all non-causal genes, as well as scores for new candidates),
#     saves a .png in the given path.
#     '''

#     # plot scores for all non-causal genes
#     seaborn.violinplot(data=results_df_new_candidates, y='score')
#     matplotlib.pyplot.title("Scores of new candidates")

#     # plot scores for new candidates
#     for idx, row in results_df_new_candidates.iterrows():
#         gene = row['GENE']
#         score = row['score']
#         matplotlib.pyplot.plot(score, 'or')
#         matplotlib.pyplot.text(0, score, s=f"{gene}, {score}")
    
#     # save plot to png
#     file_name = f"scores_{phenotype}_new_candidate_genes.png"
#     matplotlib.pyplot.savefig(pathlib.Path(out_path, file_name))


# def get_distances(G, causal_genes, nonCausal_genes):
#     '''
#     Helper function for get_gene_info() to get distances between all causal and non-causal genes.
    
#     As input, takes interactome, list of causal genes and list of non-causal genes,
#     returns a dictionary with structure:

#     {non-causal gene: {causal gene: distance,
#                         causal gene: distance,
#                         ...}
#     ...}

#     '''
    
#     dict_distances = {}

#     print("Calculating distances between causal and non-causal genes")

#     # iterate over non-causal genes
#     for source_node in tqdm.tqdm(nonCausal_genes):
#         dict_tmp = {}

#         # iterate over causal genes
#         for target_node in causal_genes:
#             try:
#                 # get distance
#                 distance = networkx.shortest_path_length(G, source_node, target_node)

#                 dict_tmp[target_node] = distance
#             except:
#                 continue

#         dict_distances[source_node] = dict_tmp

#     return dict_distances

# def causal_genes_at_distance(dict_distances, node, d):
#     '''
#     Helper function for get_gene_info() to calculate the number of causal genes at a distance d from the given gene.
#     '''
#     try:
#         return len([dist for dist in dict_distances.get(node).values() if dist == d])
#     except:
#         return 0

def main(interactome_file, causal_genes_file, canonical_genes_file):

    interactome, genes = parse_interactome(interactome_file)

    causal_genes = parse_causal_genes(causal_genes_file, canonical_genes_file, genes)

    scores = calculate_scores(interactome, causal_genes)

    # print(scores)

if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(script_name)

    parser = argparse.ArgumentParser(
        prog="newCentrality.py",
        description="Calculate new centrality for new candidates of infertility based on the guilt-by-association approach."
    )

    parser.add_argument('-i', '--interactome_file', type=pathlib.Path)
    parser.add_argument('--causal_genes_file', type=pathlib.Path)
    parser.add_argument('--canonical_genes_file', type=pathlib.Path)
    # parser.add_argument('--new_candidates', type=str, nargs='+')
    # parser.add_argument('--phenotype', type=str)
    # parser.add_argument('-o', '--output_path', type=pathlib.Path)

    args = parser.parse_args()

    try:
        main(interactome_file=args.interactome_file,
             causal_genes_file=args.causal_genes_file,
             canonical_genes_file=args.canonical_genes_file)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)