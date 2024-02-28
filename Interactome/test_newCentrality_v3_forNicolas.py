import numpy
import scipy

import pandas

import networkx

import matplotlib.pyplot
import seaborn

import tqdm

def causal_genes_at_distance(dict_distances, node, d):
    '''
    Calculates the number of causal genes from a node of interest at the given distance d.

    Arguments:
        - dict_distances: dictionary with structure 
                            {node (non-causal): 
                                {node (causal): distance
                                ...}
                            ...}
        - node: name of the node of interest, type=str
        - d: distance of interest, type=int

    Returns
        - the number of causal genes from the node of interest at the given distance d, type=int
    '''
    try:
        return len([dist for dist in dict_distances.get(node).values() if dist == d])
    except:
        return 0
    
# set phenotype name to choose from ["MMAF", "NOA", "OG"], type=str
PHENOTYPE = "MMAF"

# set alpha parameter, type=float
ALPHA = 0.5

'''
Step 1. Load data
'''
# load an interactome .tsv file generated with grexome-TIMC-interactome to 2D numpy array of size=(number of interactions, 3), dtype=str
# each row in the .tsv file should be: 
#  gene1  gene2   1 
interactome = numpy.genfromtxt("./data/Interactome_human.tsv", delimiter='\t', dtype=str)

# initialize networkx graph object
G = networkx.Graph()

# iterate over rows of the interactome numpy array
for row in interactome:
    
    # unpack row values (gene1, gene2, 1) to variables
    gene1, gene2, interaction = row

    # add edges (interactions) to the networkx graph object based on each row (interaction) in interactome numpy array
    G.add_edge(gene1, gene2)


# load a causal list .p (pickle) file generated with causalGenes.ipynb to 1D numpy array of size=(number of causal genes), dtype=str
causal_genes = numpy.load(f"./data/causalGenesList_{PHENOTYPE}.p", allow_pickle=True)

# create a list of unique causal gene names from the causal gene numpy array if present in the interactome, dtype=str
causal_genes = list(set([c for c in causal_genes if c in G.nodes()]))

# create a list of size=(number of non-causal genes) to store non-causal gene names, dtype=str
nonCausal_genes = [n for n in G.nodes() if n not in causal_genes] 

# print interactome information (number of nodes, number of edges, number of causal genes)
print(f"Interactome size: {G.number_of_nodes()} nodes, {G.number_of_edges()}, number of {PHENOTYPE} candidates in interactome: {len(causal_genes)}")

'''
Step 2. Calculate distances between causal and non-causal genes
'''
# initiate a dictionary to store distances between all causal and non-causal gene
dict_distances = {}

# iterate over non-causal genes
# note: tqdm.tqdm prints a progress bar of loop iteration
for source_node in tqdm.tqdm(nonCausal_genes):

    # initiate a dictionary to store distances between all causal genes and the current non-causal gene
    dict_tmp = {}

    # iterate over causal genes
    for target_node in causal_genes:
        # calculate the distance (shortest path) between the current causal and non-causal gene
        try:
            distance = networkx.shortest_path_length(G, source_node, target_node)
            dict_tmp[target_node] = distance
        # if path doesn't exist, skip the iteration
        except:
            continue
    
    # save the distances between all causal genes to the current non-causal gene in the dictionary
    dict_distances[source_node] = dict_tmp

'''
Step 3. Calculate adjacency matrices to the powers up to 4
'''
# initiate a dictionary to store adjacency matrices
# dictionary structure:
#   {power : A^power,
#   ...}
dict_adjacency = {}

# get adjacency matrix A with shape=(number of nodes, number of nodes), dtype=int64
A = networkx.adjacency_matrix(G)
# create a sparse matrix out of A, dtype=int64 (default int of scipy.sparse.csr_matrix)
A_sparse = scipy.sparse.csr_matrix(A, dtype=int)
# set diagonal of A to zeros
A_sparse.setdiag(0)
# save sparse A to the dictionary
dict_adjacency[1] = A_sparse

# calculate A to the power of 2 with shape=(number of nodes, number of nodes), dtype=int64
# use scipy dot product function to multiply scipy sparse A and scipy sparse A
res = A_sparse.dot(A_sparse)
# set diagonal of A^2 to zeros
res.setdiag(0)
# create a sparse matrix out of A^2, dtype=int64
res_sparse = scipy.sparse.csr_matrix(res, dtype=int)
# save sparse A^2 to the dictionary
dict_adjacency[2] = res_sparse
 
# calculate A to the powers of 3 to 4 with shape=(number of nodes, number of nodes), dtype=int64
# tqdm.tqdm prints a progress bar of loop iteration on the screen
for power in tqdm.tqdm(range(3, 5)):
    # use scipy dot product function to multiply scipy sparse A^(power-1) and scipy sparse A to get sparse A^power
    res = res.dot(A_sparse)
    # set diagonal of A^power to zeros
    res_sparse.setdiag(0)
    # create a sparse matrix out of A^power, dtype=int64
    res_sparse = scipy.sparse.csr_matrix(res, dtype=int)
    # save sparse A^power to the dictionary
    dict_adjacency[power] = res_sparse

'''
Step 4. Calculate scores
'''
# initialize a 1D array with size=len(number of nodes in interactome): 0 if non-causal and 1 for causal
causal_genes_array = numpy.array([1 if n in causal_genes else 0 for n in G.nodes()])

# initialize a 1D array for storing scores with size=len(number of nodes in interactome): all elements=0
scores = numpy.zeros((len(causal_genes_array)))
# initialize a 1D array for normalization factors with size=len(number of nodes in interactome): all elements=0
norm_factors = numpy.zeros(len(causal_genes_array))
# initialize a 1D array for storing normalized scores with size=len(number of nodes in interactome): all elements=0
scores_normalized = numpy.zeros((len(causal_genes_array)))

# iterate over distances 1 to 4
for d in range(1, 5):
    # get the adjacency matrix A^distance from dictionary
    A = dict_adjacency.get(d) 
    
    # calculate score for each node i in the following way:
    # score_i = sum_over_distances ( ALPHA^distance * ( sum_over_nonCausal_genes (A^distance)_ij * score_j ) )
    scores += ALPHA ** d * A.dot(causal_genes_array) 


    # calculate normalization factor for each node i in the following way:
    # norm_factor_i = sum_over_distances ( ALPHA^distance * ( sum_over_all_genes_j (A^distance)_ij ) )
    for i in range(len(norm_factors)):
        norm_factors[i] += ALPHA ** d * A[i, :].sum() 

# normalize score for each node i: score_normalized_i = score_i / normalization_factor_i
# note: to avoid division by zero, if normalization_factor_i=0, do as if normalization_factor_i=1
for i in range(len(scores)):
    scores_normalized[i] = scores[i] / norm_factors[i] if norm_factors[i] != 0 else scores[i]

# create a dictionary to with gene names mapped to scores with structure
#   {node: score,
#   ...}
dict_scores = dict(zip(G.nodes(), scores_normalized))

# create a dictionary based on the dictionary with scores for each node sorted by scores (dict values)
dict_scores_sorted = dict(sorted(dict_scores.items(), key=lambda v: v[1], reverse=True))

# from the dictionary remove node-score (key-value) pairs if node is a causal gene
dict_scores_sorted = {k: v for k, v in dict_scores_sorted.items() if k not in causal_genes}

'''
Step 5. Get more info about the genes (scores, degrees, #causal genes at various distances)
'''
# initialize a dictionary to store more information about each node
dict_final = {}

# iterate over node-score (key-value) pairs in the dictionary with sorted scores
for node, score in dict_scores_sorted.items():
    # update the previously initialized dictionary for more info to inlcude:
    # score, node degree, causal genes at various distances (1 to 4)
    dict_final[node] = [round(score, 4),
                        G.degree(node),
                        causal_genes_at_distance(dict_distances, node, 1), 
                        causal_genes_at_distance(dict_distances, node, 2),
                        causal_genes_at_distance(dict_distances, node, 3),
                        causal_genes_at_distance(dict_distances, node, 4)]

# to represent results in an easily readable way, create a pandas dataframe with columns:
# ['score', 'degree', 'causal genes at d=1', 'causal genes at d=2', 'causal genes at d=3', 'causal genes at d=4']
df = pandas.DataFrame.from_dict(dict_final, 
                            orient='index', 
                            columns=['score', 'degree', 'causal genes at d=1', 'causal genes at d=2', 'causal genes at d=3', 'causal genes at d=4'])

# print top (highest-scoring) 20 non-causal genes
print(df.head(20))

# create a scatter plot score vs. degree where each data point represents one node
seaborn.scatterplot(data=df, x="degree", y="score", hue='causal genes at d=1')
matplotlib.pyplot.title("New centrality scores vs. node degree")

# save the plot to .png file in the same directory as the script
matplotlib.pyplot.savefig("./test_newCentrality_v3_forNicolas.png")