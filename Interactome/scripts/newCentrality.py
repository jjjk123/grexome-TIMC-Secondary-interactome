import logging
import sys
import os

import argparse
import pathlib

import numpy

import networkx

import utils

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


def calculate_scores(interactome, adjacency_matrices, causal_genes, alpha=0.5, norm_alpha_div=1.0) -> dict:
    '''
    Calculates scores for every gene in the interactome based on the proximity to causal genes.
    NEED TO PROVIDE THE FORMULA HERE (user doesn't want to have to read the code to know what score
    you are calculating)

    arguments:
    - interactome: type=networkx.Graph
    - causal_genes: dict of causal genes with key=ENSG, value=1

    returns:
    - scores: dict with key=ENSG, value=score
    '''
    # 1D numpy array for genes in the interactome: 1 if causal gene, 0 otherwise,
    # size=len(nodes in interactome), ordered as in interactome.nodes()
    causal_genes_vec = numpy.zeros(len(interactome.nodes()), dtype=numpy.uint8)
    ni = 0
    for n in interactome.nodes():
        if n in causal_genes:
            causal_genes_vec[ni] = 1
        ni += 1

    scores_vec = numpy.zeros(len(causal_genes_array))
    norm_factors_vec = numpy.zeros(len(causal_genes_array))
    ones_vec = numpy.ones(len(causal_genes_array))

    # calculate normalized scores
    for d in range(1, len(adjacency_matrices)):
        A = adjacency_matrices[d]
        scores_vec += alpha ** d * A.dot(causal_genes_vec)
        norm_factors_vec += (alpha / norm_alpha_div) ** d * A.dot(ones_vec)

    scores_array_normalized = scores_vec / norm_factors_vec

    # map ENSGs to scores
    scores = dict(zip(interactome.nodes(), scores_array_normalized))

    return scores


def get_adjacency_matrices(interactome, max_power=5):
    '''
    Calculates powers of adjacency matrix.

    arguments:
    - interactome: type=networkx.Graph
    - max_power: int

    returns:
    - adjacency_matrices: list of scipy sparse arrays, array at index i (starting at i==1)
      is A**i (except the diagonal is zeroed) where A is the adjacency matrix of interactome,
      rows and columns are ordered as in interactome.nodes()
    '''
    # initialize, element at index 0 is never used
    adjacency_matrices = [0]

    A = networkx.to_scipy_sparse_array(interactome, dtype=numpy.uint32)  # returns scipy.sparse._csr.csr_array
    res = A
    res.setdiag(0)
    adjacency_matrices.append(res)

    # @ - matrix multiplication
    for power in range(2, max_power + 1):
        res = res @ A
        res.setdiag(0)
        adjacency_matrices.append(res)

    logger.debug("Done building %i matrices", len(adjacency_matrices) - 1)
    return adjacency_matrices


def main(interactome_file, causal_genes_file, patho="MMAF", gene2ENSG_file, alpha=0.5, norm_alpha_div=1.0, max_power=5):

    logger.info("Parsing interactome")
    interactome = utils.parse_interactome(interactome_file)

    logger.info("Parsing gene-to-ENSG mapping")
    (ENSG2gene, gene2ENSG) = utils.parse_gene2ENSG(gene2ENSG_file)

    logger.info("Parsing causal genes")
    causal_genes = utils.parse_causal_genes(causal_genes_file, gene2ENSG, patho)

    logger.info("Calculating powers of adjacency matrix")
    adjacency_matrices = get_adjacency_matrices(interactome, max_power=max_power)

    logger.info("Calculating scores")
    scores = calculate_scores(interactome, adjacency_matrices, causal_genes, alpha=alpha, norm_alpha_div=norm_alpha_div)

    logger.info("Printing scores")
    utils.scores_to_TSV(scores, ENSG2gene)

    logger.info("Done!")


if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(script_name)

    parser = argparse.ArgumentParser(
        prog=script_name,
        description="Calculate new centrality for new candidates of infertility based on the guilt-by-association approach."
    )

    parser.add_argument('-i', '--interactome_file', type=pathlib.Path)
    parser.add_argument('--causal_genes_file', type=pathlib.Path)
    parser.add_argument('--patho', type=str)
    parser.add_argument('--gene2ENSG_file', type=pathlib.Path)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--norm_alpha_div', type=float)
    parser.add_argument('--max_power', type=int)

    args = parser.parse_args()

    try:
        main(interactome_file=args.interactome_file,
             causal_genes_file=args.causal_genes_file,
             patho=args.patho,
             gene2ENSG_file=args.gene2ENSG_file,
             alpha=args.alpha,
             norm_alpha_div=args.norm_alpha_div,
             max_power=args.max_power)

    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)
