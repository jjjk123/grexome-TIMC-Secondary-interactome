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

    arguments:
    - interactome: type=networkx.Graph
    - causal_genes: dict with key=gene, value=1 if causal, 0 otherwise
    NOTE: alpha hardcoded

    returns:
    - scores: dict with key=gene, value=score
    '''
    # 1D numpy array for genes in the interactome: 1 if causal gene, 0 otherwise, size=len(nodes in interactome)
    causal_genes_array = numpy.array([1 if causal_genes.get(n) == 1 else 0 for n in interactome.nodes()])

    scores_array = numpy.zeros((len(causal_genes_array)))
    norm_factors_array = numpy.zeros((len(causal_genes_array)))

    # calculate normalized scores
    for d in range(1, len(adjacency_matrices)):
        A = adjacency_matrices[d]

        # numpy.dot is not aware of sparse arrays, todense() should be used
        scores_array += alpha ** d * numpy.dot(A.todense(), causal_genes_array)

        norm_factors_array += (alpha / norm_alpha_div) ** d * A.sum(axis=0)

    scores_array_normalized = numpy.squeeze(scores_array / norm_factors_array)

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
      is A**i (except the diagonal is zeroed) where A is the adjacency matrix of interactome
    '''
    # initialize, element at index 0 is never used
    adjacency_matrices = [0]

    A = networkx.to_scipy_sparse_array(interactome)  # returns scipy.sparse._csr.csr_array
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


def main(interactome_file, causal_genes_file, patho="MMAF", gene2ENSG_file, out_path, alpha=0.5, norm_alpha_div=1.0, max_power=5, out_file="scores.tsv"):

    logger.info("Parsing interactome")
    interactome = utils.parse_interactome(interactome_file)

    logger.info("Parsing causal genes")
    causal_genes = utils.parse_causal_genes(causal_genes_file, gene2ENSG_file, patho)

    logger.info("Calculating powers of adjacency matrix")
    adjacency_matrices = get_adjacency_matrices(interactome, max_power=max_power)

    logger.info("Calculating scores")
    scores = calculate_scores(interactome, adjacency_matrices, causal_genes, alpha=alpha, norm_alpha_div=norm_alpha_div)

    logger.info("Done!")
    utils.scores_to_TSV(scores, out_path, file_name=out_file)


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
    parser.add_argument('-o', '--out_path', type=pathlib.Path)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--norm_alpha_div', type=float)
    parser.add_argument('--max_power', type=int)
    parser.add_argument('--out_file', type=str)

    args = parser.parse_args()

    try:
        main(interactome_file=args.interactome_file,
             causal_genes_file=args.causal_genes_file,
             patho=args.patho,
             gene2ENSG_file=args.gene2ENSG_file,
             out_path=args.out_path,
             alpha=args.alpha,
             norm_alpha_div=args.norm_alpha_div,
             max_power=args.max_power,
             out_file=args.out_file)

    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)
