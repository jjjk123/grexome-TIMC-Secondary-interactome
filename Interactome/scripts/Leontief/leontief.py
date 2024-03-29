import logging
import sys
import os

import argparse
import pathlib

import numpy
import scipy

import networkx

sys.path.append("../scripts")
from utils import parse_interactome, parse_causal_genes, scores_to_TSV

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


def calculate_scores(interactome, causal_genes, L, L_norm, alpha=0.5) -> dict:
    '''
    Calculates scores for every gene in the interactome based on the proximity to causal genes.

    arguments:
    - interactome: type=networkx.Graph
    - causal_genes: dict with key=gene, value=1 if causal, 0 otherwise
    - alpha: type=float

    returns:
    - scores: dict with key=gene, value=score
    '''
    # 1D numpy array for genes in the interactome: 1 if causal gene, 0 otherwise, size=len(nodes in interactome)
    causal_genes_array = numpy.array([1 if causal_genes.get(n) == 1 else 0 for n in interactome.nodes()])

    # A = networkx.to_scipy_sparse_array(interactome, format='csc')
    # A = A / A.sum(axis=0)

    # I = scipy.sparse.eye_array(m=A.shape[0])
    # L = scipy.sparse.linalg.inv((I - alpha*A)) - I
    # L_norm = scipy.sparse.linalg.inv((I - (alpha / div)*A)) - I
    L = L.copy()
    L_norm = L_norm.copy()

    scores_array = numpy.dot(L.todense(), causal_genes_array)
    ones_arr = numpy.ones(shape=L_norm.shape[0])
    norm_array = numpy.dot(L_norm.todense(), ones_arr)

    scores_array_normalized = numpy.squeeze(scores_array / norm_array)
    
    # map ENSGs to scores
    scores = dict(zip(interactome.nodes(), scores_array_normalized))

    return scores


def main(interactome_file, causal_genes_file, canonical_genes_file, out_path, alpha=0.5, out_file="scores.tsv"):

    logger.info("Parsing interactome")
    interactome, genes = parse_interactome(interactome_file)

    logger.info("Parsing causal genes")
    causal_genes = parse_causal_genes(causal_genes_file, canonical_genes_file, genes)

    logger.info("Calculating scores")
    scores = calculate_scores(interactome, causal_genes, alpha=alpha)

    logger.info("Done!")
    scores_to_TSV(scores, out_path, file_name=out_file)


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
    parser.add_argument('-o', '--out_path', type=pathlib.Path)
    parser.add_argument('--alpha', type=float) 
    parser.add_argument('--out_file', type=str)

    args = parser.parse_args()

    try:
        main(interactome_file=args.interactome_file,
             causal_genes_file=args.causal_genes_file,
             canonical_genes_file=args.canonical_genes_file,
             out_path=args.out_path,
             alpha=args.alpha,
             out_file=args.out_file)
        
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)