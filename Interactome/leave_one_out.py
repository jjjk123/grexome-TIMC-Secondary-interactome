import logging
import sys
import os

import multiprocessing

import argparse
import pathlib

import numpy

from newCentrality_v4 import parse_interactome, parse_causal_genes, get_adjacency_matrices


def leave_one_out(interactome, causal_genes, out_path):
    '''
    arguments:
    - interactome: type=networkx.Graph
    - causal_genes: dict with key=gene, value=1 if causal, 0 otherwise

    returns:
    - scores: dict with key=gene, value=1 if causal, 0 otherwise
    ''' 
    logger.info("Calculating adjacency matrices")
    adjacency_matrices = get_adjacency_matrices(interactome, max_power=5)

    causal_genes_list = [k for k, v in causal_genes.items() if v == 1]
    
    for left_out in causal_genes_list:
        logger.info("Leaving out %s", left_out)
        causal_genes_new = causal_genes.copy()
        causal_genes_new[left_out] = 0

        logger.info("Calculating scores")
        scores = calculate_scores(interactome, adjacency_matrices, causal_genes_new)
        
        logger.info("Saving scores to TSV")
        scores_to_TSV(left_out, scores, out_path)


def calculate_scores(interactome, adjacency_matrices, causal_genes, alpha=0.5, max_power=5) -> dict:
    '''
    Calculates scores for every gene in the interactome based on the proximity to causal genes.

    arguments:
    - interactome: type=networkx.Graph
    - adjacency_matrices: dict with key=power, value=adjacency_matrix**power
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
    for d in range(1, max_power+1):
        A = adjacency_matrices.get(d)

        # numpy.dot is not aware of sparse arrays, todense() should be used
        scores_array += alpha ** d * numpy.dot(A.todense(), causal_genes_array)

        norm_factors_array += alpha ** d * A.sum(axis=0)

    scores_array_normalized = numpy.squeeze(scores_array / norm_factors_array)

    # map ENSGs to scores
    scores = dict(zip(interactome.nodes(), scores_array_normalized))

    return scores


def scores_to_TSV(left_out, scores, out_path):
    '''
    Save scoring results to a TSV file with 2 columns: gene, score.

    arguments:
    - scores: dict with key=gene, value=score
    - out_path: path to save TSV, type=pathlib.Path
    '''
    out_file = out_path / f"{left_out}_scores.tsv"
    f = open(out_file, 'w+')

    # file header
    f.write("node" + "\t" + "score" + '\n')

    for node, score in scores.items():
        f.write(str(node) + '\t' + str(score) + '\n')

    f.close()


def main(interactome_file, causal_genes_file, canonical_genes_file, out_path):

    logger.info("Parsing interactome")
    interactome, genes = parse_interactome(interactome_file)

    logger.info("Parsing causal genes")
    causal_genes = parse_causal_genes(causal_genes_file, canonical_genes_file, genes)

    leave_one_out(interactome, causal_genes, out_path)


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
        description="Calculate leave-one-out for new centrality of infertility based on the guilt-by-association approach."
    )

    parser.add_argument('-i', '--interactome_file', type=pathlib.Path)
    parser.add_argument('--causal_genes_file', type=pathlib.Path)
    parser.add_argument('--canonical_genes_file', type=pathlib.Path)
    parser.add_argument('-o', '--out_path', type=pathlib.Path)

    args = parser.parse_args()

    try:
        main(interactome_file=args.interactome_file,
             causal_genes_file=args.causal_genes_file,
             canonical_genes_file=args.canonical_genes_file,
             out_path=args.out_path)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)