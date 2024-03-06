import logging
import sys
import os

import multiprocessing

import argparse
import pathlib

from newCentrality_v4 import parse_interactome, parse_causal_genes, get_adjacency_matrices, calculate_scores, scores_to_TSV


def leave_one_out(interactome, causal_genes, out_path):
    '''
    arguments:
    - interactome: type=networkx.Graph
    - causal_genes: dict with key=gene, value=1 if causal, 0 otherwise

    Saves new scores to TSV for each left-out gene
    and plots scores vs. left-out scores.
    ''' 
    logger.info("Calculating adjacency matrices")
    adjacency_matrices = get_adjacency_matrices(interactome, max_power=5)


    # initialize dict to store left-out scores
    dict_left_out = {}

    causal_genes_list = [k for k, v in causal_genes.items() if v == 1]
    for left_out in causal_genes_list:
        logger.info("Leaving out %s", left_out)
        causal_genes_new = causal_genes.copy()
        causal_genes_new[left_out] = 0

        logger.info("Calculating scores")
        scores = calculate_scores(interactome, adjacency_matrices, causal_genes_new)
        
        logger.info("Saving scores to TSV")
        scores_to_TSV(scores, out_path, file_name=f"{left_out}_scores.tsv")


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