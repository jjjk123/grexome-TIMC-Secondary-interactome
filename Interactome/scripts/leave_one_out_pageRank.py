import logging
import os
import sys

import pathlib

import argparse

import pageRankCentrality
import utils


def leave_one_out(interactome, adjacency_matrices, causal_genes, alpha):
    '''
    arguments:
    - interactome: type=networkx.Graph
    - adjacency_matrices: list of scipy sparse arrays as returned by pageRankCentrality.get_adjacency_matrices()
    - causal_genes: dict of causal genes with key=ENSG, value=1

    returns:
    - scores_left_out: dict with key=causal gene (ENSG), value=score of this gene 
      when it is left out
    ''' 
    # initialize dict to store left-out scores
    scores_left_out = {}

    for left_out in list(causal_genes.keys()):
        logger.info("Leaving out %s", left_out)
        del causal_genes[left_out]
        scores = pageRankCentrality.calculate_scores(interactome, adjacency_matrices, causal_genes, alpha)
        scores_left_out[left_out] = scores[left_out]
        causal_genes[left_out] = 1

    return scores_left_out 


def main(interactome_file, causal_genes_file, gene2ENSG_file, patho, alpha, max_power):

    logger.info("Parsing interactome")
    interactome = utils.parse_interactome(interactome_file)

    logger.info("Parsing gene-to-ENSG mapping")
    (ENSG2gene, gene2ENSG) = utils.parse_gene2ENSG(gene2ENSG_file)

    logger.info("Parsing causal genes")
    causal_genes = utils.parse_causal_genes(causal_genes_file, gene2ENSG, interactome, patho)

    logger.info("Calculating powers of adjacency matrix")
    adjacency_matrices = pageRankCentrality.get_adjacency_matrices(interactome, max_power)

    logger.info("Calculating leave-one-out scores")
    scores = leave_one_out(interactome, adjacency_matrices, causal_genes, alpha)

    logger.info("Printing leave-one-out scores")
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
        prog="newCentrality.py",
        description="Calculate leave-one-out for PageRank centrality of infertility based on the guilt-by-association approach."
    )

    parser.add_argument('-i', '--interactome_file', type=pathlib.Path, required=True)
    parser.add_argument('--causal_genes_file', type=pathlib.Path, required=True)
    parser.add_argument('--gene2ENSG_file', type=pathlib.Path, required=True)
    parser.add_argument('--patho', default='MMAF', type=str)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--max_power', default=5, type=int) 

    args = parser.parse_args()

    try:
        main(interactome_file=args.interactome_file,
             causal_genes_file=args.causal_genes_file,
             gene2ENSG_file=args.gene2ENSG_file,
             patho=args.patho,
             alpha=args.alpha,
             max_power=args.max_power)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)
