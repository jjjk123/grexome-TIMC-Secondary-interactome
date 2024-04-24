import logging
import sys
import os

import argparse
import pathlib

import networkx

import utils

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


def calculate_betweenness_centrality(interactome) -> dict:
    '''
    Calculates betweenness centrality for every gene in the interactome.
    
    arguments:
    - interactome: type=networkx.Graph

    returns:
    - scores: dict with key=gene, value=betweenness_centrality
    '''
    scores = networkx.betweenness_centrality(interactome)

    return scores

def main(interactome_file, gene2ENSG_file):

    logger.info("Parsing interactome")
    interactome = utils.parse_interactome(interactome_file)

    logger.info("Parsing gene-to-ENSG mapping")
    (ENSG2gene, gene2ENSG) = utils.parse_gene2ENSG(gene2ENSG_file)

    logger.info("Calculating scores")
    scores = calculate_betweenness_centrality(interactome)

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
        prog="betweenness_centrality.py",
        description="Calculate betweenness centrality for human interactome."
    )

    parser.add_argument('-i', '--interactome_file', type=pathlib.Path, required=True)
    parser.add_argument('--gene2ENSG_file', type=pathlib.Path, required=True)

    args = parser.parse_args()

    try:
        main(interactome_file=args.interactome_file,
             gene2ENSG_file=args.gene2ENSG_file)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)