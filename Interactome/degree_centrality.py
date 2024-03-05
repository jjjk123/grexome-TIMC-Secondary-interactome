import logging
import sys
import os

import argparse
import pathlib

import networkx


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


def calculate_degree_centrality(interactome) -> dict:
    '''
    Calculates degree centrality for every gene in the interactome.
    
    arguments:
    - interactome: type=networkx.Graph

    returns:
    - scores: dict with key=gene, value=degree_centrality
    '''
    scores = networkx.degree_centrality(interactome)

    return scores


def scores_to_TSV(scores, out_path):
    '''
    Save scoring results to a TSV file with 2 columns: gene, score.

    arguments:
    - scores: dict with key=gene, value=score
    - out_path: path to save TSV, type=pathlib.Path
    '''
    out_file = out_path / "degree_centrality.tsv"
    f = open(out_file, 'w+')

    # file header
    f.write("node\t" + "DC" + '\n')

    for node, score in scores.items():
        f.write(str(node) + '\t' + str(score) + '\n')

    f.close()


def main(interactome_file, out_path):

    logger.info("Parsing interactome")
    interactome, genes = parse_interactome(interactome_file)

    logger.info("Calculating degree centrality")
    degree_centrality = calculate_degree_centrality(interactome)

    logger.info("Done!")
    scores_to_TSV(degree_centrality, out_path)


if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(script_name)

    parser = argparse.ArgumentParser(
        prog="degree_centrality.py",
        description="Calculate degree centrality for human interactome."
    )

    parser.add_argument('-i', '--interactome_file', type=pathlib.Path)
    parser.add_argument('-o', '--out_path', type=pathlib.Path)

    args = parser.parse_args()

    try:
        main(interactome_file=args.interactome_file,
             out_path=args.out_path)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)