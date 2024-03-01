import logging

import os
import sys

import argparse
import pathlib

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


def TSV_to_SIF(interactome_tsv, out_file):
    '''
    Reads interactome TSV and saves it to SIF.

    arguments:
    - interactome_tsv: .tsv file with 3 columns: gene1 gene2 pp
    - out_file: output .sif file name
    '''

    # load interactome TSV file
    try:
        tsv_file = open(interactome_tsv, 'r')
    except Exception as e:
        logger.error("Opening provided .tsv interactome file %s: %s", interactome_tsv, e)
        raise Exception("Cannot open provided interactome file")
    
    # open SIF file to save the interactome
    sif_file = open(out_file, 'w+')

    # read interactions from TSV
    for line in tsv_file:
        line_splitted = line.rstrip().split('\t')
        if len(line_splitted) != 3:
            logger.error(".tsv file %s has bad line (not 3 tab-separated fields): %s", interactome_tsv, line)
            raise Exception("Bad line in the interactome file")

        gene1, gene2, pp = line_splitted

        # convert interaction to SIF
        sif_line = gene1 + '\t' + "pp" + '\t' + gene2 + '\n' 
        sif_file.write(sif_line)

    tsv_file.close()
    sif_file.close()


def main(args):
    interactome_tsv = args.interactome_tsv
    out_file = args.out_file

    TSV_to_SIF(interactome_tsv, out_file)

if __name__ == '__main__':
    script_name = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(script_name)

    # parse arguments
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Converts interactome TSV to SIF."
    )

    parser.add_argument('-i', '--interactome_tsv', type=pathlib.Path)
    parser.add_argument('-o', '--out_file', type=pathlib.Path)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + script_name + " : " + repr(e) + "\n")
        sys.exit(1)