import logging
import networkx


# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


def parse_interactome(interactome_file) -> networkx.Graph:
    '''
    Creates a networkx.Graph representing the interactome

    arguments:
    - interactome_file: filename (with path) of interactome in SIF format (ie
    3 tab-separated columns: gene1 pp gene2), type=str

    returns:
    - interactome: type=networkx.Graph
    '''
    interactome = networkx.Graph()

    try:
        f = open(interactome_file, 'r')
    except Exception as e:
        logger.error("Opening provided SIF interactome file %s: %s", interactome_file, e)
        raise Exception("cannot open provided interactome file")

    for line in f:
        split_line = line.rstrip().split('\t')
        if len(split_line) != 3:
            logger.error("SIF file %s has bad line (not 3 tab-separated fields): %s", interactome_file, line)
            raise Exception("Bad line in the interactome file")

        gene1, pp, gene2 = line_splitted

        # exclude self-interactions
        if gene1 == gene2:
            continue
        # else: ppopulate structures
        interactome.add_edge(gene1, gene2)

    return (interactome)


def parse_causal_genes(causal_genes_file, gene2ENSG_file, genes) -> dict:
    '''
    Creates a dictionary of all genes in the interactome, key=ENSG, value=1 if causal, 0 otherwise.

    arguments:
    - causal_genes_file: path to known causal genes CSV file, type=str
      with 2 columns: gene, pathology
    - canonical_genes_file: path to canonical genes, type=str with 2
      columns: gene_name, ENSG
    - genes: dict with key=gene, value=0
    NOTE: pathology of interest is hardcoded

    returns:
    - causal_genes: dict with key=ENSG, value=1 if causal, 0 otherwise
    '''
    causal_genes = genes.copy()
    canonical_genes = {}

    # first, parse canonical genes
    try:
        f_canonical = open(canonical_genes_file, 'r')
    except Exception as e:
        logger.error("Opening provided canonical genes file %s: %s", canonical_genes_file, e)
        raise Exception("cannot open provided canonical genes file")

    # skip header
    next(f_canonical)

    for line in f_canonical:
        line_splitted = line.rstrip().split('\t')

        gene_name, ENSG = line_splitted

        # populate canonical genes dictionary
        canonical_genes[gene_name] = ENSG

    f_canonical.close()

    # second, parse causal genes
    try:
        f_causal = open(causal_genes_file, 'r')
    except Exception as e:
        logger.error("Opening provided causal genes file %s: %s", causal_genes_file, e)
        raise Exception("cannot open provided causal genes file")

    # doesn't this file have a header? (should be documented in top-of-function comments)
    for line in f_causal:
        line_splitted = line.rstrip().split('\t')

        gene_name, pathology = line_splitted

        # map gene names to ENSGs
        if gene_name in canonical_genes:
            ENSG = canonical_genes[gene_name]
        else:
            continue

        # skip genes not present in the interactome
        if ENSG not in genes:
            continue

        # populate structures
        if pathology == "MMAF":
            causal_genes[ENSG] = 1

    f_causal.close()

    return causal_genes


def scores_to_TSV(scores, out_path, file_name="scores.tsv"):
    '''
    Save scoring results to a TSV file with 2 columns: gene, score.

    arguments:
    - scores: dict with key=gene, value=score
    - out_path: path to save TSV, type=pathlib.Path
    '''
    out_file = out_path / file_name
    f = open(out_file, 'w+')

    # file header
    f.write("node" + "\t" + "score" + '\n')

    for (node, score) in sorted(scores.items()):
        f.write(str(node) + '\t' + str(score) + '\n')

    f.close()
