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
            logger.error("SIF file %s has bad line (not 3 tab-separated fields): %s",
                         interactome_file, line)
            raise Exception("Bad line in the interactome file")

        gene1, pp, gene2 = split_line

        # exclude self-interactions
        if gene1 == gene2:
            continue
        # else: ppopulate structures
        interactome.add_edge(gene1, gene2)

    return (interactome)


def parse_causal_genes(causal_genes_file, gene2ENSG_file, patho="MMAF") -> dict:
    '''
    Build a dict of causal ENSGs for patho

    arguments:
    - causal_genes_file: filename (with path) of known causal genes TSV file
      with 2 columns: gene_name pathologyID, type=str
    - gene2ENSG_file: filename (with path) of TSV file mapping gene names to ENSGs, 
      type=str with 2 columns: gene_name, ENSG
    - pathologyID of interest, causal genes for other pathologyIDs are ignored

    returns:
    - causal_genes: dict of all causal genes with key=ENSG, value=1
    '''
    causal_genes = {}

    # populate gene2ENSG: key=gene_name, value=ENSG
    gene2ENSG = {}
    try:
        f_gene2ENSG = open(gene2ENSG_file, 'r')
    except Exception as e:
        logger.error("Opening provided gene2ENSG file %s: %s", gene2ENSG_file, e)
        raise Exception("cannot open provided gene2ENSG file")

    # not sure if there's a header line -> just put everything in gene2ENSG
    for line in f_gene2ENSG:
        split_line = line.rstrip().split('\t')
        if len(split_line) != 2:
            logger.error("gene2ENSG file %s has bad line (not 2 tab-separated fields): %s",
                         gene2ENSG_file, line)
            raise Exception("Bad line in the gene2ENSG file")
        gene_name, ENSG = split_line
        gene2ENSG[gene_name] = ENSG

    f_gene2ENSG.close()

    # parse causal genes
    try:
        f_causal = open(causal_genes_file, 'r')
    except Exception as e:
        logger.error("Opening provided causal genes file %s: %s", causal_genes_file, e)
        raise Exception("cannot open provided causal genes file")

    # not sure if there's a header line, assume there isn't
    for line in f_causal:
        split_line = line.rstrip().split('\t')
        if len(split_line) != 2:
            logger.error("causal_genes file %s has bad line (not 2 tab-separated fields): %s",
                         causal_genes_file, line)
            raise Exception("Bad line in the causal_genes file")
        gene_name, pathology = split_line

        if pathology != patho:
            continue
        elif gene_name in gene2ENSG:
            ENSG = gene2ENSG[gene_name]
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
