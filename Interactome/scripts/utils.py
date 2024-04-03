import logging
import networkx


# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


def parse_interactome(interactome_file) -> networkx.Graph:
    '''
    Creates a networkx.Graph representing the interactome

    arguments:
    - interactome_file: filename (with path) of interactome in SIF format (ie
      3 tab-separated columns: ENSG1 pp ENSG2), type=str

    returns:
    - interactome: type=networkx.Graph
    '''
    interactome = networkx.Graph()
    num_edges = 0

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
        num_edges += 1

    logger.info("built non-redundant network from %i non-self interactions, resulting in %i edges",
                num_edges, len(interactome.edges()))
    return (interactome)

def parse_gene2ENSG(gene2ENSG_file):
    '''
    Build dicts mapping ENSGs to gene_names and gene_names to ENSGs

    arguments:
    - gene2ENSG_file: filename (with path) of TSV file mapping gene names to ENSGs, 
      type=str with 2 columns: GENE ENSG

    returns 2 dicts of all genes in the gene2ENSG_file:
    - ENSG2gene: key=ENSG, value=gene_name
    - gene2ENSG: key = gene_name, value=ENSG
    '''
    ENSG2gene = {}
    gene2ENSG = {}

    try:
        f_gene2ENSG = open(gene2ENSG_file, 'r')
    except Exception as e:
        logger.error("Opening provided gene2ENSG file %s: %s", gene2ENSG_file, e)
        raise Exception("cannot open provided gene2ENSG file")

    # check header and skip
    line = next(f_gene2ENSG)
    line = line.rstrip()
    if line != "GENE\tENSG":
        logger.error("gene2ENSG file %s doesn't have the expected header",
                     gene2ENSG_file)
        raise Exception("Bad header in the gene2ENSG file")

    for line in f_gene2ENSG:
        split_line = line.rstrip().split('\t')
        if len(split_line) != 2:
            logger.error("gene2ENSG file %s has bad line (not 2 tab-separated fields): %s",
                         gene2ENSG_file, line)
            raise Exception("Bad line in the gene2ENSG file")
        gene_name, ENSG = split_line
        if ENSG in ENSG2gene:
            logger.warning("ENSG %s mapped multiple times in %s, keeping the first mapping",
                           ENSG, gene2ENSG_file)
        else:
            ENSG2gene[ENSG] = gene_name
        if gene_name in gene2ENSG:
            logger.warning("gene_name %s mapped multiple times in %s, keeping the first mapping",
                           gene_name, gene2ENSG_file)
        else:
            gene2ENSG[gene_name] = ENSG

    f_gene2ENSG.close()
    return(ENSG2gene, gene2ENSG)


def parse_causal_genes(causal_genes_file, gene2ENSG, patho) -> dict:
    '''
    Build a dict of causal ENSGs for patho

    arguments:
    - causal_genes_file: filename (with path) of known causal genes TSV file
      with 2 columns: gene_name pathologyID, type=str
    - gene2ENSG: dict of all known genes, key=gene_name, value=ENSG
    - pathologyID of interest, causal genes for other pathologyIDs are ignored

    returns:
    - causal_genes: dict of all causal genes with key=ENSG, value=1
    '''
    causal_genes = {}

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
        else:
            logger.warning("causal gene %s from file %s is not in gene2ENSG, skipping it",
                           gene_name, causal_genes_file)

    f_causal.close()

    logger.info("found %i causal genes with known ENSG for pathology %s",
                len(causal_genes), patho)
    return(causal_genes)


def scores_to_TSV(scores, ENSG2gene):
    '''
    Print scores to stdout in TSV format, 3 columns: ENSG gene_name score

    arguments:
    - scores: dict with key=ENSG, value=score
    - ENSG2gene: dict of all known gene names, key=ENSG, value=gene_name
    '''

    # header
    print("ENSG\tGENE\tSCORE")

    for (ENSG, score) in sorted(scores.items()):
        # GENE defaults to "" if we don't know the gene name of ENSG
        gene = ""
        if ENSG in ENSG2gene:
            gene = ENSG2gene[ENSG]
        print(ENSG + "\t" + gene + "\t" + str(score))

