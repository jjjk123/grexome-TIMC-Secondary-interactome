import logging

import numpy

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


def parse_causal_genes(causal_genes_file, canonical_genes_file, genes) -> dict:
    '''
    Creates a dictionary of all genes in the interactome, key=gene, value=1 if causal, 0 otherwise.

    arguments:
    - causal_genes_file: path to known causal genes CSV file, type=pathlib.Path
      with 2 columns: gene, pathology
    - canonical_genes_file: path to canonical genes, type=pathlib.Path
      with 2 columns: gene_name, ENSG
    - pathology: phenotype for which to get the causal genes, type=str
    - genes: dict with key=gene, value=0
    NOTE: pathology hardcoded
    
    returns:
    - causal_genes: dict with key=gene, value=1 if causal, 0 otherwise
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

    for line in f_causal:
        line_splitted = line.rstrip().split('\t')

        gene_name, pathology = line_splitted

        # map gene names to ENSGs
        if gene_name in canonical_genes.keys():
            ENSG = canonical_genes.get(gene_name)
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


def calculate_scores(interactome, adjacency_matrices, causal_genes, alpha=0.5, max_power=5) -> dict:
    '''
    Calculates scores for every gene in the interactome based on the proximity to causal genes.

    arguments:
    - interactome: type=networkx.Graph
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

    for node, score in scores.items():
        f.write(str(node) + '\t' + str(score) + '\n')

    f.close()