import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.indexing import ColBERTIndexer
from itertools import islice

def main():
    trec_ds = pt.get_dataset("trec-deep-learning-passages")
    indexer = ColBERTIndexer("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./trec_index", "trec", 16)
    indexref = indexer.index(trec_ds.get_corpus_iter())


if __name__ == "__main__":
    main()
