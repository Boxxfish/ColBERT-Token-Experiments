import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.indexing import ColBERTIndexer
from itertools import islice

def main():
    msmarco_ds = pt.get_dataset("irds:msmarco-passage")
    indexer = ColBERTIndexer("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco", 16)
    indexref = indexer.index(msmarco_ds.get_corpus_iter())


if __name__ == "__main__":
    main()
