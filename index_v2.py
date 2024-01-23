import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.indexing import ColBERTIndexer

def main():
    msmarco_ds = pt.get_dataset("irds:msmarco-passage")
    indexer = ColBERTIndexer("../colbertv2.dnn", "./msmarco_index_v2", "msmarco", 16)
    indexref = indexer.index(msmarco_ds.get_corpus_iter())


if __name__ == "__main__":
    main()
