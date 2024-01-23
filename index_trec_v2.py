import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.indexing import ColBERTIndexer

def main():
    trec_ds = pt.get_dataset("trec-deep-learning-passages")
    indexer = ColBERTIndexer("../colbertv2.dnn", "./trec_index_v2", "trec", 16)
    indexref = indexer.index(trec_ds.get_corpus_iter())


if __name__ == "__main__":
    main()
