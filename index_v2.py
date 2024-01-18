from colbertv2.infra import Run, RunConfig, ColBERTConfig
from colbertv2 import Indexer
from pathlib import Path

def main():
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            nbits=2,
            root="./v2_experiments",
        )
        indexer = Indexer(checkpoint=str((Path.home() / "colbertv2.0").absolute()), config=config)
        indexer.index(name="msmarco.nbits=2", collection=str((Path.home() / "msmarco/collection.tsv").absolute()))


if __name__ == "__main__":
    main()