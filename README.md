# Experiments for "Behind the [MASK]: An Analysis of Structural Tokens in ColBERT"

## Installation
Make sure you've cloned the submodules as well. Install the environment with the following command:

```bash
conda env create -f environment.yml
``` 

If you run into errors saying `pyterrier_colbert` can't be found, you may have to manually install it into the
environment.

```bash
cd pyterrier_colbert
pip install -e .
```

You will also need a Rust compiler (available [here](https://www.rust-lang.org/learn/get-started)) to install our accelerated ranker.

```bash
cd colbert_rs
./build-and-install.sh
```

## Running Experiments

### Using ColBERT v2

Before ColBERT v2 can be used, it has to be converted into a `.dnn` checkpoint. This can be done with the following commands:

```bash
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz    # Downloads the official ColBERTv2 checkpoint
unzip colbertv2.0.tar.gz
python v2_to_v1_chkpt.py --v2-dir ./colbertv2.0 --out-file ../colbertv2.dnn             # The experiments expect the checkpoint to be in the outer directory
```

### Indexing

We use the MS MARCO and TREC 2019-2020 corpuses for our experiments. The TREC 2019-2020 index should technically
contain the same documents as the MS MARCO index. Indexing can be performed with the following.

```bash
python index.py             # For MS MARCO
python index_trec.py        # For TREC 2019-2020
python index_v2.py          # For MS MARCO (ColBERT v2)
python index_trec_v2.py     # For TREC 2019-2020 (ColBERT v2)

mv msmarco_index/msmarco/ivfpq.100.faiss msmarco_index/msmarco/ivfpq.faiss
mv msmarco_index_v2/msmarco/ivfpq.100.faiss msmarco_index_v2/msmarco/ivfpq.faiss
mv trec_index/trec/ivfpq.100.faiss trec_index/trec/ivfpq.faiss
mv trec_index_v2/trec/ivfpq.100.faiss trec_index_v2/trec/ivfpq.faiss

mkdir results
```

### Visualization Tool

We've developed a tool to visualize ColBERT embeddings and what they map to, found here:
https://github.com/Boxxfish/vis_colbert. This tool requires collecting embeddings from queries and statistics on what
they match to, then compiling said data into files for the frontend. This process can be run with the following command.

```bash
python collect_d_embs.py
python compute_query_data.py --queries NUM_QUERIES --k TOP_K_DOCS
python process_query_data.py
```

This also generates a 3D PCA model fit on all document embeddings (`pca.pkl`), useful for further visualization. Many
other experiments use `pca_2d.pkl`; run `create_2d_pca.py` after `collect_d_embs.py` to get this model.

### Key Experiments

To run an experiment with ColBERT v2, use the `--v2` flag.

**Shift in query embeddings after moving two words from the beginning to the end**

```bash
mkdir shift_artifacts
python compute_query_shift.py [--v2]
python analyze_query_shift.py --experiment-compare-dists [--v2]
```

**Remap [MASK] embeddings to nearest query text embedding**

```bash
python cluster_mapping.py [--v2]
python compare_res_cluster.py [--v2]
```

**Can [CLS] and [SEP] act as dense retrievers?**

```bash
python analyze_dense_retriever.py
```

**Perturbing [MASK]s**

```bash
python analyze_masks.py --experiment-adapt-masks    # Does performance get worse if we remove half of the [MASK]s before contextualization?
python analyze_masks.py --visualize-adapt-masks     # What do [MASK] embeddings look like if half the [MASK]s are removed?
python analyze_masks.py --visualize-query-mask     # What do query text embeddings look like if [MASK]s are removed?
python analyze_masks.py --visualize-contiguous     # What do query text embeddings look like if [MASK]s are placed before the query text?
```

**How do [Q] and [D] affect the query?**

```bash
python analyze_d_q_mask.py --compute-embeddings
python analyze_d_q_mask.py --analyze            # Project embeddings to 2D and show how they change.
python analyze_d_q_mask.py --experiment         # Check how changing [Q] to [D] affects performance.
```

**How does ColBERT do in retrieving a passage within a document?**

```bash
python analyze_intrapassage.py
```

### Helper Scripts

To facilitate running experiments, we've developed a number of helper scripts, described below.

- `trec_utils.py`: Combines the TREC 2019 and TREC 2020 test sets and returns the queries and qrels.
- `mod_utils.py`: Contains a number of utilities for modifying the query before and after contextualization.
- `metrics.py`: Contains the evalulation metrics used across all experiments.
