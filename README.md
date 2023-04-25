# IR 2023 Final Project
Repo for the final project. Install with the following command:
```
conda env create -f environment.yml
``` 
An indexing demo where 1000 documents are indexed from MSMARCO is available at `index.py`. To test search after indexing, write the following in the REPL:
```python
from pyterrier_colbert.ranking import ColBERTFactory
pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco")
dense_e2e = pytcolbert.end_to_end()
dense_e2e.search("test")
```
