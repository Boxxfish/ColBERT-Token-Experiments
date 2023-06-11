use std::collections::HashSet;

use bitvec::prelude::*;
use half::f16;
use ndarray::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use slice_of_array::prelude::*;

/// Contains a contiguous shard of document token embeddings.
struct DocEmbeddingsShard {
    embeddings: Py<PyArray2<f16>>,
    start_offsets: Vec<usize>,
}

impl DocEmbeddingsShard {
    fn as_ref<'py>(&'py self, py: Python<'py>) -> DocEmbeddingsShardRef<'py> {
        let embeddings = self.embeddings.as_ref(py);
        embeddings.readonly();
        DocEmbeddingsShardRef {
            // SAFETY: If the embeddings array were mutably borrowed, then the
            // above .readonly() call would have panicked.
            // (.readonly() can't be used directly here because rustc complains
            // about returning a reference to a temporary.)
            embeddings: unsafe { embeddings.as_array() },
            start_offsets: &self.start_offsets,
        }
    }
}

#[pyclass]
pub struct Scorer {
    /// A Vec containing the token mask for each document. Within each mask, true means the token
    /// should be ignored and false means it should be used for scoring.
    doc_masks: Vec<BitBox>,
    /// The shards of document embeddings.
    doc_emb_shards: Vec<DocEmbeddingsShard>,
    /// The first PID in each shard. This Vec has the same length as doc_emb_shards.
    shard_pid_start_offsets: Vec<usize>,
}

#[pymethods]
impl Scorer {
    /// Construct a new Scorer.
    /// doc_token_offsets must have length (num_docs + 1).
    /// doc_emb_parts should be a list of tuples, each with:
    ///   - the array of indices in the doc embeddings array of the start of each doc
    ///   - the (N, embed_dim) doc embeddings array
    #[new]
    pub fn new(
        py: Python,
        doc_token_offsets: PyReadonlyArray1<i64>,
        all_token_ids: PyReadonlyArray1<i64>,
        doc_emb_parts: Vec<(PyReadonlyArray1<i64>, Py<PyArray2<f16>>)>,
        token_ids_to_prune: HashSet<i64>,
    ) -> Self {
        // compute the token masks for all documents
        let doc_token_offsets = doc_token_offsets
            .as_slice()
            .expect("doc_token_offsets should be contiguous");
        let all_token_ids = all_token_ids
            .as_slice()
            .expect("all_token_ids should be contiguous");

        let get_mask = |pid: usize| -> BitBox {
            let start_offset = doc_token_offsets[pid] as usize;
            let end_offset = doc_token_offsets[pid + 1] as usize;
            let doc_token_ids = &all_token_ids[start_offset..end_offset];
            doc_token_ids
                .iter()
                .map(|token_id| token_ids_to_prune.contains(token_id))
                .collect()
        };

        let num_docs = doc_token_offsets.len() - 1;
        let doc_masks = (0..num_docs).into_par_iter().map(get_mask).collect();

        // assemble the document embeddings shards
        let mut acc = 0;
        let shard_pid_start_offsets = doc_emb_parts
            .iter()
            .map(|(start_offsets, _)| {
                let shard_first_pid = acc;
                acc += start_offsets.len();
                shard_first_pid
            })
            .collect();
        let doc_emb_shards = doc_emb_parts
            .into_iter()
            .map(|(start_offsets, embeddings)| {
                let num_embs = embeddings.as_ref(py).shape()[0];
                DocEmbeddingsShard {
                    embeddings,
                    start_offsets: start_offsets
                        .as_array()
                        .iter()
                        .map(|i| *i as usize)
                        .chain(std::iter::once(num_embs))
                        .collect(),
                }
            })
            .collect();

        // return the final struct
        Scorer {
            doc_masks,
            doc_emb_shards,
            shard_pid_start_offsets,
        }
    }

    /// Score a batch of documents specified by their passage IDs.
    /// This function processes the documents in parallel, so prefer it over
    /// calling score_document in a loop when processing multiple documents.
    pub fn score_documents(
        &self,
        py: Python,
        query_embs: PyReadonlyArray2<f32>,
        pids: Vec<usize>,
    ) -> Vec<f32> {
        let query_embs = query_embs.as_array();
        let scorer = self.as_ref(py);
        py.allow_threads(move || scorer.score_documents(query_embs, pids))
    }

    /// Score a document specified by its passage ID.
    pub fn score_document(&self, py: Python, query_embs: PyReadonlyArray2<f32>, pid: usize) -> f32 {
        let query_embs = query_embs.as_array();
        let scorer = self.as_ref(py);
        py.allow_threads(move || scorer.score_document(query_embs, pid))
    }
}

impl Scorer {
    fn as_ref<'py>(&'py self, py: Python<'py>) -> ScorerRef<'py> {
        ScorerRef {
            doc_masks: &self.doc_masks,
            doc_emb_shards: self
                .doc_emb_shards
                .iter()
                .map(|shard| shard.as_ref(py))
                .collect(),
            shard_pid_start_offsets: &self.shard_pid_start_offsets,
        }
    }
}

/// A GIL-bound reference to a DocEmbeddingsShard.
struct DocEmbeddingsShardRef<'a> {
    embeddings: ArrayView2<'a, f16>,
    start_offsets: &'a [usize],
}

impl DocEmbeddingsShardRef<'_> {
    fn get_embeddings(&self, local_pid: usize) -> ArrayView2<f16> {
        let start_offset = self.start_offsets[local_pid];
        let end_offset = self.start_offsets[local_pid + 1];
        self.embeddings.slice(s![start_offset..end_offset, ..])
    }
}

/// A GIL-bound reference to a Scorer.
pub struct ScorerRef<'a> {
    doc_masks: &'a [BitBox],
    doc_emb_shards: Vec<DocEmbeddingsShardRef<'a>>,
    shard_pid_start_offsets: &'a [usize],
}

const EMBED_DIM: usize = 128;

// functions not (directly) exposed to Python
impl ScorerRef<'_> {
    /// Score a batch of documents specified by their passage IDs.
    /// This function processes the documents in parallel, so prefer it over
    /// calling score_document in a loop when processing multiple documents.
    pub fn score_documents(&self, query_embs: ArrayView2<f32>, pids: Vec<usize>) -> Vec<f32> {
        pids.into_par_iter()
            .map(move |pid| self.score_document(query_embs, pid))
            .collect()
    }

    /// Score a document specified by its passage ID.
    pub fn score_document(&self, query_embs: ArrayView2<f32>, pid: usize) -> f32 {
        // get the mask and embeddings for the document
        let mask = &self.doc_masks[pid];
        let doc_embs = self.doc_token_embs(pid);
        assert_eq!(doc_embs.shape()[0], mask.len());
        assert_eq!(doc_embs.shape()[1], EMBED_DIM);
        assert_eq!(query_embs.shape()[1], EMBED_DIM);

        // prune the doc embeddings and convert from f16 -> f32
        let mut doc_embs_pruned = Vec::with_capacity(doc_embs.shape()[0]);
        doc_embs_pruned.extend(
            doc_embs
                .outer_iter()
                .zip(mask.iter())
                .filter(|(_, is_pruned)| !**is_pruned)
                .map(|(doc_emb, _)| {
                    let doc_emb_f32: [f32; EMBED_DIM] =
                        array_init::from_iter(doc_emb.into_iter().map(|x| x.to_f32())).unwrap();
                    doc_emb_f32
                }),
        );
        let doc_embs_pruned =
            ArrayView2::from_shape((doc_embs_pruned.len(), EMBED_DIM), doc_embs_pruned.flat())
                .unwrap();

        // compute the dot products and reduce with max/sum to get the final score
        query_embs
            .dot(&doc_embs_pruned.t())
            .map_axis(Axis(1), |r| array_max(&r))
            .sum()
    }

    /// Returns a view of the token embeddings for the document with the given ID.
    fn doc_token_embs(&self, pid: usize) -> ArrayView2<f16> {
        // get the index of the shard that this PID belongs to
        let shard_index = self.shard_pid_start_offsets.partition_point(|x| *x <= pid) - 1;
        let shard_start_pid = self.shard_pid_start_offsets[shard_index];
        let local_pid = pid - shard_start_pid;
        self.doc_emb_shards[shard_index].get_embeddings(local_pid)
    }
}

/// Returns the maximum value in an array.
/// The return value is unspecified if any elements in the array can't be compared.
/// Panics if the array is empty.
fn array_max<A, S, D>(array: &ArrayBase<S, D>) -> A
where
    S: ndarray::Data<Elem = A>,
    D: Dimension,
    A: PartialOrd + Copy,
{
    array
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
}
