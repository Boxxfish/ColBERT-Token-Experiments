pub mod retrieve_df_builder;
pub mod scorer;

use pyo3::prelude::*;

/// ColBERT helper functionality implemented in Rust.
#[pymodule]
fn colbert_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<retrieve_df_builder::RetrieveDFBuilder>()?;
    m.add_class::<scorer::Scorer>()?;
    Ok(())
}
