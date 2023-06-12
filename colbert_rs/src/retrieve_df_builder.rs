use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct RetrieveDFBuilder {
    rows: Vec<Py<PyList>>,
}

#[pymethods]
impl RetrieveDFBuilder {
    #[new]
    pub fn empty() -> Self {
        RetrieveDFBuilder { rows: Vec::new() }
    }

    /// Appends a [qid, query, pid, ids, q_cpu] row for each pid in pids.
    pub fn append_rows(
        &mut self,
        py: Python,
        qid: &PyAny,
        query: &PyAny,
        pids: &PyList,
        ids: &PyAny,
        q_cpu: &PyAny,
    ) {
        self.rows.extend(
            pids.iter()
                .map(|pid| PyList::new(py, [qid, query, pid, ids, q_cpu]).into()),
        );
    }

    /// Converts the builder into a Python list of lists.
    pub fn to_list<'py>(&self, py: Python<'py>) -> &'py PyList {
        PyList::new(py, &self.rows)
    }
}
