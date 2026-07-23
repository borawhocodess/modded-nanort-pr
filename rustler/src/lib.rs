use pyo3::prelude::*;

mod common;
pub mod fly;
pub mod graph;

#[pymodule]
fn rustler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fly::Sampler>()?;
    m.add_class::<graph::GraphInspector>()?;

    Ok(())
}
