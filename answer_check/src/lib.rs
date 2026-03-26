use pyo3::prelude::*;

/// This is the Rust function we will call from Python.
#[pyfunction]
fn calculate_similarity(ground_truth: &str, generated_answer: &str) -> PyResult<f32> {
    println!("Rust received: Truth='{}', Gen='{}'", ground_truth, generated_answer);
    
    let dummy_score = 0.95;
    Ok(dummy_score)
}

/// A Python module implemented in Rust.
#[pymodule]
fn answer_check(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_similarity, m)?)?;
    Ok(())
}