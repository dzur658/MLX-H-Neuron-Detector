use pyo3::prelude::*;
use std::path::{Path, PathBuf};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use tokenizers::Tokenizer;
use ndarray::Array2;

const HF_BASE_URL: &str = "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main";
const EMBEDDING_DIM: usize = 384;

// ---------------------------------------------------------
// Phase 1: The Class Definition
// ---------------------------------------------------------
#[pyclass]
pub struct VibeChecker {
    session: Session,
    tokenizer: Tokenizer,
}

#[pymethods]
impl VibeChecker {
    
    /// This runs exactly once when Streamlit calls `VibeChecker()`
    #[new]
    fn new() -> PyResult<Self> {
        let model_dir = PathBuf::from(".models");
        std::fs::create_dir_all(&model_dir)?;

        // Detect Apple Silicon (aarch64) vs Intel (x86_64)
        let model_filename = if cfg!(target_arch = "aarch64") {
            "model_qint8_arm64.onnx"
        } else {
            "model_quint8_avx2.onnx"
        };

        let model_path = model_dir.join(model_filename);
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Phase 2: Auto-Download
        // If the user doesn't have the 40MB model yet, fetch it automatically
        if !model_path.exists() {
            println!("Rust: Downloading MiniLM ONNX model (this only happens once)...");
            download_file(&format!("{}/onnx/{}", HF_BASE_URL, model_filename), &model_path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }
        if !tokenizer_path.exists() {
            println!("Rust: Downloading Tokenizer...");
            download_file(&format!("{}/tokenizer.json", HF_BASE_URL), &tokenizer_path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }

        // Initialize the Hugging Face Tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Initialize the ONNX Engine (Maximum optimization for CPU)
        let session = Session::builder()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        println!("Rust: VibeChecker Engine Armed and Ready.");
        Ok(VibeChecker { session, tokenizer })
    }

    /// The function Python calls to get the final score
    fn calculate_similarity(&mut self, truth: &str, generated_answer: &str) -> PyResult<f32> {
        // Embed both strings into 384-dimensional vectors
        let vec_truth = self.embed_text(truth)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let vec_gen = self.embed_text(generated_answer)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Phase 4: The Math (Cosine Similarity)
        let mut dot_product = 0.0;
        let mut norm_truth = 0.0;
        let mut norm_gen = 0.0;

        for i in 0..EMBEDDING_DIM {
            dot_product += vec_truth[i] * vec_gen[i];
            norm_truth += vec_truth[i] * vec_truth[i];
            norm_gen += vec_gen[i] * vec_gen[i];
        }

        if norm_truth == 0.0 || norm_gen == 0.0 {
            return Ok(0.0);
        }

        let similarity = dot_product / (norm_truth.sqrt() * norm_gen.sqrt());
        Ok(similarity)
    }
}

// ---------------------------------------------------------
// Phase 3: The Brain (Forward Pass & Mean Pooling)
// This is a private Rust-only method, Python can't see it.
// ---------------------------------------------------------
impl VibeChecker {
    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, String> {
        // 1. Tokenize the text
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        
        let seq_len = input_ids.len();

        // 2. Convert to ndarray formats that ONNX expects [batch_size=1, sequence_length]
        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids).unwrap();
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask.clone()).unwrap();
        let token_type_ids_arr = Array2::from_shape_vec((1, seq_len), token_type_ids).unwrap();

        // convert to ONNX Value containers to satisfy requirements for inputs
        let input_ids_val = Value::from_array(input_ids_arr).unwrap();
        let attention_mask_val = Value::from_array(attention_mask_arr).unwrap();
        let token_type_ids_val = Value::from_array(token_type_ids_arr).unwrap();

        // 3. Fire the ONNX session
        let outputs = self.session.run(ort::inputs! {
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        })
        .map_err(|e| format!("ONNX inference failed: {}", e))?;

        // 4. Extract the raw tensor from the last hidden state
        let output_tensor = outputs["last_hidden_state"].try_extract_array::<f32>()
            .map_err(|e| format!("Failed to extract tensor: {}", e))?;

        // convert tensor to 3d array
        let output_view = output_tensor.into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| format!("Failed to convert tensor to 3D array: {}", e))?;
        
        // 5. Mean Pooling (The Sentence Transformer Magic)
        // We compress the multi-token matrix into a single 384-length vector, 
        // ignoring padding tokens using the attention mask.
        let mut pooled = vec![0.0f32; EMBEDDING_DIM];
        let mut mask_sum = 0.0f32;

        for token_idx in 0..seq_len {
            let mask_val = attention_mask[token_idx] as f32;
            mask_sum += mask_val;
            
            // let offset = token_idx * EMBEDDING_DIM;
            for dim in 0..EMBEDDING_DIM {
                pooled[dim] += output_view[[0, token_idx, dim]] * mask_val;
            }
        }

        if mask_sum > 0.0 {
            for dim in 0..EMBEDDING_DIM {
                pooled[dim] /= mask_sum;
            }
        }

        Ok(pooled)
    }
}

// ---------------------------------------------------------
// Helper: File Downloader
// ---------------------------------------------------------
fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    let agent = ureq::Agent::new();
    let response = agent.get(url).call().map_err(|e| e.to_string())?;
    
    let mut file = std::fs::File::create(dest).map_err(|e| e.to_string())?;
    std::io::copy(&mut response.into_reader(), &mut file).map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------
// Python Module Registration
// ---------------------------------------------------------
#[pymodule]
fn answer_check(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VibeChecker>()?;
    Ok(())
}