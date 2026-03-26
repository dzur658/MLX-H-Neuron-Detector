import streamlit as st
import answer_check  # This is your compiled Rust module!

st.set_page_config(page_title="H-Neuron Calibrator", layout="wide")

st.title("🧠 Local H-Neuron Calibrator")
st.markdown("This dashboard extracts activations from MLX and uses a Rust backend to verify factual accuracy.")

st.divider()

# A simple UI to test our Rust pipeline
st.subheader("Test the Rust Vibe Check")
col1, col2 = st.columns(2)

with col1:
    truth = st.text_input("Ground Truth", value="March 1 1981")
with col2:
    gen = st.text_input("Generated Answer", value="March, 1, 1981")

if st.button("Calculate Semantic Similarity"):
    # Here we are calling the blazing fast Rust function directly from Python
    score = answer_check.calculate_similarity(truth, gen)
    
    if score >= 0.90:
        st.success(f"Match! Score: {score:.4f}")
    else:
        st.error(f"Hallucination! Score: {score:.4f}")

st.divider()

st.info("Next up: Hooking this into the MLX forward pass...")