import streamlit as st
import answer_check

st.set_page_config(page_title="H-Neuron Calibrator", layout="wide")

st.title("🧠 Local H-Neuron Calibrator")

# Cache the Rust engine so it stays in memory between UI clicks!
@st.cache_resource
def load_vibe_checker():
    return answer_check.VibeChecker()

# Initialize it
checker = load_vibe_checker()

st.divider()
st.subheader("Test the Rust Vibe Check")
col1, col2 = st.columns(2)

with col1:
    truth = st.text_input("Ground Truth", value="March 1 1981")
with col2:
    gen = st.text_input("Generated Answer", value="March, 1, 1981")

if st.button("Calculate Semantic Similarity"):
    # Call the method on our cached Rust object
    score = checker.calculate_similarity(truth, gen)
    
    if score >= 0.90:
        st.success(f"Match! Score: {score:.4f}")
    else:
        st.error(f"Hallucination! Score: {score:.4f}")