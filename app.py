# app.py - Fixed Streamlit app for GPT-2 Fine-tuned Model
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import traceback

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "naeaeaem/gpt2-finetuned"  # Your HuggingFace repo

st.set_page_config(page_title="Pseudocode → C++", page_icon="🐍", layout="wide")

# -------------------------
# Model Loading
# -------------------------
@st.cache_resource
def load_model_and_tokenizer(model_id: str):
    """
    Load fine-tuned GPT-2 model with LoRA adapter
    """
    debug_info = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Step 1: Load base GPT-2 model
        debug_info.append("Loading base GPT-2 model...")
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        debug_info.append("✓ Base model loaded")
        
        # Step 2: Load tokenizer from your repo (or fallback to base)
        debug_info.append(f"Loading tokenizer from {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            debug_info.append("✓ Tokenizer loaded from repo")
        except Exception as e:
            debug_info.append(f"⚠ Tokenizer from repo failed: {str(e)[:100]}")
            debug_info.append("Loading base GPT-2 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            debug_info.append("✓ Base tokenizer loaded")
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Step 3: Resize embeddings if tokenizer has special tokens
        vocab_size = len(tokenizer)
        if vocab_size != base_model.config.vocab_size:
            debug_info.append(f"Resizing embeddings: {base_model.config.vocab_size} → {vocab_size}")
            base_model.resize_token_embeddings(vocab_size)
            debug_info.append("✓ Embeddings resized")
        
        # Step 4: Load LoRA adapter
        debug_info.append(f"Loading LoRA adapter from {model_id}...")
        model = PeftModel.from_pretrained(base_model, model_id)
        debug_info.append("✓ LoRA adapter loaded")
        
        # Step 5: Move to device
        model = model.to(device)
        model.eval()
        debug_info.append(f"✓ Model ready on {device.upper()}")
        
        return model, tokenizer, device, debug_info, None
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
        debug_info.append(f"❌ {error_msg}")
        return None, None, None, debug_info, error_msg

# -------------------------
# Code Generation
# -------------------------
def generate_code(model, tokenizer, device, pseudo, max_new_tokens=120, num_beams=5):
    """
    Generate C++ code from pseudocode
    """
    SPECIAL_PSEUDO = "<|pseudo|>"
    SPECIAL_CODE = "<|code|>"
    
    # Build prompt
    prompt = f"{SPECIAL_PSEUDO}\n{pseudo.strip()}\n{SPECIAL_CODE}\n"
    
    # Tokenize
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=450
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract code after SPECIAL_CODE
    if SPECIAL_CODE in generated_text:
        code = generated_text.split(SPECIAL_CODE, 1)[1]
    else:
        code = generated_text[len(prompt):]
    
    # Stop at special tokens
    for stop in [SPECIAL_PSEUDO, SPECIAL_CODE, "<|endoftext|>"]:
        if stop in code:
            code = code.split(stop)[0]
            break
    
    return code.strip()

# -------------------------
# UI
# -------------------------
st.title("🐍 Pseudocode → C++ Code Generator")
st.markdown("**Fine-tuned GPT-2 model for converting pseudocode to C++ code**")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown(f"**Model:** `{MODEL_ID}`")

max_tokens = st.sidebar.slider("Max New Tokens", 50, 200, 120, 10)
num_beams = st.sidebar.slider("Beam Size", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")

# Load model
with st.spinner("🔄 Loading model... (first load takes ~30s)"):
    model, tokenizer, device, debug_info, error = load_model_and_tokenizer(MODEL_ID)

# Show debug info in sidebar
with st.sidebar.expander("🔍 Loader Traces", expanded=False):
    for info in debug_info:
        st.text(info)

# Check if model loaded successfully
if model is None or error:
    st.error("❌ Failed to load model!")
    if error:
        st.error(error)
    st.stop()

st.sidebar.success(f"✅ Model ready ({device.upper()})")
st.sidebar.info(f"Vocabulary: {len(tokenizer)} tokens")

# Main content
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Pseudocode")
    
    # Example pseudocodes
    examples = {
        "Simple I/O": "read integer n\nprint n",
        "Conditional": "read integer n\nif n greater than 0\n  print n",
        "Loop": "for i from 1 to 10\n  print i",
        "Addition": "read integer x\nread integer y\nprint x plus y",
        "Nested Loop": "read integer n\nfor i from 1 to n\n  if i modulo 2 equals 0\n    print i"
    }
    
    selected_example = st.selectbox("Choose an example:", list(examples.keys()))
    default_pseudo = examples[selected_example]
    
    pseudo = st.text_area(
        "Enter pseudocode:", 
        value=default_pseudo, 
        height=250,
        help="Enter your pseudocode line by line"
    )
    
    generate_btn = st.button("🚀 Generate C++ Code", type="primary", use_container_width=True)

with col2:
    st.subheader("💻 Generated C++ Code")
    
    if generate_btn:
        if not pseudo.strip():
            st.warning("⚠️ Please enter pseudocode first!")
        else:
            with st.spinner("Generating code..."):
                try:
                    code = generate_code(
                        model, 
                        tokenizer, 
                        device, 
                        pseudo, 
                        max_tokens, 
                        num_beams
                    )
                    
                    if code:
                        st.code(code, language="cpp")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download C++ Code",
                            data=code,
                            file_name="generated_code.cpp",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ Generated code is empty. Try adjusting parameters.")
                        
                except Exception as e:
                    st.error(f"❌ Generation error: {str(e)}")
                    with st.expander("See error details"):
                        st.code(traceback.format_exc())
    else:
        st.info("👆 Click 'Generate C++ Code' to see the output")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with 🤗 Transformers + PEFT + Streamlit</p>
    <p>Model: GPT-2 fine-tuned on SPOC dataset (Pseudocode → C++)</p>
</div>
""", unsafe_allow_html=True)

# Additional info in expander
with st.expander("ℹ️ About this model"):
    st.markdown("""
    ### Model Details
    - **Base Model:** GPT-2
    - **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
    - **Dataset:** SPOC (Pseudocode to Code)
    - **Task:** Convert structured pseudocode to C++ code
    
    ### Evaluation Metrics
    - **BLEU Score:** 13.93
    - **CodeBLEU:** 0.41
    - **Code Quality:** 64%
    - **Success Rate:** 100%
    
    ### Usage Tips
    - Write clear, structured pseudocode
    - Use simple constructs (if, for, while)
    - Specify data types (integer, string, etc.)
    - Keep pseudocode concise
    """)
