import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Page configuration
st.set_page_config(
    page_title="Pseudocode to C++ Generator",
    page_icon="🐍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    """Load the fine-tuned GPT-2 model with LoRA"""
    try:
        with st.spinner("🔄 Loading model... This may take a minute..."):
            # IMPORTANT: Replace with your Hugging Face model ID
            MODEL_ID = "naeaeaem/gpt2-finetuned"
            
            base_model = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            base_model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base_model, MODEL_ID)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
            return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure your model is uploaded to Hugging Face and MODEL_ID is correct.")
        return None, None, None

def generate_code(model, tokenizer, device, pseudo, max_new_tokens=120, num_beams=5):
    """Generate C++ code from pseudocode"""
    SPECIAL_PSEUDO = "<|pseudo|>"
    SPECIAL_CODE = "<|code|>"
    
    prompt = f"{SPECIAL_PSEUDO}\n{pseudo.strip()}\n{SPECIAL_CODE}\n"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=450, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=1,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.3,
            length_penalty=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if SPECIAL_CODE in generated_text:
        generated = generated_text.split(SPECIAL_CODE, 1)[1]
    else:
        generated = generated_text[len(prompt):]
    
    for stop in [SPECIAL_PSEUDO, SPECIAL_CODE, '<|endoftext|>']:
        if stop in generated:
            generated = generated.split(stop)[0]
            break
    
    lines = generated.strip().split('\n')
    cleaned_lines = []
    seen_lines = set()
    brace_count = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped and cleaned_lines:
            continue
        brace_count += stripped.count('{') - stripped.count('}')
        if stripped in seen_lines and len(stripped) > 5:
            continue
        cleaned_lines.append(line.rstrip())
        seen_lines.add(stripped)
        if brace_count == 0 and cleaned_lines and stripped == '}':
            break
    
    return '\n'.join(cleaned_lines).strip()

def main():
    st.markdown('<h1 class="main-header">🐍 Pseudocode to C++ Code Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This app converts pseudocode into C++ code using a fine-tuned GPT-2 model.
    Enter your pseudocode and click **Generate Code**!
    """)
    
    model, tokenizer, device = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    max_tokens = st.sidebar.slider("Max Tokens", 50, 200, 120, 10)
    num_beams = st.sidebar.slider("Beam Size", 1, 10, 5, 1)
    st.sidebar.info(f"Running on: **{device.upper()}**")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Examples")
    
    examples = {
        "Simple I/O": "read integer n\nprint n",
        "Addition": "read integer x\nread integer y\nprint x plus y",
        "Conditional": "read integer n\nif n greater than 0\n  print n",
        "Loop": "for i from 1 to 10\n  print i",
        "Sum Loop": "read integer n\nset sum to 0\nfor i from 1 to n\n  add i to sum\nprint sum",
        "Even Numbers": "read integer n\nfor i from 1 to n\n  if i modulo 2 equals 0\n    print i"
    }
    
    selected_example = st.sidebar.selectbox("Choose example:", ["Custom"] + list(examples.keys()))
    
    # Main area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Pseudocode")
        default_pseudo = examples[selected_example] if selected_example != "Custom" else ""
        pseudocode = st.text_area(
            "Enter pseudocode:",
            value=default_pseudo,
            height=300,
            placeholder="Example:\nread integer n\nfor i from 1 to n\n  print i"
        )
        generate_button = st.button("🚀 Generate Code", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("💻 Generated C++ Code")
        if generate_button:
            if not pseudocode.strip():
                st.warning("⚠️ Please enter pseudocode first!")
            else:
                with st.spinner("⏳ Generating..."):
                    try:
                        generated_code = generate_code(model, tokenizer, device, pseudocode, max_tokens, num_beams)
                        st.code(generated_code, language="cpp")
                        st.download_button(
                            label="📥 Download Code",
                            data=generated_code,
                            file_name="generated_code.cpp",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👈 Enter pseudocode and click 'Generate Code'")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit | GPT-2 + LoRA on SPOC Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
