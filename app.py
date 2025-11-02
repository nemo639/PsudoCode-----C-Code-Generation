# app.py - Fixed with automatic config cleaning
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import traceback
import json
from huggingface_hub import hf_hub_download

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "naeaeaem/gpt2-finetuned"

st.set_page_config(page_title="Pseudocode → C++", page_icon="🐍", layout="wide")

# -------------------------
# Model Loading with Config Fix
# -------------------------
@st.cache_resource
def load_model_and_tokenizer(model_id: str):
    """
    Load fine-tuned GPT-2 model with LoRA adapter (with automatic config fixing)
    """
    debug_info = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Step 1: Load base GPT-2 model
        debug_info.append("Loading base GPT-2 model...")
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        debug_info.append("✓ Base model loaded")
        
        # Step 2: Load tokenizer
        debug_info.append(f"Loading tokenizer from {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            debug_info.append("✓ Tokenizer loaded from repo")
        except Exception as e:
            debug_info.append(f"⚠ Tokenizer from repo failed, using base")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            debug_info.append("✓ Base tokenizer loaded")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Step 3: Resize embeddings
        vocab_size = len(tokenizer)
        if vocab_size != base_model.config.vocab_size:
            debug_info.append(f"Resizing embeddings: {base_model.config.vocab_size} → {vocab_size}")
            base_model.resize_token_embeddings(vocab_size)
            debug_info.append("✓ Embeddings resized")
        
        # Step 4: Load LoRA with config cleaning
        debug_info.append(f"Loading LoRA adapter from {model_id}...")
        
        try:
            # Try direct loading first
            model = PeftModel.from_pretrained(base_model, model_id)
            debug_info.append("✓ LoRA adapter loaded directly")
        except TypeError as e:
            error_msg = str(e)
            debug_info.append(f"⚠ Direct load failed: {error_msg[:100]}")
            debug_info.append("Attempting to fix config...")
            
            # Download and clean config
            config_path = hf_hub_download(repo_id=model_id, filename="adapter_config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            debug_info.append(f"Original config keys: {len(config_dict)}")
            
            # Keep only peft 0.7.1 compatible parameters
            compatible_params = {
                'auto_mapping', 'base_model_name_or_path', 'bias', 
                'fan_in_fan_out', 'inference_mode', 'init_lora_weights',
                'lora_alpha', 'lora_dropout', 'modules_to_save', 'peft_type',
                'r', 'target_modules', 'task_type', 'revision'
            }
            
            cleaned_config = {k: v for k, v in config_dict.items() if k in compatible_params}
            debug_info.append(f"Cleaned config keys: {len(cleaned_config)}")
            
            # Create LoraConfig with cleaned parameters
            lora_config = LoraConfig(**cleaned_config)
            
            # Load model with cleaned config
            model = PeftModel.from_pretrained(
                base_model, 
                model_id,
                config=lora_config,
                ignore_mismatched_sizes=True
            )
            debug_info.append("✓ LoRA adapter loaded with cleaned config")
        
        # Step 5: Move to device
        model = model.to(device)
        model.eval()
        debug_info.append(f"✓ Model ready on {device.upper()}")
        
        return model, tokenizer, device, debug_info, None
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        debug_info.append(f"❌ {error_msg}")
        return None, None, None, debug_info, error_msg

# -------------------------
# Code Generation
# -------------------------
def generate_code(model, tokenizer, device, pseudo, max_new_tokens=120, num_beams=5):
    """Generate C++ code from pseudocode"""
    SPECIAL_PSEUDO = "<|pseudo|>"
    SPECIAL_CODE = "<|code|>"
    
    prompt = f"{SPECIAL_PSEUDO}\n{pseudo.strip()}\n{SPECIAL_CODE}\n"
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=450
    ).to(device)
    
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
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if SPECIAL_CODE in generated_text:
        code = generated_text.split(SPECIAL_CODE, 1)[1]
    else:
        code = generated_text[len(prompt):]
    
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

if model is None or error:
    st.error("❌ Failed to load model!")
    if error:
        st.error(error)
    st.info("💡 Try: Clear cache and retry (Settings → Clear cache)")
    st.stop()

st.sidebar.success(f"✅ Model ready ({device.upper()})")
st.sidebar.info(f"Vocabulary: {len(tokenizer)} tokens")

# Main content
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Pseudocode")
    
    examples = {
        "Simple I/O": "read integer n\nprint n",
        "Conditional": "read integer n\nif n greater than 0\n  print n",
        "Loop": "for i from 1 to 10\n  print i",
        "Addition": "read integer x\nread integer y\nprint x plus y",
        "Nested": "read integer n\nfor i from 1 to n\n  if i modulo 2 equals 0\n    print i"
    }
    
    selected = st.selectbox("Choose example:", list(examples.keys()))
    
    pseudo = st.text_area(
        "Enter pseudocode:", 
        value=examples[selected], 
        height=250
    )
    
    generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

with col2:
    st.subheader("💻 Generated C++")
    
    if generate_btn:
        if not pseudo.strip():
            st.warning("⚠️ Please enter pseudocode!")
        else:
            with st.spinner("Generating..."):
                try:
                    code = generate_code(model, tokenizer, device, pseudo, max_tokens, num_beams)
                    
                    if code:
                        st.code(code, language="cpp")
                        st.download_button(
                            "📥 Download",
                            code,
                            "generated.cpp",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ Empty output")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Click 'Generate' to see output")

# Footer
st.markdown("---")
st.caption("Built with 🤗 Transformers + PEFT + Streamlit")
