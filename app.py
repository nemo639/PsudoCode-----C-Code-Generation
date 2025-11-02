# app.py - FINAL WORKING VERSION
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
# Model Loading - FIXED EMBEDDINGS
# -------------------------
@st.cache_resource
def load_model_and_tokenizer(model_id: str):
    """
    Load fine-tuned GPT-2 with LoRA - FIXED VERSION
    """
    debug_info = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Step 1: Load base GPT-2
        debug_info.append("Loading base GPT-2...")
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        debug_info.append("✓ Base model loaded (vocab: 50257)")
        
        # Step 2: Load tokenizer - ALWAYS load from repo first to get special tokens!
        debug_info.append(f"Loading tokenizer...")
        
        # Method 1: Try loading tokenizer from repo
        tokenizer = None
        try:
            # Download tokenizer files to check vocabulary
            vocab_file = hf_hub_download(repo_id=model_id, filename="vocab.json")
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            vocab_size = len(vocab)
            debug_info.append(f"✓ Repo tokenizer vocab size: {vocab_size}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            debug_info.append(f"✓ Tokenizer loaded from repo")
        except Exception as e:
            debug_info.append(f"⚠ Repo tokenizer failed: {str(e)[:100]}")
            debug_info.append("Creating tokenizer with special tokens...")
            
            # Fallback: Create tokenizer with special tokens
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # Add the 2 special tokens that were used during training
            special_tokens = {
                "additional_special_tokens": ["<|pseudo|>", "<|code|>"]
            }
            tokenizer.add_special_tokens(special_tokens)
            vocab_size = len(tokenizer)
            debug_info.append(f"✓ Created tokenizer with special tokens (vocab: {vocab_size})")
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Step 3: Resize base model BEFORE loading LoRA
        current_vocab = len(tokenizer)
        if current_vocab != base_model.config.vocab_size:
            debug_info.append(f"Resizing embeddings: {base_model.config.vocab_size} → {current_vocab}")
            base_model.resize_token_embeddings(current_vocab)
            debug_info.append("✓ Embeddings resized")
        
        # Step 4: Load LoRA adapter with config cleaning
        debug_info.append(f"Loading LoRA adapter...")
        
        try:
            # Try direct loading
            model = PeftModel.from_pretrained(base_model, model_id)
            debug_info.append("✓ LoRA loaded directly")
        except (TypeError, RuntimeError) as e:
            debug_info.append(f"⚠ Direct load failed, cleaning config...")
            
            # Download and clean config
            config_path = hf_hub_download(repo_id=model_id, filename="adapter_config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Compatible parameters for peft 0.7.1
            compatible = {
                'auto_mapping', 'base_model_name_or_path', 'bias',
                'fan_in_fan_out', 'inference_mode', 'init_lora_weights',
                'lora_alpha', 'lora_dropout', 'modules_to_save', 
                'peft_type', 'r', 'target_modules', 'task_type', 'revision'
            }
            
            cleaned = {k: v for k, v in config_dict.items() if k in compatible}
            lora_config = LoraConfig(**cleaned)
            
            model = PeftModel.from_pretrained(base_model, model_id, config=lora_config)
            debug_info.append("✓ LoRA loaded with cleaned config")
        
        # Step 5: Move to device
        model = model.to(device)
        model.eval()
        debug_info.append(f"✓ Model ready on {device.upper()}")
        debug_info.append(f"✓ Final vocab size: {len(tokenizer)}")
        
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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=450).to(device)
    
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
st.markdown("**Fine-tuned GPT-2 with LoRA for pseudocode to C++ translation**")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown(f"**Model:** `{MODEL_ID}`")

max_tokens = st.sidebar.slider("Max Tokens", 50, 200, 120, 10)
num_beams = st.sidebar.slider("Beam Size", 1, 10, 5)

st.sidebar.markdown("---")

# Load model
with st.spinner("🔄 Loading model (first load ~30s)..."):
    model, tokenizer, device, debug_info, error = load_model_and_tokenizer(MODEL_ID)

# Debug info
with st.sidebar.expander("🔍 Loader Details"):
    for info in debug_info:
        st.text(info)

# Check loading status
if model is None or error:
    st.error("❌ Failed to load model!")
    if error:
        with st.expander("Error Details"):
            st.code(error)
    st.info("💡 Try: Manage app → Reboot app")
    st.stop()

st.sidebar.success(f"✅ Ready ({device.upper()})")
st.sidebar.info(f"Vocab: {len(tokenizer)} tokens")

# Main UI
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Pseudocode Input")
    
    examples = {
        "Hello World": "read integer n\nprint n",
        "Conditional": "read integer n\nif n greater than 0\n  print n",
        "For Loop": "for i from 1 to 10\n  print i",
        "Addition": "read integer x\nread integer y\nprint x plus y",
        "Even Numbers": "read integer n\nfor i from 1 to n\n  if i modulo 2 equals 0\n    print i"
    }
    
    example_name = st.selectbox("Examples:", list(examples.keys()))
    pseudo = st.text_area("Enter pseudocode:", examples[example_name], height=250)
    
    gen_btn = st.button("🚀 Generate C++", type="primary", use_container_width=True)

with col2:
    st.subheader("💻 Generated Code")
    
    if gen_btn:
        if not pseudo.strip():
            st.warning("⚠️ Enter pseudocode first")
        else:
            with st.spinner("Generating..."):
                try:
                    code = generate_code(model, tokenizer, device, pseudo, max_tokens, num_beams)
                    
                    if code:
                        st.code(code, language="cpp")
                        st.download_button(
                            "📥 Download Code",
                            code,
                            "generated.cpp",
                            "text/plain",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ Empty output generated")
                        
                except Exception as e:
                    st.error(f"❌ Generation error: {str(e)}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())
    else:
        st.info("👆 Click 'Generate C++' to see output")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with 🤗 Transformers + PEFT + Streamlit<br>
    GPT-2 fine-tuned on SPOC dataset (BLEU: 13.93 | Quality: 64%)
</div>
""", unsafe_allow_html=True)
