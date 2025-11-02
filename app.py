# app.py — robust, full-file Streamlit app that tolerates missing huggingface_hub
import streamlit as st
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Try to import HF hub utilities (not guaranteed in some deploy environments)
try:
    from huggingface_hub import hf_hub_download, HfApi, RepositoryNotFoundError, HfHubHTTPError
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None
    HfApi = None
    RepositoryNotFoundError = Exception
    HfHubHTTPError = Exception

# -------------------------
# Configuration
# -------------------------
# Replace with your HF repo if different
MODEL_ID = "naeaeaem/gpt2-finetuned"

st.set_page_config(page_title="Pseudocode → C++", page_icon="🐍", layout="wide")
st.markdown("# 🐍 Pseudocode → C++ (Robust Loader)")

# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def try_load_adapter_config(model_id: str):
    """Return parsed adapter_config.json if available (requires huggingface_hub)."""
    if not HF_HUB_AVAILABLE:
        return None, "huggingface_hub not available"
    try:
        cfg_path = hf_hub_download(repo_id=model_id, filename="adapter_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg, None
    except RepositoryNotFoundError as e:
        return None, f"repo not found: {e}"
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_model_and_tokenizer(model_id: str):
    """
    Robust loading strategy (returns model, tokenizer, device, debug_info):
     1) If adapter_config.json exists in repo: load base model then PeftModel.from_pretrained(adapter_repo).
     2) Try loading repo as full model (AutoModelForCausalLM.from_pretrained).
     3) Fallback to base 'gpt2'.
    """
    debug = {"steps": []}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Try adapter_config path (preferred)
    cfg, err = try_load_adapter_config(model_id)
    if cfg:
        debug["steps"].append("Found adapter_config.json in repo.")
        base_name = cfg.get("base_model_name_or_path") or cfg.get("base_model") or "gpt2"
        debug["steps"].append(f"Adapter base model inferred: {base_name}")

        # tokenizer: prefer repo tokenizer files
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            debug["steps"].append("Tokenizer loaded from repo.")
        except Exception as e:
            debug["steps"].append(f"Tokenizer from repo failed: {e}; trying base tokenizer.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
                debug["steps"].append(f"Tokenizer loaded from base '{base_name}'.")
            except Exception as e2:
                debug["steps"].append(f"Tokenizer fallback to 'gpt2' (because base '{base_name}' failed): {e2}")
                tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

        # base model + attach PEFT adapter
        try:
            base_model = AutoModelForCausalLM.from_pretrained(base_name)
            model = PeftModel.from_pretrained(base_model, model_id)
            model = model.to(device).eval()
            debug["steps"].append("Loaded base model and attached PEFT adapter successfully.")
            return model, tokenizer, device, debug
        except Exception as e:
            debug["steps"].append(f"PEFT attach failed: {e}")

    else:
        debug["steps"].append(f"No adapter_config.json: {err or 'not present'}")

    # 2) Try loading repo as a full model
    try:
        debug["steps"].append("Attempting to load repository as full model...")
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            debug["steps"].append("Tokenizer loaded from repo for full-model attempt.")
        except Exception as e:
            debug["steps"].append(f"Tokenizer from repo failed: {e}; falling back to 'gpt2'.")
            tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device).eval()
        debug["steps"].append("Loaded full model from repo successfully.")
        return model, tokenizer, device, debug
    except Exception as e:
        debug["steps"].append(f"Full-model load attempt failed: {e}")

    # 3) Final fallback: load base GPT-2
    try:
        debug["steps"].append("Falling back to base 'gpt2' model.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
        debug["steps"].append("Loaded base 'gpt2' model.")
        return model, tokenizer, device, debug
    except Exception as e:
        debug["steps"].append(f"Base fallback failed: {e}")
        return None, None, None, debug

# -------------------------
# Generation
# -------------------------
def generate_code(model, tokenizer, device, pseudo, max_new_tokens=120, num_beams=5):
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
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if SPECIAL_CODE in gen_text:
        result = gen_text.split(SPECIAL_CODE, 1)[1]
    else:
        result = gen_text[len(prompt):]
    for stop in [SPECIAL_PSEUDO, SPECIAL_CODE, "<|endoftext|>"]:
        if stop in result:
            result = result.split(stop)[0]
            break
    return result.strip()

# -------------------------
# UI
# -------------------------
st.sidebar.title("⚙️ Settings & Diagnostics")
st.sidebar.markdown(f"**MODEL_ID:** `{MODEL_ID}`")
if not HF_HUB_AVAILABLE:
    st.sidebar.warning("`huggingface_hub` not available in environment — repo inspection may be limited. Add `huggingface_hub` to requirements.txt and redeploy for full diagnostics.")

with st.spinner("Loading model (may take ~30s on first run)..."):
    model, tokenizer, device, debug = load_model_and_tokenizer(MODEL_ID)

# Show debug traces (useful to paste here if something fails)
st.subheader("Loader traces")
if debug and debug.get("steps"):
    for step in debug["steps"]:
        st.text("• " + str(step))

if model is None or tokenizer is None:
    st.error("❌ Model or tokenizer failed to load. See loader traces above.")
    st.stop()

st.success(f"✅ Model ready (running on {device.upper()})")

# Main UI
st.markdown("---")
st.header("Generate C++ from pseudocode")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Pseudocode")
    default_pseudo = "read integer n\nfor i from 1 to n\n  print i"
    pseudo = st.text_area("Enter pseudocode:", value=default_pseudo, height=300)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 200, 120)
    num_beams = st.sidebar.slider("Beam Size", 1, 10, 5)

with col2:
    st.subheader("💻 Generated C++")
    if st.button("🚀 Generate Code"):
        if not pseudo.strip():
            st.warning("Please enter pseudocode first.")
        else:
            with st.spinner("Generating..."):
                try:
                    code = generate_code(model, tokenizer, device, pseudo, max_tokens, num_beams)
                    st.code(code, language="cpp")
                    st.download_button("📥 Download code", code, file_name="generated_code.cpp", mime="text/plain")
                except Exception as e:
                    st.error(f"Generation error: {e}")

st.markdown("---")
st.caption("Built with Transformers + PEFT + Streamlit")
