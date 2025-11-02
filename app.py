import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from huggingface_hub import hf_hub_download, HfApi, RepositoryNotFoundError, HfHubHTTPError
import json
import os

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "naeaeaem/gpt2-finetuned"  # <-- replace if needed

st.set_page_config(page_title="Pseudocode → C++ (robust loader)", page_icon="🐍", layout="wide")

st.markdown(
    """
    # 🐍 Pseudocode → C++
    Robust model loader for PEFT adapter and full model repos.
    """
)

# -------------------------
# Helpers: inspect repo (uses HfApi.list_repo_files for compatibility)
# -------------------------
@st.cache_resource
def repo_files(model_id: str):
    try:
        api = HfApi()
        files = api.list_repo_files(model_id)
        # list_repo_files returns list[str]
        return {"ok": True, "files": files}
    except RepositoryNotFoundError:
        return {"ok": False, "error": f"Repository '{model_id}' not found."}
    except HfHubHTTPError as e:
        return {"ok": False, "error": f"HfHubHTTPError: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Unknown error listing repo: {e}"}

# -------------------------
# Model loader (robust)
# -------------------------
@st.cache_resource
def load_model_and_tokenizer(model_id: str):
    """
    Attempts multiple strategies to load:
      1) PEFT adapter (read adapter_config.json -> load base -> attach adapter)
      2) Full model (AutoModelForCausalLM.from_pretrained(model_id))
      3) Base 'gpt2' fallback
    Returns (model, tokenizer, device, debug_info)
    """
    debug = {"traces": []}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Repo file listing
    repo_info = repo_files(model_id)
    debug["repo_info"] = repo_info

    # Attempt: if adapter_config.json present, use it to get base model
    try:
        debug["traces"].append("Checking for adapter_config.json in repo...")
        if repo_info.get("ok") and any("adapter_config.json" in fname for fname in repo_info.get("files", [])):
            debug["traces"].append("adapter_config.json detected in repo. Downloading...")
            cfg_path = hf_hub_download(repo_id=model_id, filename="adapter_config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_model_name = cfg.get("base_model_name_or_path") or cfg.get("base_model") or "gpt2"
            debug["traces"].append(f"adapter_config.json parsed. base_model_name_or_path = {base_model_name}")

            # tokenizer: prefer repo tokenizer files if present
            try:
                debug["traces"].append("Attempting to load tokenizer from repo...")
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                debug["traces"].append("Tokenizer loaded from repo.")
            except Exception as e_tok_repo:
                debug["traces"].append(f"Tokenizer from repo failed: {e_tok_repo}. Falling back to base tokenizer '{base_model_name}' or 'gpt2'.")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
                    debug["traces"].append(f"Tokenizer loaded from base_model_name '{base_model_name}'.")
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
                    debug["traces"].append("Tokenizer fallback to 'gpt2' succeeded.")

            # Load base model then attach PEFT adapter
            debug["traces"].append(f"Loading base model '{base_model_name}'...")
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            debug["traces"].append("Base model loaded. Attaching PEFT adapter from repo...")
            model = PeftModel.from_pretrained(base_model, model_id)
            model = model.to(device).eval()
            debug["traces"].append("PeftModel.from_pretrained succeeded.")
            return model, tokenizer, device, debug
        else:
            debug["traces"].append("No adapter_config.json detected in repo.")
    except Exception as e_adapter:
        debug["traces"].append(f"PEFT adapter path failed: {repr(e_adapter)}")

    # Attempt: load repo as a full model (AutoModelForCausalLM)
    try:
        debug["traces"].append("Attempting to load repo as a full model with AutoModelForCausalLM...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            debug["traces"].append("Tokenizer loaded from repo (full-model attempt).")
        except Exception as te:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            debug["traces"].append(f"Tokenizer load from repo failed ({te}); fallback to 'gpt2' tokenizer.")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device).eval()
        debug["traces"].append("AutoModelForCausalLM.from_pretrained(repo) succeeded.")
        return model, tokenizer, device, debug
    except Exception as e_full:
        debug["traces"].append(f"Full-model load attempt failed: {repr(e_full)}")

    # Final fallback: base gpt2
    try:
        debug["traces"].append("Final fallback: loading base 'gpt2' model & tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
        debug["traces"].append("Base 'gpt2' loaded successfully.")
        return model, tokenizer, device, debug
    except Exception as e_base:
        debug["traces"].append(f"Base fallback failed: {repr(e_base)}")
        return None, None, None, debug

# -------------------------
# Generation function
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
# UI / Main
# -------------------------
def main():
    st.sidebar.title("⚙️ Settings & Diagnostics")
    st.sidebar.write(f"MODEL_ID = `{MODEL_ID}` (edit top of file if needed)")

    # Load model & tokenizer (shows cached results if re-run)
    model, tokenizer, device, debug = load_model_and_tokenizer(MODEL_ID)

    # Show repo inspection
    st.subheader("Repository inspection")
    repo_info = debug.get("repo_info") if debug else None
    if repo_info:
        if repo_info.get("ok"):
            st.write("Files in repo (first 200):")
            st.write(repo_info.get("files")[:200])
            if any(name.lower().endswith(("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json", "pytorch_model.bin")) for name in repo_info.get("files", [])):
                st.success("Repository appears to contain adapter/model files.")
        else:
            st.error("Could not list repo files: " + repo_info.get("error", "unknown"))

    # Loader traces
    st.subheader("Loader traces (debug)")
    if debug and debug.get("traces"):
        for t in debug["traces"]:
            st.text("- " + t)

    # If no model / tokenizer
    if model is None or tokenizer is None:
        st.error("❌ Model/tokenizer failed to load. See loader traces above for details.")
        st.info(
            """
            Common fixes:
            - If repo contains a PEFT adapter, ensure `adapter_config.json` lists `base_model_name_or_path`.
            - If repo is adapter-only, keep tokenizer files in the repo (tokenizer.json/vocab/merges) or use `AutoTokenizer.from_pretrained('gpt2')`.
            - If you fine-tuned locally, export with `peft_model.save_pretrained(...)` and upload the folder to HF (use git-lfs for large files).
            - Ensure `peft`, `transformers`, `huggingface_hub`, and `safetensors` are in requirements.
            """
        )
        st.stop()

    st.success(f"✅ Model loaded and running on {device.upper()}")

    # Generation UI
    st.markdown("---")
    st.header("Generate C++ from pseudocode")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Input Pseudocode")
        default_pseudo = "read integer n\nfor i from 1 to n\n  print i"
        pseudo = st.text_area("Enter pseudocode:", value=default_pseudo, height=300)
        max_tokens = st.slider("Max Tokens", 50, 200, 120)
        num_beams = st.slider("Beam Size", 1, 10, 5)
        generate = st.button("🚀 Generate")

    with col2:
        st.subheader("💻 Generated C++")
        if generate:
            if not pseudo.strip():
                st.warning("Enter pseudocode first")
            else:
                with st.spinner("Generating..."):
                    try:
                        out = generate_code(model, tokenizer, device, pseudo, max_tokens, num_beams)
                        st.code(out, language="cpp")
                        st.download_button("📥 Download code", out, file_name="generated_code.cpp", mime="text/plain")
                    except Exception as e:
                        st.error(f"Generation error: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#888'>Built with Streamlit | GPT-2 + PEFT</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
