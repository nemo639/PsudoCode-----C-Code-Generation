import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from huggingface_hub import hf_hub_list, hf_hub_download, RepositoryNotFoundError, HfHubHTTPError

st.set_page_config(page_title="Pseudocode → C++ (debug loader)", page_icon="🐍", layout="wide")

st.markdown("# 🐍 Pseudocode → C++ (robust loader with diagnostics)")

MODEL_ID = "naeaeaem/gpt2-finetuned"  # <-- replace if needed


@st.cache_resource
def inspect_repo(model_id):
    """Return list of files in the HF repo (or error message)."""
    try:
        files = hf_hub_list(model_id)
        return {"ok": True, "files": [f.rfilename for f in files]}
    except RepositoryNotFoundError:
        return {"ok": False, "error": f"Repository `{model_id}` not found (RepositoryNotFoundError)"}
    except HfHubHTTPError as e:
        return {"ok": False, "error": f"HfHubHTTPError: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Unknown error listing repo: {e}"}


@st.cache_resource
def load_model_and_tokenizer(model_id):
    """
    Tries several strategies to load the model + tokenizer.
    Returns tuple (model, tokenizer, device, debug_info)
    debug_info is a dict with tracing info and repo file list.
    """
    debug = {"traces": []}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_inspect = inspect_repo(model_id)
    debug["repo_inspect"] = repo_inspect

    # Attempt 1: Assume repo is a PEFT adapter and contains a PeftConfig
    try:
        debug["traces"].append("Attempt: PeftConfig.from_pretrained()")
        peft_conf = PeftConfig.from_pretrained(model_id)
        debug["traces"].append(f"PeftConfig found: base_model_name_or_path={peft_conf.base_model_name_or_path}")
        tokenizer = None
        # load tokenizer: try model_id then fallback to base_model_name_or_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            debug["traces"].append("Tokenizer loaded from model_id")
        except Exception as te:
            try:
                tokenizer = AutoTokenizer.from_pretrained(peft_conf.base_model_name_or_path)
                debug["traces"].append("Tokenizer loaded from base_model_name_or_path")
            except Exception as te2:
                debug["traces"].append(f"Tokenizer load fallback failed: {te2}")
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                debug["traces"].append("Tokenizer fallback to 'gpt2'")

        base_model = AutoModelForCausalLM.from_pretrained(peft_conf.base_model_name_or_path)
        debug["traces"].append(f"Base model loaded: {peft_conf.base_model_name_or_path}")
        model = PeftModel.from_pretrained(base_model, model_id)
        debug["traces"].append("PeftModel.from_pretrained succeeded")
        model = model.to(device).eval()
        return model, tokenizer, device, debug
    except Exception as e_peft:
        debug["traces"].append(f"Peft attempt failed: {repr(e_peft)}")

    # Attempt 2: Try loading the repo as a full AutoModel
    try:
        debug["traces"].append("Attempt: AutoModelForCausalLM.from_pretrained(MODEL_ID)")
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            debug["traces"].append("Tokenizer loaded from model_id (full model path)")
        except Exception as te:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            debug["traces"].append(f"Tokenizer load from model_id failed: {te}; fallback to 'gpt2'")

        model = AutoModelForCausalLM.from_pretrained(model_id)
        debug["traces"].append("AutoModelForCausalLM.from_pretrained succeeded")
        model = model.to(device).eval()
        return model, tokenizer, device, debug
    except Exception as e_full:
        debug["traces"].append(f"Full-model attempt failed: {repr(e_full)}")

    # Attempt 3: Load base GPT-2 (no adapter) and tell user to provide adapter
    try:
        debug["traces"].append("Attempt: load base 'gpt2' (no adapter).")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        base_model = base_model.to(device).eval()
        debug["traces"].append("Base gpt2 loaded")
        return base_model, tokenizer, device, debug
    except Exception as e_base:
        debug["traces"].append(f"Base model attempt failed: {repr(e_base)}")
        return None, None, None, debug


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
    result = gen_text.split(SPECIAL_CODE, 1)[1] if SPECIAL_CODE in gen_text else gen_text[len(prompt):]
    for stop in [SPECIAL_PSEUDO, SPECIAL_CODE, "<|endoftext|>"]:
        if stop in result:
            result = result.split(stop)[0]
            break
    return result.strip()


def main():
    st.sidebar.title("⚙️ Settings")
    st.sidebar.write("MODEL_ID = `" + MODEL_ID + "` (change at top of file if needed)")

    model, tokenizer, device, debug = load_model_and_tokenizer(MODEL_ID)

    st.subheader("Repository inspection")
    if debug and "repo_inspect" in debug:
        ri = debug["repo_inspect"]
        if ri.get("ok"):
            st.write("Files in repo (first 200):")
            st.write(ri["files"][:200])
            if any(f.lower().endswith(("adapter_model.bin", "adapter_config.json", "pytorch_model.bin")) for f in ri["files"]):
                st.success("Repo appears to contain model/adapter files (good).")
        else:
            st.error("Could not list repo files: " + ri.get("error", "unknown"))

    st.subheader("Loader traces")
    if debug and "traces" in debug:
        for t in debug["traces"]:
            st.text("- " + t)

    if model is None or tokenizer is None:
        st.error("❌ Model/tokenizer failed to load. See traces above. Likely causes:")
        st.markdown(
            """
            - Repo is not a PEFT adapter or full model in expected format.  
            - Tokenizer is missing from the repo (common for adapter-only repos).  
            - `peft` / `transformers` version mismatch.  
            
            **Next steps:**  
            1. Check the repo files above — does it have `adapter_config.json`, `adapter_model.bin`, or `pytorch_model.bin`?  
            2. If the repo is adapter-only, set `tokenizer = AutoTokenizer.from_pretrained('gpt2')` or upload tokenizer files to the repo.  
            3. If you fine-tuned locally, export with `peft_model.save_pretrained(...)` and upload that folder to HF (include adapter files + adapter_config).  
            4. Consider loading the full merged model (use `AutoModelForCausalLM.from_pretrained` after merging weights).
            """
        )
        st.stop()

    st.success(f"✅ Model loaded and running on {device.upper()}")

    st.markdown("---")
    st.header("Generate C++ from pseudocode")
    col1, col2 = st.columns(2)
    with col1:
        pseudocode = st.text_area("Enter pseudocode:", value="read integer n\nfor i from 1 to n\n  print i", height=300)
        max_tokens = st.slider("Max Tokens", 50, 200, 120)
        num_beams = st.slider("Beam Size", 1, 10, 5)
        generate = st.button("🚀 Generate")
    with col2:
        st.subheader("Generated C++")
        if generate:
            if not pseudocode.strip():
                st.warning("Enter pseudocode first")
            else:
                with st.spinner("Generating..."):
                    try:
                        out = generate_code(model, tokenizer, device, pseudocode, max_tokens, num_beams)
                        st.code(out, language="cpp")
                        st.download_button("Download", out, "generated.cpp", "text/plain")
                    except Exception as e:
                        st.error(f"Generation error: {e}")

if __name__ == "__main__":
    main()
