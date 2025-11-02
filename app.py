import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
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

@st.cache_resource
def load_model():
    """Load the fine-tuned GPT-2 or LoRA adapter model."""
    MODEL_ID = "naeaeaem/gpt2-finetuned"  # ✅ Replace with your Hugging Face model ID

    try:
        with st.spinner("🔄 Loading model... Please wait..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

            # Detect if this is a PEFT (LoRA) model
            try:
                config = PeftConfig.from_pretrained(MODEL_ID)
                base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
                model = PeftModel.from_pretrained(base_model, MODEL_ID)
            except Exception:
                # Not a PEFT model, load directly
                model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device).eval()

            return model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Ensure the model repo is public and MODEL_ID is correct.")
        return None, None, None


def generate_code(model, tokenizer, device, pseudo, max_new_tokens=120, num_beams=5):
    """Generate C++ code from pseudocode."""
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
    st.markdown('<h1 class="main-header">🐍 Pseudocode to C++ Code Generator</h1>', unsafe_allow_html=True)
    st.markdown("Convert pseudocode into **C++** using a fine-tuned GPT-2 model.")

    model, tokenizer, device = load_model()
    if model is None:
        st.stop()

    st.sidebar.title("⚙️ Settings")
    max_tokens = st.sidebar.slider("Max Tokens", 50, 200, 120, 10)
    num_beams = st.sidebar.slider("Beam Size", 1, 10, 5, 1)
    st.sidebar.info(f"Running on: **{device.upper()}**")

    examples = {
        "Simple I/O": "read integer n\nprint n",
        "Addition": "read integer x\nread integer y\nprint x plus y",
        "Loop": "for i from 1 to 10\n  print i"
    }

    example = st.sidebar.selectbox("Examples", ["Custom"] + list(examples.keys()))
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Input Pseudocode")
        pseudo = st.text_area(
            "Enter pseudocode:",
            value=examples.get(example, ""),
            height=300,
            placeholder="Example:\nread integer n\nfor i from 1 to n\n  print i"
        )
        generate = st.button("🚀 Generate Code", use_container_width=True)

    with col2:
        st.subheader("💻 Generated C++ Code")
        if generate:
            if not pseudo.strip():
                st.warning("⚠️ Please enter pseudocode first!")
            else:
                with st.spinner("⏳ Generating..."):
                    try:
                        code = generate_code(model, tokenizer, device, pseudo, max_tokens, num_beams)
                        st.code(code, language="cpp")
                        st.download_button("📥 Download Code", code, "generated_code.cpp", "text/plain")
                    except Exception as e:
                        st.error(f"❌ Generation error: {e}")
        else:
            st.info("👈 Enter pseudocode and click Generate Code")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#888'>Built with Streamlit | GPT-2 + LoRA</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
