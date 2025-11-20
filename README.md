🚀 Pseudocode-to-C++ Code Generator using GPT-2 + LoRA

An intelligent AI system that converts natural pseudocode descriptions into fully executable C++ programs — fine-tuned using GPT-2 with Parameter-Efficient Fine-Tuning (LoRA).

Built by Muhammad Naeem and Aneela Bashir.

✨ Overview

This project bridges the gap between human logic and machine implementation. By fine-tuning GPT-2 with LoRA, we created a lightweight yet powerful model capable of generating context-aware, structurally correct C++ code from simple English-like pseudocode instructions.

The system is deployed as a free Streamlit web app, allowing students and developers to generate code instantly.

💡 Key Highlights
🔹 1. Dataset Processing

Used 18,356+ pseudocode-to-code pairs from the SPOC dataset.

Handled:

Multi-line C++ code

Indentation + brace alignment

Syntax variations + formatting inconsistencies

Added custom markers:

<|pseudo|> … <|code|>

🔹 2. LoRA-Based Fine-Tuning

Applied LoRA to GPT-2 for efficient training.

Only 1.2% trainable parameters
→ 1.5M trainable out of 126M total.

Trained for 5 epochs with:

FP16 mixed precision

Gradient accumulation

Early stopping

Google Colab T4 GPU

🔹 3. Smart Training Pipeline

Custom tokenization + special tokens

Masked pseudocode tokens so the loss is only computed on generated C++

Beam search decoding (num_beams=5) for high-quality outputs

Automatic and manual evaluation

📊 Evaluation Results
Metric	Score
BLEU	13.93
Approx. CodeBLEU	0.405
Code Quality (Manual)	85%
Structural Accuracy	82%
Generation Success	100%
🧠 Capabilities of the Model

✔ Generates syntactically correct C++ code
✔ Proper brace management ({} alignment)
✔ Handles loops, conditionals, functions, I/O
✔ Understands multi-step logic
✔ Context-aware variable usage
✔ Fully deterministic or creative output (beam vs sampling)

🖥️ Live Demo

🎯 Try the web app here:
👉 https://lnkd.in/deRaZdes

📘 Documentation & Code

📄 Full Project Breakdown:
👉 https://lnkd.in/dbUq5yRJ
🚀 How It Works

User enters pseudocode in plain English.

Model converts to structured <|pseudo|> ... <|code|> format.

GPT-2 (fine-tuned with LoRA) generates accurate C++ code.

Output appears instantly in the Streamlit interface.

🛠️ Tech Stack

GPT-2 (Hugging Face Transformers)

LoRA (Parameter Efficient Fine-Tuning)

PyTorch

Streamlit

Beam Search Decoding

Google Colab GPU


⭐ Future Improvements

Support for Python, Java, and C

Add CodeBLEU official evaluation pipeline

Integrate syntax error auto-fixing

Add function decomposition and multi-file generation

📢 Contributions

Pull requests and enhancements are welcome!
If you find bugs or have feature requests, open an issue in the repository.

📜 License

This project is released under the MIT License.
