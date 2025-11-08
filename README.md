# LLM-101: Building Large Language Models from Scratch

A comprehensive learning project focused on understanding and implementing Large Language Models (LLMs) from the ground up. This repository contains hands-on implementations covering everything from data preprocessing to model training and fine-tuning.

## 📚 Learning Resources

This project follows:
- **Book**: "Build a Large Language Model (from Scratch)" by Sebastian Raschka
- **Video Series**: [Building LLMs from Scratch - YouTube Playlist](https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)

## 🎯 Project Overview

This repository documents the journey of building a GPT-2-like language model from scratch, covering all the fundamental concepts and implementation details. Each notebook focuses on a specific topic, building upon previous knowledge to create a complete understanding of how modern LLMs work.

## 📓 Notebooks Structure

### Part 1: Data Preparation & Tokenization (Notebooks 1-6)
Introduction to data preparation, Byte Pair Encoding (BPE) tokenization, input-target pair creation, token embeddings, positional encodings, and complete data preprocessing pipeline.

### Part 2: Attention Mechanisms (Notebooks 7-11)
Basic attention mechanism, self-attention with trainable weights, causal masking for autoregressive models, multi-head attention implementation, and high-level overview of attention mechanisms.

### Part 3: Transformer Architecture (Notebooks 12-16)
Layer normalization, GELU activation function, residual/shortcut connections, complete transformer block implementation, and full GPT-2 model architecture.

### Part 4: Training & Evaluation (Notebooks 17-20)
Next token prediction task, loss function implementation, model evaluation metrics, and complete pre-training pipeline.

### Part 5: Text Generation & Sampling (Notebooks 21-23)
Temperature scaling for text generation, top-k sampling strategy, and end-to-end revision notes.

### Part 6: Model Management (Notebooks 24-25)
Model checkpointing and loading pre-trained GPT-2 weights.

### Part 7: Fine-tuning (Notebooks 26-34)
Introduction to fine-tuning, data loaders for classification, architecture for classification tasks, spam classification example, instruction fine-tuning introduction, batching for instruction tuning, instruction fine-tuning loop, extended instruction training, and model evaluation with Ollama.

## 🗂️ Project Structure

```
llm-101/
├── data/
│   └── the-verdict.txt          # Training dataset
├── 01_data_preparation_and_sampling.ipynb
├── 02_byte_pair_encoding.ipynb
├── ...
├── 34_evaluating_finetuned_LLM_using_Ollama.ipynb
└── README.md
```

## 🚀 Key Concepts Covered

- **Tokenization**: Byte Pair Encoding (BPE)
- **Embeddings**: Token and positional embeddings
- **Attention Mechanisms**: Self-attention, multi-head attention, causal masking
- **Transformer Architecture**: Layer normalization, GELU activation, residual connections
- **Model Architecture**: GPT-2 implementation
- **Training**: Pre-training pipeline, loss functions, evaluation metrics
- **Text Generation**: Temperature scaling, top-k sampling
- **Fine-tuning**: Classification and instruction tuning
- **Model Deployment**: Loading/saving weights, evaluation with Ollama

## 🛠️ Technologies Used

- Python
- PyTorch
- Jupyter Notebooks

## 📖 Learning Path

The notebooks are designed to be followed sequentially, as each builds upon concepts introduced in previous notebooks. Start with data preparation and work your way through to fine-tuning and evaluation.

## 🙏 Acknowledgments

- Sebastian Raschka for the excellent book "Build a Large Language Model (from Scratch)"
- The YouTube series creators for the comprehensive video tutorials
- All contributors to the open-source LLM community

## 📝 Notes

This is an educational project focused on understanding the fundamentals of LLMs. The implementations are designed for learning purposes and may not be optimized for production use.

---

Happy Learning! 🎓
