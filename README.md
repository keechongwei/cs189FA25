# COMPSCI 189: Introduction To Machine Learning Coursework 

This repository contains a collection of coursework completed as part of COMPSCI 189, an upper division Machine Learning Course at UC Berkeley.

Across five assignments, I implemented, trained, analyzed, and debugged models spanning **representation learning, clustering, generative modeling, control, and large language model fine-tuning**.

The work emphasizes **hands-on interaction with core ML concepts**, not just model usage — including optimization, evaluation, failure modes, and distributional considerations.

---

## 📚 Topics & Skills Demonstrated

Across these assignments, I worked directly with:

- **Supervised & unsupervised learning**
- **Representation learning and feature spaces**
- **Clustering and evaluation of latent structure**
- **Generative modeling and conditional generation**
- **Control and sequential decision-making**
- **Neural network optimization and diagnostics**
- **Modern LLM fine-tuning pipelines**
- **PyTorch, Hugging Face Transformers, TRL**

---

## 🗂 Repository Structure
hw1/ **Representation learning & clustering** <br>
hw2/ **Control & style conditioning** <br>
hw3/ **Modern ML pipelines & evaluation** <br>
hw4/ **Generative Modeling** <br>
hw5/ **Large Language Model fine-tuning** <br>

Each folder contains self-contained Jupyter notebooks with experiments, visualizations, and analysis.

---

## Coursework Breakdown

### Homework 1 — Representation Learning & Clustering (Fashion-MNIST)
**Notebooks:** `fashion_pt_1.ipynb`, `fashion_pt_2.ipynb`

- Learned image representations using neural networks
- Applied **K-means clustering** in learned feature spaces
- Visualized cluster structure and failure cases
- Compared raw pixel space vs learned embeddings

**Concepts:**  
Feature learning · Latent spaces · Unsupervised learning

---

### Homework 2 — Arena Control & Style Conditioning
**Notebooks:** `arena_warmup.ipynb`, `arena_style_control.ipynb`

- Built neural controllers for an interactive environment
- Trained policies under different objectives and constraints
- Introduced **style conditioning** to control behavior
- Analyzed how conditioning signals affect learned policies

**Concepts:**  
Control · Conditioning · Sequential decision-making 

---

### Homework 3 — Modern ML Pipelines & Evaluation
**Notebook:** `hw3.ipynb`

- Worked with contemporary ML tooling and datasets
- Implemented training, evaluation, and visualization pipelines
- Compared models under different objectives and metrics
- Focused on **generalization and evaluation rigor**

**Concepts:**  
Train/validation splits · Metrics · Overfitting 

---

### Homework 4 — Generative Modeling
**Notebooks:** `hw4_part1.ipynb`, `hw4_part2.ipynb`

- Implemented and analyzed generative models
- Studied how sampling procedures affect outputs
- Visualized learned distributions and generation quality
- Explored tradeoffs between fidelity and diversity

**Concepts:**  
Generative models · Sampling · Distribution learning · Model diagnostics

---

### Homework 5 — Large Language Model Fine-Tuning
**Notebook:** `finetuning_tutorial.ipynb`

- Fine-tuned a **Qwen 2.5 (0.5B) instruction-tuned LLM**
- Used Hugging Face `transformers` and `trl` libraries
- Performed **full supervised fine-tuning**
- Innovated use of Retrieval Augmented Generation to improve performance of small Qwen 2.5 model
- Evaluated performance before vs after fine-tuning
- Worked with real constraints: GPU memory, batch sizes, optimization stability

**Concepts:**  
LLMs · Supervised fine-tuning (SFT) · Cross-entropy optimization · RAG

---

## ⚙️ Tools & Libraries

- **PyTorch**
- **Hugging Face Transformers & Datasets**
- **TRL (SFTTrainer)**
- NumPy, Pandas, scikit-learn
- Matplotlib / Plotly for visualization

---

## 📄 Disclaimer

This repository contains coursework completed for educational purposes.  
It is shared to demonstrate technical experience and understanding of machine learning concepts.
