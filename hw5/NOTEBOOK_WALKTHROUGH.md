# HW5 — Notebook Walkthrough (LLM Fine-Tuning + RAG)

Notebook covered: `finetuning_tutorial.ipynb`

**Goal in the top-level README:** fully fine-tune Qwen2.5-0.5B-Instruct with Hugging Face `transformers` + `trl`, add Retrieval-Augmented Generation to help the small model, and evaluate before vs after under GPU-memory and stability constraints.

This notebook matches the README well. Task: answer **multiple-choice (A–E) questions from past CS 189 exams** while keeping general knowledge, scored by a hidden Kaggle test set of 169 questions.

---

## Libraries used

| Library | Used for |
|---|---|
| `transformers==4.57.2` | `AutoModelForCausalLM`, `AutoTokenizer`, chat templates, `generate` |
| `trl` | `SFTTrainer`, `SFTConfig` (supervised fine-tuning) |
| `datasets` | MMLU (`cais/mmlu`), `Dataset.map`, `add_faiss_index` |
| `bitsandbytes` | 8-bit AdamW (`optim="adamw_8bit"`) |
| `accelerate` | `device_map="auto"` placement |
| `sentence-transformers` | `BAAI/bge-small-en-v1.5` embeddings for retrieval |
| `faiss-cpu` | nearest-neighbour index (through `datasets`) |
| `torch`, `pandas`, `matplotlib`, `re`, `random` | tensors, CSVs, loss plots, answer parsing |

---

## Pipeline, step by step

### 1. Model and prompt format
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`, loaded in fp16 on GPU.
- **Prompt:** *"Choose exactly one correct option from A, B, C, D, and E. Return your answer inside a LaTeX box."* followed by the question, lettered options, and `Answer:`.
- **Target:** the assistant turn is just `\boxed{X}`. `tokenizer.apply_chat_template` renders the user/assistant pair into Qwen's `<|im_start|>` / `<|im_end|>` format.
- **Answer extraction:** `parse_choice_from_boxed` regex-matches `\boxed{A–E}`, falling back to the last standalone A–E in the output.

### 2. Evaluation harness
`eval_mcq_accuracy` applies the chat template with a generation prompt, runs greedy `generate` (`do_sample=False`, 64 new tokens), parses the letter, and compares with the gold answer. It can also return a per-question details DataFrame.

### 3. Training data: MMLU machine-learning subset
112 questions (`cais/mmlu`, `machine_learning`), converted to SFT text with the same prompt format. The sample CS 189 evaluation CSV (25 questions) is used as the eval set.

### 4. SFT configuration (`SFTConfig`)
- Batch size 1 × gradient accumulation 4 = effective batch 4; **50 steps** (about 2 epochs over 112 examples).
- LR `1e-5`, weight decay 0.01, 5 warm-up steps, `adamw_8bit`, seed 189.
- Evaluation and checkpoints every 5–10 steps, `load_best_model_at_end=True` with `eval_loss` as the criterion.
- Small batch, gradient accumulation and 8-bit optimizer states are the memory-saving choices. Full fine-tuning of the 0.5B model (no LoRA) fits on a single Colab GPU this way.

---

## Results

### Tutorial baseline (naïve SFT)
| | Accuracy on the 25-question sample |
|---|---|
| Base model, zero-shot | **28%** (7/25) |
| After the tutorial's SFT on 4-option MMLU | **16%** (4/25), *worse* |

The tutorial SFT hurt, which motivated improving the recipe.

### Improved recipe (the "YOUR TURN" section)
1. **Match the test format with an extra option E.** `add_option_E` appends `"None of the answers."` to every 4-option MMLU question and, with probability 0.2, moves the correct answer into slot E (making "None of the answers" a distractor). That trains the model on the 5-option format and gives it some practice answering E.
2. Cosine LR schedule instead of linear, `eval_loss` for best-checkpoint selection.
3. Train again from freshly loaded base weights.

Outcome: the sample-set accuracy rises to **40%** (10/25), versus 28% base. Loss curves (50 steps): training loss falls from about **3.09 → 0.67**; eval loss falls from about **2.10 → 1.38** by step 20 and then **plateaus (~1.40)**, so most of the useful learning happens in the first 20 steps. Mean token accuracy on the eval set is about 0.72.

### Retrieval-Augmented Generation
- **Corpus:** all 14,042 questions in MMLU `test` (every subject), each formatted as question + choices + *"Correct answer: X. text"*, with the option-E treatment applied.
- **Embeddings and index:** `BAAI/bge-small-en-v1.5` (normalised embeddings) stored in a dataset column, and `Dataset.add_faiss_index(column="embedding")`.
- **Query:** question + all options → top-**k = 2** nearest chunks.
- **Prompt:** the chunks are prepended as `[CONTEXT i]` blocks with *"Use the following context if it is relevant to the question."* before the normal MCQ prompt.
- **Inference:** `predict_mcq_test_with_rag` runs greedy generation on the 169 Kaggle questions. Unparseable outputs default to `"A"`, and the output is written to `submission.csv` with columns `id`, `prediction`.

**Things to be aware of (from reading the code):**
- A RAG-aware training-text builder (`build_mmlu_sft_text_rag`, with a 10% "no context" dropout) and `prepare_mmlu_dataset_rag` are defined, but the training cell uses the **non-RAG** dataset. The model is therefore fine-tuned without context blocks but prompted with them at inference.
- The 40% figure is measured **without RAG**. The notebook does not report a RAG accuracy on the sample set, and the Kaggle score is not recorded in the notebook, so RAG's benefit is not quantified here.
- The retrieval corpus contains the answers, including those for the 112 ML questions used in SFT. It helps when test questions resemble MMLU material and is less useful for questions that do not.
- The 25-question sample set is used both for best-checkpoint selection and for the reported accuracy, so those numbers are optimistic and noisy at n = 25.

### Side-effects check (catastrophic forgetting / unintended skills)
The notebook reloads the base model and compares it with the fine-tuned model on four free-form prompts:
- **Powerhouse of a cell:** the base model answers correctly (mitochondria). The fine-tuned model wrongly answers *nucleus*. This is clear **forgetting** of general knowledge.
- **Umbrella/rain logic:** the tuned model says "yes" (the right conclusion, since the statement is the contrapositive) but with muddled reasoning, and is far shorter than the base model.
- **"2 + 2" multiple-choice:** the tuned model collapses to a bare `B.` (the base model also picks B). The terse format shows the `\boxed`-style training bleeding into other prompts. The answer is wrong in both cases (D is correct).
- **30-word limit:** both models overshoot, though the tuned one is much closer (about 32 words).

**Take-away.** Small-data full SFT on a 0.5B model can steer format and slightly raise exam accuracy, but it narrows behaviour and can erase general knowledge. The RAG layer is the README's proposed way to add knowledge without further weight updates, and this notebook sets that up. A fair evaluation would measure RAG and non-RAG on a held-out set larger than 25 questions.
