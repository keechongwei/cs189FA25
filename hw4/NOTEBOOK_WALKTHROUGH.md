# HW4 — Notebook Walkthrough (CNNs, Transformers & Transfer Learning)

Notebooks covered: `hw4_part1.ipynb`, `hw4_part2.ipynb`

**Goal in the top-level README:** "generative models", sampling procedures, learned distributions, and the fidelity-vs-diversity trade-off.

> **Mismatch with the README.** The notebooks are about **CNN, ResNet and Transformer architectures** and **transfer learning** (DNABERT for DNA sequences, ConvNeXt for audio spectrograms). Part 1 does end with a small *generative* step: a from-scratch transformer that greedily generates text. There are no sampling strategies, distribution visualisations or fidelity/diversity analysis. The sections below describe what the code actually does.

---

## Libraries used

| Library | Used for |
|---|---|
| `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim.AdamW` | all models and training loops |
| `torchvision` (`transforms`, `models.convnext_base`) | image transforms; pretrained ConvNeXt-Base |
| `torchaudio` | loading `.wav` files and `transforms.Spectrogram` |
| `transformers` (`AutoTokenizer`, `AutoModel`) | pretrained **DNABERT-6** (`zhihan1996/DNA_bert_6`) |
| `datasets` | mini-ImageNet and TinyStories from Hugging Face |
| `scikit-learn` | `train_test_split` (with `stratify` for DNA) |
| `pandas`, `numpy`, `matplotlib`, `tqdm` | data handling, metric plots, progress bars |
| `pygame` | local audio playback (Colab uses `IPython.display.Audio`) |

---

## `hw4_part1.ipynb`

### Subpart 1 — A plain CNN
- **Data:** a 10-class subset of mini-ImageNet. A custom `MiniImageNetDataset` (`__len__`, `__getitem__` → image tensor and label) is wrapped in `DataLoader(batch_size=32, shuffle=True)`.
- **Transforms:** train uses `Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ToTensor → Normalize(ImageNet mean/std)`; validation uses `CenterCrop` instead of the random operations.
- **Model:** `Conv(3→16, 3×3, s2, p1) → ReLU → Conv(16→16, 7×7, s2, p1) → ReLU → flatten → Linear(46656→10)`. The 46,656 comes from the 16×54×54 feature map after the two strided convolutions.
- **Training:** a hand-written loop (train / eval modes, `zero_grad → forward → loss → backward → step`, `torch.no_grad()` for validation). `AdamW(lr=3e-4)` and `CrossEntropyLoss`, 20 epochs.
- **Result:** train accuracy **0.579**, validation accuracy **0.52** (noisy from epoch to epoch). The reported "loss" is summed over batches rather than averaged (the first epoch shows about 68), so it is not comparable to a per-sample cross-entropy.

### Subpart 2 — ResNet-18
- **`ResidualBlock`:** `Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → (+ skip) → ReLU`. When downsampling, the first convolution uses stride 2 and the skip path is a 1×1 stride-2 convolution so shapes match. Every block here uses a 1×1 convolution on its skip path (also when not downsampling).
- **`ResNet18`:** 7×7 stride-2 conv + BatchNorm + ReLU + MaxPool → four stages of two blocks (64 → 128 → 256 → 512 channels, downsampling at the start of stages 2–4) → `AdaptiveAvgPool2d → Flatten → Linear(512→10)`. Each stage is also wrapped in an extra 1×1-conv skip connection (`skip1…skip4`), which goes beyond the standard architecture. A shape-check cell confirms the stage outputs `56→56→28→14→7`.
- **Training:** `AdamW(lr=5e-4)`, 50 epochs.
- **Result:** train accuracy **0.876**, validation accuracy **0.685**, much better than the plain CNN. The gap between train and validation also shows mild overfitting.

### Subpart 3 — Transformer from scratch
Every component is implemented and shape-tested against PyTorch where one exists:

| Component | Notes |
|---|---|
| `softmax` | numerically stable (subtracts the max); checked with `torch.allclose` against `F.softmax` |
| `scaled_dot_product_attention` | `softmax(QKᵀ/√d_k + mask) V`; masked positions set to `−inf`; matches `F.scaled_dot_product_attention` |
| `AttentionHead` | separate `W_q`, `W_k`, `W_v` linear layers. Supports self-attention and cross-attention (K and V from `encoder_output`) |
| `MultiHeadAttention` | 8 heads of size `d_model/8` in an `nn.ModuleList`, concatenated then projected with `W_out` |
| `EncoderLayer` | self-attention → residual → LayerNorm → FFN (512→2048→512, ReLU) → residual → LayerNorm (post-norm) |
| `DecoderLayer` | causal mask via `torch.triu(..., diagonal=1)` → masked self-attention → cross-attention to encoder → FFN, each with residual + LayerNorm |
| `PositionalEncoding` | sinusoidal (sin on even dims, cos on odd) |
| `TransformerEncoder` / `Decoder` / `Transformer` | embedding + positional encoding + 6 layers each; the decoder ends in a `Linear(d_model→vocab)` |

**Language-modelling task (TinyStories):**
- 1,000 stories are lower-cased, split into paragraphs and tokenised at **word level** with a regex, plus `<pad>`, `<start>` and `<end>` tokens.
- `create_sequences` makes sliding windows of 20 tokens, with the target equal to the input shifted by one token (next-token prediction). This gives 88,269 sequences. The notebook trains on the first 2,000 and validates on 200, in batches of 32.
- **Training:** `AdamW(lr=3e-4)`, `CrossEntropyLoss`, gradient clipping at 1.0, 10 epochs. Training loss falls 6.24 → **0.086** while validation loss only reaches **1.19** (from 6.01), a large generalisation gap on a tiny subset.
- **Generation:** `generate_sample` is greedy (`argmax`) autoregression with a sliding 20-token window. From "once upon a time" it produces degenerate output ("once upon a time a time time time time …").
- **Likely contributing factor (my reading of the code):** `Transformer.forward` passes the *same* token sequence `x` to both encoder and decoder. The encoder is bidirectional, so the decoder can cross-attend to the very tokens it is supposed to predict. This is a probable reason for the very low training loss and for the poor behaviour at generation time, when no future tokens are available.

---

## `hw4_part2.ipynb` — transfer learning on non-image/non-text modalities

### Subpart 1 — DNABERT for DNA sequences
- **Data:** `chimpanzee_train.txt`, `dog_train.txt`, `human_train.txt`. The target is the **species** (0/1/2); the `gene_family` column is kept but not used as the label. Each species is sampled down to the smallest count (656), giving 1,968 balanced sequences, of which 1,000 are sampled for training.
- **k-mer tokenisation:** `sequence_to_kmer` splits each DNA string into overlapping 6-mers separated by spaces, matching DNABERT-6's vocabulary. The tokenizer pads or truncates to 512 tokens.
- **Model:** `DNAClassifier` wraps `AutoModel.from_pretrained("zhihan1996/DNA_bert_6")` and adds `Linear(768→3)` on the **[CLS]** embedding (`last_hidden_state[:, 0, :]`).
- **Pipeline:** custom `DNADataset` → stratified 80/20 `train_test_split` → `DataLoader(32)`; `AdamW(lr=1e-4)`.
- **Results:**
  - 5 epochs: train **0.62**, validation **0.445**.
  - 25 epochs: train **0.949**, validation **0.43**, with validation loss rising to about 2.4. The model memorises the 800 training sequences and does not generalise (validation is only 200 sequences, so this is a noisy estimate).
  - Predictions on `dna_test.txt` are written to `dna_test_predictions.csv` for the class Kaggle competition.

### Subpart 2 — ConvNeXt for sound (UrbanSound8K, fold 1)
- **Audio → image:** `SpectrogramDataset` loads each `.wav`, averages channels to mono, applies `torchaudio.transforms.Spectrogram(n_fft=1024, normalized=True)`, expands to 3 channels (`spec.expand(3, -1, -1)`), and resizes to 224×224. The label is parsed from the filename (`[fsID]-[classID]-…`). An 80/20 split is used (about 140 validation clips).
- **Head swap:** `replace_final_convnext_linear_layer` replaces `classifier[-1]` of `convnext_base` with `Linear(in_features→10)`.
- **Three fine-tuning regimes (5 epochs, AdamW 1e-4):**

| Regime | Train acc | Val acc |
|---|---|---|
| Random init (`weights=None`) | 0.32 | 0.24 |
| Pretrained, **frozen** backbone (only the head trains) | 0.33 | 0.28 |
| Pretrained, **fully unfrozen** | 0.94 | **0.85** |

- **Interpretation:** full fine-tuning wins by a wide margin. Frozen features were tuned for natural images, and the raw (non-log, non-ImageNet-normalised) spectrograms look different from ImageNet, so the head alone cannot compensate. The written answer also notes that unfreezing only the head can be preferable when data is scarce.
- **Final model:** the fully unfrozen ConvNeXt is retrained for 25 epochs (train 0.98, val 0.836) and used to write `urbansound8k_test_predictions.csv` for Kaggle. A single-clip demo correctly classifies a clip as `dog_bark`.
