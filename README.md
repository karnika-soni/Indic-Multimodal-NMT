# Indic Multimodal Neural Machine Translation

A vision-grounded multimodal Neural Machine Translation (NMT) system for low-resource Indic language translation, combining textual representations from **mBART** with visual region features extracted using **Detectron2**.

The project investigates how visual context can improve translation quality, particularly for **named entities, contextual ambiguity, and visually grounded content** in image-text pairs.

---

## Overview

Traditional Neural Machine Translation systems rely entirely on textual context. However, image-caption datasets contain additional visual information that can help resolve ambiguities that cannot be inferred reliably from text alone.

This project develops a **multimodal Transformer architecture** that integrates:

* **mBART-50** for multilingual text representation and generation
* **Detectron2** for extracting region-level visual features
* **Multimodal fusion** between textual and visual representations
* **NER masking** to improve named-entity consistency
* **Parameter-efficient fine-tuning** under compute constraints
* Distributed training and mixed-precision optimization for scalable experimentation

The final multimodal system achieved:

| Model                    |      BLEU |
| ------------------------ | --------: |
| Text-only mBART baseline | **38.84** |
| Multimodal mBART         | **40.51** |

**Improvement: +1.67 BLEU**

---

## Architecture

The overall pipeline is:

```text
                 Image
                   │
                   ▼
              Detectron2
                   │
                   ▼
        Visual Region Features
             (36 × 1024)
                   │
                   ▼
            Vision Encoder
                   │
                   │
                   ▼
Text ──► mBART Encoder ──► Text Features
                   │
                   │
                   ▼
             Multimodal Fusion
                   │
                   ▼
             mBART Decoder
                   │
                   ▼
          Indic Translation
```

### Text Encoder

The system uses:

```text
facebook/mbart-large-50-many-to-many-mmt
```

The mBART encoder produces contextualized text representations with hidden dimension:

```text
1024
```

### Visual Encoder

Images are processed using Detectron2 to obtain region-level visual representations.

The extracted feature representation used by the model has the shape:

```text
36 × 1024
```

A lightweight vision encoder applies normalization and dropout before multimodal fusion.

### Multimodal Fusion

The text and visual representations are concatenated along the sequence dimension:

```text
Text features:
(batch, text_length, 1024)

Visual features:
(batch, 36, 1024)

Fused representation:
(batch, text_length + 36, 1024)
```

The resulting representation is provided to the mBART decoder through the encoder output interface.

---

## Dataset

The project uses image-text pairs derived from **Flickr30K**.

The processed dataset contains more than **30K images** and associated multilingual/textual annotations.

After matching the metadata against extracted Detectron2 features, samples with available visual representations are retained for multimodal training.

Example visual feature:

```text
image → Detectron2 → 36 region features → 36 × 1024
```

---

## Tokenization Experiments

Several tokenization strategies were investigated for Indic text:

* BPE
* WordPiece
* SentencePiece

The approaches were evaluated using:

* Token fragmentation
* Morphological preservation
* Representation efficiency

### Selected Tokenizer

**SentencePiece** was selected because it provided better preservation of Indic morphological structure and reduced unnecessary fragmentation.

---

## Named Entity Handling

Named entities are particularly important in machine translation because incorrect segmentation or translation can significantly change meaning.

The project incorporates **NER masking** to improve the consistency of named entities between source and target translations.

The goal is to preserve entities such as:

```text
PERSON
LOCATION
ORGANIZATION
PRODUCT
```

rather than allowing the translation model to incorrectly alter entity boundaries or identity.

---

## Multimodal Fusion Experiments

The project compares different approaches to integrating visual information.

### Early Fusion

Visual representations are introduced earlier in the model pipeline so that the network can jointly process textual and visual information.

### Late Fusion

Textual representations are first encoded independently, after which visual features are appended to the encoded sequence.

The implemented late-fusion approach produces:

```text
[textual hidden states] + [visual hidden states]
```

and passes the resulting sequence to the mBART decoder.

---

## Parameter-Efficient Fine-Tuning

Different fine-tuning strategies were investigated:

### Full Fine-Tuning

The complete mBART model is updated during training.

Advantages:

* Maximum adaptation capacity
* Potentially better task-specific performance

Disadvantages:

* Very high GPU memory requirements
* Large number of trainable parameters

### LoRA

Low-Rank Adaptation was investigated as a parameter-efficient alternative.

Advantages:

* Significantly fewer trainable parameters
* Lower memory requirements
* Faster experimentation
* Easier checkpoint storage

The project compares these approaches under constrained computational resources.

---

## Distributed Training

For scalable experiments, the project uses:

```text
PyTorch DistributedDataParallel (DDP)
```

The training workflow supports:

* Multi-GPU training
* Mixed-precision training
* Gradient-based optimization
* Checkpoint recovery
* Periodic evaluation
* Reproducible experiment configuration

Conceptually:

```text
                Training Dataset
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       GPU 0         GPU 1        GPU N
          │            │            │
          └────────────┼────────────┘
                       │
                  DDP Synchronization
                       │
                       ▼
                 Updated Model
```

---

## Training

The model is optimized using **AdamW**.

A typical training configuration includes:

```text
Optimizer: AdamW
Learning rate: 3e-5 / task-dependent
Mixed precision: Enabled
Distributed training: PyTorch DDP
Checkpointing: Enabled
Periodic evaluation: Enabled
```

The exact hyperparameters are configurable depending on available GPU memory and the selected fine-tuning strategy.

---

## Evaluation

Translation quality is evaluated using **BLEU**.

The final results are:

| Experiment       |      BLEU |
| ---------------- | --------: |
| Text-only mBART  |     38.84 |
| Multimodal mBART | **40.51** |

The multimodal model improves over the text-only baseline by:

```text
40.51 − 38.84 = +1.67 BLEU
```

This indicates that incorporating visual information provides additional useful context beyond the textual input alone.

---

## Results

### Multimodal vs Text-only

```text
BLEU

Text-only mBART     ███████████████████████████████████████  38.84
Multimodal mBART    █████████████████████████████████████████ 40.51
```

The improvement demonstrates the potential of visual grounding for multimodal Indic language translation.

---

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* mBART-50
* Detectron2
* SentencePiece
* PyTorch DistributedDataParallel
* CUDA
* Mixed-Precision Training
* SacreBLEU
* NumPy
* Pandas

---

## Key Contributions

1. **Vision-grounded Indic NMT**

   Integrates image-derived visual representations with multilingual textual representations for context-aware translation.

2. **Multimodal mBART architecture**

   Extends mBART with region-level visual features extracted from images.

3. **Indic tokenization analysis**

   Compares BPE, WordPiece, and SentencePiece using fragmentation and morphological preservation criteria.

4. **Named Entity Preservation**

   Uses NER masking to improve named-entity consistency during translation.

5. **Fusion strategy comparison**

   Investigates early and late multimodal fusion strategies.

6. **Parameter-efficient training**

   Compares LoRA and full fine-tuning approaches under compute constraints.

7. **Scalable training pipeline**

   Supports DDP, mixed precision, checkpoint recovery, and periodic evaluation.

---

## Results Summary

The final multimodal system achieved:

> **40.51 BLEU**

compared with:

> **38.84 BLEU**

for the text-only mBART baseline.

This corresponds to an improvement of:

> **+1.67 BLEU**

demonstrating that visual grounding can provide useful additional context for multimodal Indic translation.

---

## Future Work

Potential extensions include:

* Cross-attention based visual-text fusion
* Vision-language pretrained encoders
* More Indic language pairs
* Larger multilingual datasets
* Improved visual feature extraction
* Contrastive multimodal objectives
* Stronger NER and entity alignment
* Human evaluation alongside BLEU
* COMET and chrF evaluation
* More efficient LoRA/QLoRA training

---

## Author

**Karnika Soni**

GitHub: `karnika-soni`
