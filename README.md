# NanoGPT Story Generation (COMP8650 Assignment)

## Overview
This project focuses on improving short story generation using NanoGPT on the ROCStories dataset. The objective was to generate coherent and grammatically correct five-sentence stories while staying within a 32M parameter constraint.

Instead of increasing model size, improvements were achieved through better data preparation, augmentation, and decoding strategies.

---

## Key Results
| Model | Validation Loss | Perplexity |
|------|----------------|-----------|
| Baseline | 3.342 | 28.28 |
| Final Model | 3.233 | 25.36 |

- Improved coherence and readability  
- Reduced grammatical errors  
- Better overall story structure  

---

## Approach

### 1. Data Preparation
- Cleaned and normalized raw text (punctuation, spacing, formatting)
- Removed low-quality and inconsistent samples
- Retained only valid 5-sentence stories
- Final training format:

sentence1 sentence2 sentence3 sentence4 sentence5 <|endoftext|>


---

### 2. Sentence Tone Augmentation
- Kept the first sentence fixed  
- Rewrote sentences 2–5 with:
  - Positive continuation  
  - Negative continuation  
- Increased diversity of story outcomes  
- Automated using Llama 3.2 via Ollama  

---

### 3. Strategic Decoding
- Tuned temperature and top-k sampling  
- Final configuration:

temperature = 0.95
top_k = 40

- Achieved a balance between creativity and grammatical correctness  

---

## Experiments & Insights

### What Worked
- Clean training data improved stability  
- Augmentation improved diversity  
- Controlled decoding improved readability  

### What Failed
- `<|story|>` format  
- `Prompt: ... Continuation: ...` format  

These caused train-test mismatch and token leakage in generated outputs.

---

## Sample Output

**Prompt:**  
Tom decided to cook dinner for his friends.

**Generated:**  
He bought all the ingredients and made it online. At the end of the night Tom finished cooking the meal. Everyone loved his meal. Tom’s friends didn’t want to eat with him anymore.

---

## Limitations
- Occasional repetition  
- Weak or inconsistent endings  
- Logical flow issues in final sentences  

---

## Tech Stack
- Python  
- PyTorch  
- NanoGPT  
- tiktoken (GPT-2 tokenizer)  
- Ollama (Llama 3.2 for augmentation)  

---

## How to Run

### 1. Data Preparation

python data/rocstories/prepare_text_files.py


### 2. Train Model

python train.py config/train_rocstories.py


### 3. Generate Stories

python sample.py --out_dir=out-rocstories


---

## Key Takeaway
Better data and decoding strategies can significantly improve model performance without increasing model size.

---

## Author
Justin Paul Kolengadan  
Master of Machine Learning and Computer Vision  
Australian National University  
