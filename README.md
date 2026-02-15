# Food Content Classifier with LoRA Fine-Tuning

*A journey in parameter-efficient fine-tuning: From ambitious goals to practical solutions*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LoRA Fine-Tuning](https://img.shields.io/badge/PEFT-Fine--Tuning-pink?style=for-the-badgeor=white)](https://huggingface.co/docs/peft/en/index)
[![LoRA Fine-Tuning](https://img.shields.io/badge/LoRa-Fine--Tuning-orange?style=for-the-badgeor=white)](https://huggingface.co/docs/peft/conceptual_guides/lora)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


##  Project Overview

This project demonstrates **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning. Through an iterative process of experimentation and refinement, it showcases both the **technical implementation of LoRA** and the **practical decision-making** required in real ML projects.

##  Final Results

| Metric | Value |
|--------|-------|
| **Task** | Binary classification (food/non-food) |
| **Accuracy** | 95%+ |
| **Training Loss** | 8.21 → 0.12 |
| **Parameters Trained** | 294,912 / 60,801,536 (0.48%) |
| **Training Time** | ~2 minutes (GPU) |

## Project Journey

### Initial Goal: Multi-field Food Extraction
Initially aimed to extract structured food information (foods, drinks, categories) from text.

### Challenge Encountered
The dataset had severe class imbalance:
- `none`: 681 examples (53%)
- `beverage`: 299 examples (23%)  
- `breakfast`: 18 examples (1.4%)
- `lunch`: 10 examples (0.8%)

**Result:** Multi-category model achieved only 75% accuracy with poor generalization. The model over-predicted common categories and failed on rare ones.

### Solution: Pivot to Binary Classification
Simplified the task to: **"Is this text about food/drinks?"** → "yes" or "no"

**Why this decision matters:**
- ✅ Demonstrates LoRA fundamentals clearly
- ✅ Achieves reliable performance (95%+ accuracy)
- ✅ Shows practical engineering judgment
- ✅ Provides solid foundation for future extension

> **Key Learning:** In ML projects, matching task complexity to available data is crucial. A simpler task done well is more valuable than a complex task done poorly.

##  Quick Start

### Installation
```bash
git clone https://github.com/Ravisankar-S/LoRA-Fine-Tuning-exp
cd LoRA-Fine-Tuning-exp
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Verify Fine-Tuning Actually Worked
```bash
python evaluate.py
```

This compares the fine-tuned model against base T5-small to prove LoRA made a difference.

### Inference
```python
python inference.py
```

##  Usage Example
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load model
base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
model = PeftModel.from_pretrained(base_model, "./food_classifier")
tokenizer = AutoTokenizer.from_pretrained("./food_classifier")

def classify(text):
    prompt = f"Is this about food or drinks? {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(classify("I had pizza for lunch"))        # → "yes"
print(classify("I bought a new laptop"))        # → "no"
```

##  Verification: Did Fine-Tuning Work?

Run `evaluate.py` to see side-by-side comparison:
```
BASE MODEL (no fine-tuning):
  Package with zucchini and carrots    → 'food and drinks'
  I ate pizza for lunch                → 'yes'
  Picture of my laptop                 → 'picture'
  
FINE-TUNED MODEL (with LoRA):
  Package with zucchini and carrots    → 'yes'
  I ate pizza for lunch                → 'yes'
  Picture of my laptop                 → 'no'

✅ Fine-tuned Accuracy: 100% (5/5)
⚠️  Base Model Accuracy: 40% (2/5)
```

**Conclusion:** Fine-tuning clearly improved task-specific performance!

##  Technical Implementation

### Architecture
- **Base Model:** T5-Small (60M parameters)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Task:** Seq2seq text classification

### LoRA Configuration
```python
LoraConfig(
    r=8,                          # Rank: 8
    lora_alpha=16,                # Alpha: 16
    target_modules=["q", "v"],    # Attention Q & V matrices
    lora_dropout=0.05,
    task_type="SEQ_2_SEQ_LM"
)
```

### Why LoRA?

| Approach | Parameters | Memory | Time |
|----------|-----------|--------|------|
| Full Fine-tuning | 60.8M (100%) | ~4GB | ~30min |
| **LoRA** | **295k (0.48%)** | **~1GB** | **~2min** |

**Benefits:**
- 200x fewer parameters
- 4x less memory
- 15x faster training
- Multiple adapters per base model

### Training Details
- **Dataset:** [FoodExtract-1k](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k)
- **Train/Test:** 1,278 / 142 examples
- **Epochs:** 15
- **Batch Size:** 16
- **Learning Rate:** 3e-4
- **Optimizer:** AdamW

##  Results

### Training Curve
```
Epoch  | Loss
-------|-------
1      | 8.21
5      | 0.20
10     | 0.12
15     | 0.12  ✅ Converged
```

### Test Performance
| Category | Examples | Accuracy |
|----------|----------|----------|
| Food-related | 71 | 97% |
| Non-food | 71 | 94% |
| **Overall** | **142** | **95%** |

##  What This Project Demonstrates

### Technical Skills
- ✅ LoRA implementation and configuration
- ✅ PEFT library usage
- ✅ Seq2seq model training
- ✅ Model evaluation and comparison
- ✅ Hugging Face ecosystem

### ML Engineering Skills
- ✅ Problem scoping and pivoting
- ✅ Dataset analysis and limitation identification
- ✅ Trade-off evaluation (complexity vs. performance)
- ✅ Reproducible experiment design
- ✅ Clear documentation and communication

### Lessons Learned
1. **Data quality > Model size:** 1k examples with class imbalance → simplify task
2. **Iteration is key:** Started with extraction → pivoted to classification
3. **Validate assumptions early:** Check data distribution before training
4. **Engineering judgment matters:** Knowing when to simplify is a skill

##  Future Extensions

### Short-term (Feasible)
- [ ] Multi-label categories (breakfast, lunch, dinner)
- [ ] Confidence scores
- [ ] Support for longer texts (512+ tokens)

### Long-term (Requires more data)
- [ ] Named entity extraction for specific foods
- [ ] Multi-lingual support
- [ ] Nutritional information extraction

##  Project Structure
```
.
├── train.py              # Training script
├── evaluate.py           # Compare base vs fine-tuned
├── inference.py          # Simple usage examples
├── notebooks/
│   └── exploration.ipynb # Optional: data exploration
├── food_classifier/      # Saved LoRA adapters (gitignored)
├── requirements.txt
└── README.md
```

## Why Not Multi-Category Extraction?

**Attempted:** Extract foods, drinks, and meal categories simultaneously

**Challenge:** Severe class imbalance made reliable extraction impossible with available data:
- Majority class (none): 53% of data
- Minority classes (breakfast, lunch): <2% each
- Result: Model defaulted to predicting majority classes

**Decision:** Focus on binary classification to demonstrate LoRA fundamentals clearly

**Learning:** In production ML, delivering a reliable 95% solution is better than an unreliable 75% solution with more features.

## References

- [LoRA Paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [T5 Paper (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [Parameter-Efficient Transfer Learning](https://arxiv.org/abs/1902.00751)

## 📄 License

MIT License - see LICENSE file

## 👤 Author

**[Ravisankar S]**
- LinkedIn: [Ravisankar-S](https://www.linkedin.com/in/ravisankar-cs/)
- Email: sravisankar11@gmail.com

---

*This project demonstrates that successful ML engineering isn't just about implementing algorithms—it's about making smart decisions when faced with real-world constraints.*

⭐ **If you found this project insightful, please star it!**