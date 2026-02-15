"""
Training script for food content binary classifier using LoRA.

This script fine-tunes T5-small using LoRA for binary classification
of food-related content.

Usage:
    python train.py
"""

import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training pipeline."""
    set_seed(42)
    
    print("="*80)
    print("FOOD CONTENT CLASSIFIER - LORA FINE-TUNING")
    print("="*80)
    
    # Configuration
    MODEL_NAME = "t5-small"
    MAX_INPUT = 256
    MAX_OUTPUT = 10
    OUTPUT_DIR = "./food_classifier"
    
    # Load dataset and tokenizer
    print("\n1. Loading dataset...")
    dataset = load_dataset("mrdbourke/FoodExtract-1k")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Format dataset
    def format_fn(ex):
        """Convert to binary classification format."""
        label = ex["gpt-oss-120b-label-condensed"]
        
        if "food_or_drink: 1" in label:
            target = "yes"
        else:
            target = "no"
        
        text = ex["sequence"][:300]  # Truncate long texts
        
        return {
            "input_text": f"Is this about food or drinks? {text}",
            "target_text": target
        }
    
    print("2. Formatting dataset...")
    formatted = dataset["train"].map(format_fn)
    split = formatted.train_test_split(test_size=0.1, seed=42)
    
    print(f"   ✅ Train examples: {len(split['train'])}")
    print(f"   ✅ Test examples: {len(split['test'])}")
    print(f"   Sample: '{split['train'][0]['input_text'][:50]}...' → '{split['train'][0]['target_text']}'")
    
    # Tokenize
    def tokenize_fn(ex):
        """Tokenize input and target text."""
        inputs = tokenizer(ex["input_text"], max_length=MAX_INPUT, truncation=True)
        labels = tokenizer(ex["target_text"], max_length=MAX_OUTPUT, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs
    
    print("3. Tokenizing...")
    tokenized = split.map(tokenize_fn, remove_columns=split["train"].column_names)
    
    # Load model and apply LoRA
    print(f"4. Loading {MODEL_NAME} and applying LoRA...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        task_type="SEQ_2_SEQ_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False # type: ignore
    
    print("\n Trainable Parameters:")
    model.print_trainable_parameters()
    
    # Prepare training
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=15,
        per_device_train_batch_size=16,
        learning_rate=3e-4,
        warmup_steps=100,
        logging_steps=50,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=MAX_OUTPUT,
        report_to="none",
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"], # type: ignore
        data_collator=collator
    )
    
    # Train
    print("\n" + "="*80)
    print("5. TRAINING")
    print("="*80)
    print("Expected: Loss should drop from ~8 to <0.5")
    print("="*80 + "\n")
    
    trainer.train()
    
    print("\n✅ Training complete!")
    
    # Show final metrics
    if trainer.state.log_history:
        final_metrics = trainer.state.log_history[-1]
        if 'loss' in final_metrics:
            print(f"   Final training loss: {final_metrics['loss']:.4f}")
        if 'eval_loss' in final_metrics:
            print(f"   Final eval loss: {final_metrics['eval_loss']:.4f}")
    
    # Test on examples
    print("\n" + "="*80)
    print("6. QUICK TEST")
    print("="*80)
    
    model.eval()
    device = next(model.parameters()).device
    
    def test(text):
        """Quick inference test."""
        inp = tokenizer(
            f"Is this about food or drinks? {text[:300]}",
            return_tensors="pt",
            max_length=MAX_INPUT,
            truncation=True
        )
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_length=MAX_OUTPUT)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    
    test_cases = [
        "Package with zucchini and carrots",
        "I ate pizza for lunch",
        "Picture of my laptop"
    ]
    
    print("\nSample predictions:")
    for t in test_cases:
        result = test(t)
        print(f"  {t[:45]:45s} → {result}")
    
    # Save model
    print("\n" + "="*80)
    print(f"7. Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n🎉 All done! Model saved and ready to use.")
    print("\nNext steps:")
    print("  • Run 'python evaluate.py' to verify fine-tuning worked")
    print("  • Run 'python inference.py' for more examples")
    print("="*80)


if __name__ == "__main__":
    main()