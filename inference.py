"""
Simple inference examples for the food classifier.

Shows how to use the fine-tuned model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def load_model(model_path="./food_classifier"):
    """Load the fine-tuned model."""
    print("Loading model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def classify(text, model, tokenizer):
    """
    Classify if text is about food/drinks.
    
    Args:
        text: Input text to classify
        model: Fine-tuned model
        tokenizer: Tokenizer
        
    Returns:
        str: "yes" or "no"
    """
    prompt = f"Is this about food or drinks? {text[:300]}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=10)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # Load model
    model, tokenizer = load_model()
    
    # Example texts
    examples = [
        
        # New Menu Test
        "The sashimi was served with fresh wasabi.", # does it know wasabi is a food item ?
        "I grabbed a protein bar before the gym.", # does it recognise modern processed food ?
        "We ordered a Margherita with extra basil.", # does it know margherita is a food ?

        # Tech trap (context aware ??)
        "I bought a new MacBook Pro at the Apple Store", # does it know Apple is a company here and not a fruit ?
        "I am running a Raspberry Pi to host my server.", # does it know Raspberry Pi is a computer and not a fruit here ?
        "The Apple was overpriced and had a bad battery.", # does it know Apple is a company here and not a fruit ? (ambiguous statement) 
        "I went to the store to buy a toaster.", # does it know toaster is a food item or a kitchen appliance ? (ambiguous statement)
    ]
    
    print("\n" + "="*80)
    print("FOOD CLASSIFIER - INFERENCE EXAMPLES")
    print("="*80 + "\n")
    
    for text in examples:
        result = classify(text, model, tokenizer)
        icon = "🍕" if result.lower() == "yes" else "💻"
        print(f"{icon} {text}")
        print(f"   → {result}")
        print()
    
    print("="*80)


if __name__ == "__main__":
    main()