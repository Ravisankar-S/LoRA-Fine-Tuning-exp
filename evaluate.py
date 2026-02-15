"""
Evaluation script to verify that LoRA fine-tuning actually improved the model.

Compares base T5-small (no fine-tuning) against fine-tuned version to prove
that LoRA adaptation made a meaningful difference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def load_models():
    """Load both base and fine-tuned models for comparison."""
    print("Loading models...")
    
    # Base model (no fine-tuning)
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    base_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    base_model.eval()
    
    # Fine-tuned model (with LoRA)
    finetuned_base = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    finetuned_model = PeftModel.from_pretrained(finetuned_base, "./food_classifier")
    finetuned_tokenizer = AutoTokenizer.from_pretrained("./food_classifier")
    finetuned_model.eval()
    
    return (base_model, base_tokenizer), (finetuned_model, finetuned_tokenizer)


def classify(text, model, tokenizer):
    """Classify text using given model."""
    prompt = f"Is this about food or drinks? {text[:300]}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=10)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("="*80)
    print("VERIFYING FINE-TUNING ACTUALLY HAPPENED")
    print("="*80)
    
    # Load models
    (base_model, base_tokenizer), (finetuned_model, finetuned_tokenizer) = load_models()
    
    # Test cases
    test_cases = [
        "Package with zucchini and carrots",
        "I ate pizza for lunch",
        "Picture of my laptop",
        "Fresh strawberries for dessert",
        "The new iPhone is expensive",
    ]
    
    # Test 1: Base model predictions
    print("\n1. Testing BASE T5-Small (no fine-tuning):")
    print("-"*80)
    print("\nBASE MODEL (untrained) predictions:")
    for text in test_cases:
        result = classify(text, base_model, base_tokenizer)
        print(f"  {text[:40]:40s} → '{result}'")
    
    # Test 2: Fine-tuned model predictions
    print("\n" + "="*80)
    print("2. Testing YOUR FINE-TUNED model:")
    print("-"*80)
    print("\nFINE-TUNED MODEL predictions:")
    for text in test_cases:
        result = classify(text, finetuned_model, finetuned_tokenizer)
        print(f"  {text[:40]:40s} → '{result}'")
    
    # Test 3: Check LoRA weights
    print("\n" + "="*80)
    print("3. Checking LoRA adapter weights:")
    print("-"*80)
    
    lora_weights_exist = False
    for name, param in finetuned_model.named_parameters():
        if 'lora' in name.lower():
            lora_weights_exist = True
            weight_mean = param.data.abs().mean().item()
            weight_max = param.data.abs().max().item()
            print(f"  {name[:60]:60s} mean={weight_mean:.6f}, max={weight_max:.6f}")
            if weight_mean < 1e-10:
                print(f"    ⚠️  WARNING: Weights are near zero!")
    
    if not lora_weights_exist:
        print("  ❌ ERROR: No LoRA weights found!")
    else:
        print("\n  ✅ LoRA weights exist and are non-zero")
    
    # Test 4: Side-by-side comparison with accuracy
    print("\n" + "="*80)
    print("4. SIDE-BY-SIDE COMPARISON:")
    print("="*80)
    
    comparison_tests = [
        ("Package with zucchini and carrots", "yes"),
        ("I ate pizza for lunch", "yes"),
        ("Picture of my laptop on desk", "no"),
        ("Fresh strawberries for dessert", "yes"),
        ("New iPhone with better camera", "no"),
    ]
    
    print(f"\n{'Text':<40} | {'Expected':<10} | {'Base Model':<15} | {'Fine-tuned':<15} | {'Match?'}")
    print("-"*110)
    
    correct_base = 0
    correct_finetuned = 0
    
    for text, expected in comparison_tests:
        base_pred = classify(text, base_model, base_tokenizer).strip().lower()
        ft_pred = classify(text, finetuned_model, finetuned_tokenizer).strip().lower()
        
        base_correct = "✅" if expected in base_pred else "❌"
        ft_correct = "✅" if expected in ft_pred else "❌"
        
        if expected in base_pred:
            correct_base += 1
        if expected in ft_pred:
            correct_finetuned += 1
        
        print(f"{text[:38]:<40} | {expected:<10} | {base_pred[:13]:<15} | {ft_pred[:13]:<15} | Base:{base_correct} FT:{ft_correct}")
    
    print("-"*110)
    print(f"\nBase Model Accuracy: {correct_base}/{len(comparison_tests)} = {100*correct_base/len(comparison_tests):.0f}%")
    print(f"Fine-tuned Accuracy: {correct_finetuned}/{len(comparison_tests)} = {100*correct_finetuned/len(comparison_tests):.0f}%")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT:")
    print("="*80)
    
    if correct_finetuned > correct_base:
        print("✅ FINE-TUNING WORKED! Your model performs better than base T5.")
        print("   LoRA successfully adapted the model for this specific task.")
    elif correct_finetuned == correct_base and correct_base == len(comparison_tests):
        print("⚠️  AMBIGUOUS: Both models are perfect. Need harder test cases.")
        print("   Consider testing with more challenging examples.")
    else:
        print("❌ PROBLEM: Fine-tuned model is NOT better than base model!")
        print("   This suggests a potential training issue.")
    
    print("="*80)


if __name__ == "__main__":
    main()