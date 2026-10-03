"""
Demo script for TSL Translator
"""

import sys
import io
sys.path.insert(0, '.')

# Fix Windows console encoding for Arabic/Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.languages.dictionary import TSLDictionary
from src.translation.translator import translate_tsl_to_text, translate_text_to_tsl


def demo_dictionary():
    """Demo dictionary functionality"""
    print("="*60)
    print("TSL Dictionary Demo (W7CH dataset labels)")
    print("="*60)
    
    dictionary = TSLDictionary()
    
    print(f"\nTotal TSL signs: {len(dictionary)}")
    cats = {}
    for label in dictionary.get_all_labels():
        c = dictionary.get_category(label)
        cats[c] = cats.get(c, 0) + 1
    print("Categories:", ", ".join(f"{c} ({n})" for c, n in sorted(cats.items())))
    
    # Test translations using W7CH-aligned labels
    test_cases = [
        ("labes", "Bien"),
        ("non", "Non"),
        ("bou", "Pere"),
        ("dar", "Maison"),
        ("thnin", "Lundi"),
        ("car", "Voiture"),
    ]
    
    print("\n" + "-"*60)
    print("Example Translations:")
    print("-"*60)
    
    for tsl_label, expected in test_cases:
        result = dictionary.get_all_translations(tsl_label)
        if result:
            print(f"\nTSL: {tsl_label}  [{result.get('category', '')}]")
            print(f"  Tunisian: {result['tunisian']}")
            print(f"  French:   {result['french']}")
            print(f"  English:  {result['english']}")
        else:
            print(f"\nTSL: {tsl_label} -> NOT FOUND")


def demo_text_to_tsl():
    """Demo text to TSL translation"""
    print("\n" + "="*60)
    print("Text to TSL Translation Demo")
    print("="*60)
    
    # W7CH-aligned Tunisian words
    test_words = ["labes", "non", "ab", "dar", "ahad", "karhba"]
    
    for word in test_words:
        tsl_label = translate_text_to_tsl(word)
        if tsl_label:
            print(f"{word} -> {tsl_label}")
        else:
            print(f"{word} -> Not found")
    

def demo_tsl_to_text():
    """Demo TSL to text translation"""
    print("\n" + "="*60)
    print("TSL to Text Translation Demo")
    print("="*60)
    
    test_labels = ["labes", "non", "bou", "dar", "thnin"]
    
    for label in test_labels:
        result = translate_tsl_to_text(label)
        if result:
            print(f"{label} -> {result['tunisian']} ({result['french']})")


def main():
    """Main demo function"""
    print("\n" + "*"*60)
    print("  TSL TRANSLATOR - DEMONSTRATION")
    print("*"*60)
    
    demo_dictionary()
    demo_text_to_tsl()
    demo_tsl_to_text()
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nTo run the full application:")
    print("  - Web interface: streamlit run web/app.py")
    print("  - API server: uvicorn api.main:app --reload")
    print("  - Camera demo: python start.py")


if __name__ == "__main__":
    main()
