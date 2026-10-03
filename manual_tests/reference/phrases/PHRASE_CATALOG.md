# Phrase Catalog — 15 Test Phrases (Supported Signs Only)

All phrases use ONLY the 53 supported signs. No unsupported signs.

---

## Level 1 — Single Sign (5 phrases)

### Phrase 01
- **Sign sequence:** oui
- **Tunisian meaning:** oui
- **French:** oui
- **Expected raw output:** `oui`
- **Expected LLM output:** `Oui.`

### Phrase 02
- **Sign sequence:** non
- **Tunisian meaning:** non
- **French:** non
- **Expected raw output:** `non`
- **Expected LLM output:** `Non.`

### Phrase 03
- **Sign sequence:** 3aslema
- **Tunisian meaning:** bonjour
- **French:** bonjour
- **Expected raw output:** `3aslema`
- **Expected LLM output:** `Bonjour.`

### Phrase 04
- **Sign sequence:** mar7ba
- **Tunisian meaning:** bienvenue
- **French:** bienvenue
- **Expected raw output:** `mar7ba`
- **Expected LLM output:** `Bienvenue.`

### Phrase 05
- **Sign sequence:** labes
- **Tunisian meaning:** bien (comment ca va)
- **French:** bien
- **Expected raw output:** `labes`
- **Expected LLM output:** `Bien.`

---

## Level 2 — Two Signs (5 phrases)

### Phrase 06
- **Sign sequence:** oui, labes
- **Tunisian meaning:** oui, ca va bien
- **French:** oui, ca va bien
- **Expected raw output:** `oui labes`
- **Expected LLM output:** `Oui, ca va bien.`

### Phrase 07
- **Sign sequence:** enti, labes
- **Tunisian meaning:** toi, ca va bien
- **French:** toi, ca va bien
- **Expected raw output:** `enti labes`
- **Expected LLM output:** `Toi, ca va bien.`

### Phrase 08
- **Sign sequence:** lyoum, sebt
- **Tunisian meaning:** aujourd'hui samedi
- **French:** aujourd'hui samedi
- **Expected raw output:** `lyoum sebt`
- **Expected LLM output:** `Aujourd'hui samedi.`

### Phrase 09
- **Sign sequence:** n3awnek, non
- **Tunisian meaning:** je t'aide, non
- **French:** je t'aide, non
- **Expected raw output:** `n3awnek non`
- **Expected LLM output:** `Je t'aide, non.`

### Phrase 10
- **Sign sequence:** mar7ba, 3ayla
- **Tunisian meaning:** bienvenue famille
- **French:** bienvenue famille
- **Expected raw output:** `mar7ba 3ayla`
- **Expected LLM output:** `Bienvenue famille.`

---

## Level 3 — Three or More Signs (5 phrases)

### Phrase 11
- **Sign sequence:** 3aslema, labes
- **Tunisian meaning:** bonjour, ca va bien
- **French:** bonjour, ca va bien
- **Expected raw output:** `3aslema labes`
- **Expected LLM output:** `Bonjour, ca va bien.`

### Phrase 12
- **Sign sequence:** lyoum, thnin, ta9ra
- **Tunisian meaning:** aujourd'hui lundi tu lis
- **French:** aujourd'hui lundi tu lis
- **Expected raw output:** `lyoum thnin ta9ra`
- **Expected LLM output:** `Aujourd'hui lundi tu lis.`

### Phrase 13
- **Sign sequence:** bou, om, 3ayla
- **Tunisian meaning:** pere mere famille
- **French:** pere mere famille
- **Expected raw output:** `bou om 3ayla`
- **Expected LLM output:** `Pere mere famille.`

### Phrase 14
- **Sign sequence:** enti, t7eb, siye7a
- **Tunisian meaning:** toi tu veux tourisme
- **French:** toi tu veux tourisme
- **Expected raw output:** `enti t7eb siye7a`
- **Expected LLM output:** `Toi tu veux tourisme.`

### Phrase 15
- **Sign sequence:** jad, jadda, 3ayla, chabeb
- **Tunisian meaning:** grand-pere grand-mere famille jeunesse
- **French:** grand-pere grand-mere famille jeunesse
- **Expected raw output:** `jad jadda 3ayla chabeb`
- **Expected LLM output:** `Grand-pere grand-mere famille jeunesse.`

---

## Notes
- All signs in these phrases are from the 53 supported signs
- Raw output is the space-separated sign labels from the model
- LLM output is the French sentence after LLM post-processing
- The LLM may add punctuation and capitalization automatically
