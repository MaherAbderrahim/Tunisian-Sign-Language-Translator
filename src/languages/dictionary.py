"""
TSL Dictionary - Mapping between Tunisian Sign Language (W7CH dataset) and text

Labels are aligned with the actual W7CH dataset folder names:
  data/raw/tsl_dataset/data/Data/<Category>/<class>/

Categories (57 classes total):
  - Demandes     (24): 3aslema, 5adamet, assam, barnamjk, chabeb, cv, demande,
                       enti, labes, lyoum, mar7ba, n3awnek, nekteblk, non, oui,
                       radio, se7a, siye7a, t7eb, ta3lim, ta3raf, ta9ra,
                       telvza, tha9afa
  - Destinations (8):  baladya, banka, bousta, dar, ma7kma, mostawsaf, sbitar, wzara
  - Famille      (12): 3ayla, 5al-3am, 5ou, bent, bou, eben, jad, jadda,
                       mar2a, o5t, om, tfol
  - Jours        (7):  5mis, a7ad, erb3a, jom3a, sebt, thleth, thnin
  - Transport    (6):  car, karhba, louage, metro, taxi, train

Languages: tunisian (dialect), french, english, arabic (tunisian-spelled),
           arabic_msa (Modern Standard Arabic — formal, written form).
"""

from typing import Dict, List, Optional


# Single source of truth: TSL label -> translations + category.
# `arabic` = Tunisian-spelled form (as recorded in the W7CH dataset).
# `arabic_msa` = Modern Standard Arabic (formal, pan-Arab) — added so the
# project supports non-Tunisian Arabic comprehension too.
# Tuned to the W7CH dataset folder names (case-sensitive matches the folders).
_TSL_ENTRIES = {
    # ===== Demandes (24) =====
    "3aslema":   {"tunisian": "aslema",     "french": "bonjour",      "english": "hello",      "arabic": "اسلامة",    "arabic_msa": "السلام عليكم",  "category": "Demandes"},
    "5adamet":   {"tunisian": "khademti",   "french": "ma boisson",   "english": "my drink",   "arabic": "خدمتي",    "arabic_msa": "مشروبي",        "category": "Demandes"},
    "assam":     {"tunisian": "essa",       "french": "poisson",      "english": "fish",       "arabic": "السمة",    "arabic_msa": "السمك",         "category": "Demandes"},
    "barnamjk":  {"tunisian": "barnamesh",  "french": "programme",    "english": "program",    "arabic": "برنامش",     "arabic_msa": "البرنامج",      "category": "Demandes"},
    "chabeb":    {"tunisian": "chabeb",     "french": "jeunesse",      "english": "youth",      "arabic": "شباب",     "arabic_msa": "الشباب",        "category": "Demandes"},
    "cv":        {"tunisian": "cv",         "french": "cv",            "english": "resume",      "arabic": "سي في",     "arabic_msa": "السيرة الذاتية", "category": "Demandes"},
    "demande":   {"tunisian": "talab",     "french": "demande",       "english": "request",    "arabic": "طلب",     "arabic_msa": "الطلب",         "category": "Demandes"},
    "enti":      {"tunisian": "enti",       "french": "toi (fém.)",    "english": "you (f)",    "arabic": "انتي",     "arabic_msa": "أنتِ",          "category": "Demandes"},
    "labes":     {"tunisian": "labes",      "french": "bien",          "english": "well",       "arabic": "مخيبش",     "arabic_msa": "بخير",          "category": "Demandes"},
    "lyoum":     {"tunisian": "lyoum",      "french": "aujourd'hui",   "english": "today",      "arabic": "اليوم",     "arabic_msa": "اليوم",         "category": "Demandes"},
    "mar7ba":    {"tunisian": "marhba",     "french": "bienvenue",     "english": "welcome",    "arabic": "مرحبا",     "arabic_msa": "أهلاً",          "category": "Demandes"},
    "n3awnek":   {"tunisian": "n3awnek",    "french": "je t'aide",     "english": "I help you", "arabic": "نعاونك",     "arabic_msa": "أساعدك",        "category": "Demandes"},
    "nekteblk":  {"tunisian": "nekteblik", "french": "je t'écris",    "english": "I write you","arabic": "نكتبلك",     "arabic_msa": "أكتب لك",       "category": "Demandes"},
    "non":       {"tunisian": "non",        "french": "non",          "english": "no",         "arabic": "لا",       "arabic_msa": "لا",            "category": "Demandes"},
    "oui":       {"tunisian": "oui",        "french": "oui",          "english": "yes",        "arabic": "ايه",       "arabic_msa": "نعم",           "category": "Demandes"},
    "radio":     {"tunisian": "radio",      "french": "radio",        "english": "radio",      "arabic": "راديو",     "arabic_msa": "الراديو",       "category": "Demandes"},
    "se7a":      {"tunisian": "seha",       "french": "santé",        "english": "health",    "arabic": "صحة",       "arabic_msa": "الصحة",         "category": "Demandes"},
    "siye7a":    {"tunisian": "siyaha",     "french": "tourisme",      "english": "tourism",    "arabic": "سياحة",     "arabic_msa": "السياحة",       "category": "Demandes"},
    "t7eb":      {"tunisian": "t7eb",       "french": "tu veux",       "english": "you want",   "arabic": "تحب",       "arabic_msa": "تريد",          "category": "Demandes"},
    "ta3lim":    {"tunisian": "ta3lim",     "french": "éducation",     "english": "education",  "arabic": "تعليم",     "arabic_msa": "التعليم",       "category": "Demandes"},
    "ta3raf":    {"tunisian": "ta3ref",     "french": "tu connais",    "english": "you know",   "arabic": "تعرف",      "arabic_msa": "تعرف",          "category": "Demandes"},
    "ta9ra":     {"tunisian": "ta9ra",      "french": "tu lis",        "english": "you read",   "arabic": "تقرا",       "arabic_msa": "تقرأ",          "category": "Demandes"},
    "telvza":    {"tunisian": "talvza",     "french": "télévision",    "english": "television", "arabic": "تلفزة",     "arabic_msa": "التلفاز",       "category": "Demandes"},
    "tha9afa":   {"tunisian": "tha9afa",    "french": "culture",      "english": "culture",    "arabic": "ثقافة",     "arabic_msa": "الثقافة",       "category": "Demandes"},

    # ===== Destinations (8) =====
    "baladya":   {"tunisian": "baladiya",  "french": "mairie",        "english": "town hall",   "arabic": "بلدية",      "arabic_msa": "البلدية",       "category": "Destinations"},
    "banka":     {"tunisian": "banka",      "french": "banque",        "english": "bank",        "arabic": "بانك",      "arabic_msa": "البنك",          "category": "Destinations"},
    "bousta":   {"tunisian": "bosta",      "french": "poste",         "english": "post office", "arabic": "بوستة",       "arabic_msa": "مكتب البريد",    "category": "Destinations"},
    "dar":       {"tunisian": "dar",        "french": "maison",       "english": "house",       "arabic": "دار",       "arabic_msa": "المنزل",         "category": "Destinations"},
    "ma7kma":    {"tunisian": "ma7kma",     "french": "tribunal",      "english": "court",       "arabic": "محكمة",      "arabic_msa": "المحكمة",        "category": "Destinations"},
    "mostawsaf": {"tunisian": "mostawsaf",  "french": "hôpital",       "english": "clinic",      "arabic": "مستوصف",     "arabic_msa": "المستوصف",      "category": "Destinations"},
    "sbitar":    {"tunisian": "sbitar",     "french": "hôpital",       "english": "hospital",    "arabic": "سبيطار",     "arabic_msa": "المستشفى",        "category": "Destinations"},
    "wzara":     {"tunisian": "wzara",      "french": "ministère",     "english": "ministry",    "arabic": "وزارة",      "arabic_msa": "الوزارة",        "category": "Destinations"},

    # ===== Famille (12) =====
    "3ayla":     {"tunisian": "3ayla",      "french": "famille",       "english": "family",       "arabic": "عايلها",    "arabic_msa": "العائلة",       "category": "Famille"},
    "5al-3am":  {"tunisian": "khal 3am",   "french": "oncle",         "english": "uncle",        "arabic": "خال عم",    "arabic_msa": "الخال",         "category": "Famille"},
    "5ou":       {"tunisian": "ekhwa",       "french": "frères",       "english": "brothers",     "arabic": "اخو",      "arabic_msa": "الإخوة",         "category": "Famille"},
    "bent":      {"tunisian": "bent",        "french": "fille",         "english": "girl",         "arabic": "بنت",       "arabic_msa": "الفتاة",        "category": "Famille"},
    "bou":       {"tunisian": "ab",         "french": "père",         "english": "father",       "arabic": "ابو",       "arabic_msa": "الأب",          "category": "Famille"},
    "eben":      {"tunisian": "iben",        "french": "fils",         "english": "son",          "arabic": "ابن",       "arabic_msa": "الابن",         "category": "Famille"},
    "jad":       {"tunisian": "jed",         "french": "grand-père",   "english": "grandfather",  "arabic": "جاد",       "arabic_msa": "الجد",          "category": "Famille"},
    "jadda":     {"tunisian": "jedda",       "french": "grand-mère",   "english": "grandmother",  "arabic": "جدة",        "arabic_msa": "الجدة",          "category": "Famille"},
    "mar2a":     {"tunisian": "mra",         "french": "femme",        "english": "woman",        "arabic": "مرة",        "arabic_msa": "المرأة",        "category": "Famille"},
    "o5t":       {"tunisian": "okht",        "french": "sœur",         "english": "sister",       "arabic": "اخت",       "arabic_msa": "الأخت",         "category": "Famille"},
    "om":        {"tunisian": "om",         "french": "mère",         "english": "mother",       "arabic": "ام",        "arabic_msa": "الأم",           "category": "Famille"},
    "tfol":      {"tunisian": "tefl",        "french": "enfant",       "english": "child",        "arabic": "طفل",        "arabic_msa": "الطفل",         "category": "Famille"},

    # ===== Jours (7) =====
    "5mis":      {"tunisian": "khmis",       "french": "jeudi",        "english": "Thursday",   "arabic": "الخميس",     "arabic_msa": "الخميس",        "category": "Jours"},
    "a7ad":      {"tunisian": "ahad",        "french": "dimanche",      "english": "Sunday",     "arabic": "الاحد",      "arabic_msa": "الأحد",         "category": "Jours"},
    "erb3a":     {"tunisian": "arba3",       "french": "mercredi",      "english": "Wednesday",  "arabic": "الاربعاء",    "arabic_msa": "الأربعاء",       "category": "Jours"},
    "jom3a":     {"tunisian": "jom3a",       "french": "vendredi",      "english": "Friday",     "arabic": "الجمعة",      "arabic_msa": "الجمعة",         "category": "Jours"},
    "sebt":      {"tunisian": "sebt",        "french": "samedi",        "english": "Saturday",    "arabic": "السبت",      "arabic_msa": "السبت",          "category": "Jours"},
    "thleth":    {"tunisian": "thleth",      "french": "mardi",         "english": "Tuesday",    "arabic": "الثلاث",      "arabic_msa": "الثلاثاء",        "category": "Jours"},
    "thnin":     {"tunisian": "thnin",       "french": "lundi",         "english": "Monday",     "arabic": "الاثنين",      "arabic_msa": "الاثنين",         "category": "Jours"},

    # ===== Transport (6) =====
    # NOTE: W7CH folder is "m\x8etro" (Latin-1 mojibake of "métro"); we also accept "metro".
    "metro":     {"tunisian": "metro",       "french": "métro",        "english": "subway",     "arabic": "ميترو",      "arabic_msa": "المترو",        "category": "Transport"},
    "car":       {"tunisian": "karhba",       "french": "voiture",      "english": "car",         "arabic": "كار",        "arabic_msa": "السيارة",       "category": "Transport"},
    "karhba":    {"tunisian": "karhba",       "french": "voiture",      "english": "car",         "arabic": "كرهبة",       "arabic_msa": "السيارة",       "category": "Transport"},
    "louage":    {"tunisian": "louage",       "french": "louage",       "english": "minibus taxi","arabic": "لواج",      "arabic_msa": "الحافلة المشتركة","category": "Transport"},
    "taxi":      {"tunisian": "taxi",         "french": "taxi",         "english": "taxi",       "arabic": "طاكسي",      "arabic_msa": "سيارة الأجرة",   "category": "Transport"},
    "train":     {"tunisian": "train",        "french": "train",        "english": "train",      "arabic": "طارو",       "arabic_msa": "القطار",        "category": "Transport"},
}


def _build_canonical_entries():
    """Add a 'metro' alias for the mojibake W7CH folder name."""
    entries = dict(_TSL_ENTRIES)
    return entries


class TSLDictionary:
    """
    Dictionnaire pour la Langue des Signes Tunisienne (TSL)
    Mappe les labels TSL vers le dialecte tunisien + français + anglais + arabe.

    Les labels correspondent exactement aux noms des dossiers du dataset W7CH
    (data/raw/tsl_dataset/data/Data/<Category>/<label>/).
    """

    def __init__(self):
        entries = _build_canonical_entries()

        # Stable, deterministic ordering: by category, then label.
        # This IS the class-order contract used by the model checkpoint.
        self._labels: List[str] = sorted(
            entries.keys(),
            key=lambda label: (entries[label]["category"], label),
        )

        # Build lookup tables
        self.tsl_to_tunisian: Dict[str, str] = {
            label: entries[label]["tunisian"] for label in self._labels
        }
        self.tsl_to_category: Dict[str, str] = {
            label: entries[label]["category"] for label in self._labels
        }
        self.tunisian_to_tsl: Dict[str, str] = {
            v.lower(): k for k, v in self.tsl_to_tunisian.items()
        }
        self.tunisian_to_french: Dict[str, str] = {
            entries[label]["tunisian"]: entries[label]["french"]
            for label in self._labels
        }
        self.tunisian_to_english: Dict[str, str] = {
            entries[label]["tunisian"]: entries[label]["english"]
            for label in self._labels
        }
        self.tunisian_to_arabic: Dict[str, str] = {
            entries[label]["tunisian"]: entries[label]["arabic"]
            for label in self._labels
        }

        # Also map tsl label directly to french/english/arabic (so we can
        # translate even when the tunisian form differs slightly).
        self.tsl_to_french: Dict[str, str] = {
            label: entries[label]["french"] for label in self._labels
        }
        self.tsl_to_english: Dict[str, str] = {
            label: entries[label]["english"] for label in self._labels
        }
        self.tsl_to_arabic: Dict[str, str] = {
            label: entries[label]["arabic"] for label in self._labels
        }
        self.tsl_to_arabic_msa: Dict[str, str] = {
            label: entries[label].get("arabic_msa", entries[label]["arabic"])
            for label in self._labels
        }
        self.tunisian_to_arabic_msa: Dict[str, str] = {
            entries[label]["tunisian"]: entries[label].get("arabic_msa", entries[label]["arabic"])
            for label in self._labels
        }

        # Alias map: dataset-folder-name -> canonical TSL label.
        # The W7CH "Transport/métro" folder is stored as "m\x8etro" (latin-1
        # mojibake). We map it to the canonical "metro" entry.
        self.dataset_folder_aliases: Dict[str, str] = {
            "metro": "metro",
            "m\x8etro": "metro",
            "m\xe9tro": "metro",
            "m\xe9tro".encode("latin-1").decode("utf-8", errors="ignore"): "metro",
        }

    def canonical_label(self, folder_name: str) -> Optional[str]:
        """Convert a dataset folder name to the canonical TSL label.

        Handles the W7CH "métro" mojibake folder and any other aliases.
        Returns None if the folder doesn't correspond to a known label.
        """
        if folder_name in self.dataset_folder_aliases:
            return self.dataset_folder_aliases[folder_name]
        if folder_name in self.tsl_to_tunisian:
            return folder_name
        return None

    def get_tunisian(self, tsl_label: str) -> Optional[str]:
        """Get Tunisian dialect from TSL label"""
        return self.tsl_to_tunisian.get(tsl_label)

    def get_french(self, tunisian: str) -> Optional[str]:
        """Get French translation from Tunisian dialect"""
        return self.tunisian_to_french.get(tunisian)

    def get_english(self, tunisian: str) -> Optional[str]:
        """Get English translation from Tunisian dialect"""
        return self.tunisian_to_english.get(tunisian)

    def get_arabic(self, tunisian: str) -> Optional[str]:
        """Get Arabic (Tunisian-spelled) translation from Tunisian dialect"""
        return self.tunisian_to_arabic.get(tunisian)

    def get_arabic_msa(self, tsl_label: str) -> Optional[str]:
        """Get Modern Standard Arabic (MSA) translation from a TSL label.

        MSA is the formal, written form of Arabic understood across the Arab
        world — useful when the audience is non-Tunisian.
        """
        return self.tsl_to_arabic_msa.get(tsl_label)

    def get_category(self, tsl_label: str) -> Optional[str]:
        """Get the category (Demandes/Destinations/...) of a TSL label"""
        return self.tsl_to_category.get(tsl_label)

    def get_all_translations(self, tsl_label: str) -> Dict[str, str]:
        """Get all translations for a TSL label"""
        if tsl_label not in self.tsl_to_tunisian:
            return {}
        tunisian = self.tsl_to_tunisian[tsl_label]
        return {
            "tsl": tsl_label,
            "tunisian": tunisian,
            "french": self.tsl_to_french.get(tsl_label) or self.get_french(tunisian),
            "english": self.tsl_to_english.get(tsl_label) or self.get_english(tunisian),
            "arabic": self.tsl_to_arabic.get(tsl_label) or self.get_arabic(tunisian),
            "arabic_msa": self.tsl_to_arabic_msa.get(tsl_label),
            "category": self.tsl_to_category.get(tsl_label),
        }

    def get_all_labels(self) -> List[str]:
        """Get all TSL labels in canonical (model class) order"""
        return list(self._labels)

    def get_all_tunisian_words(self) -> List[str]:
        """Get all Tunisian dialect words (in label order)"""
        return [self.tsl_to_tunisian[label] for label in self._labels]

    def get_labels_by_category(self, category: str) -> List[str]:
        """Get all labels belonging to a category"""
        return [l for l in self._labels if self.tsl_to_category.get(l) == category]

    def __len__(self) -> int:
        """Get number of entries"""
        return len(self._labels)

    def __contains__(self, label: str) -> bool:
        """Check if label exists"""
        return label in self.tsl_to_tunisian
