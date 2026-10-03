# Tunisian Sign Language (TSL) — Complete Sign Catalog

> **57 signs** across 5 categories.  
> Generated from the training dataset labels. Use this as the single source of truth for manual test references.

---

## Category: Demandes (24 signs)

| # | Label | Tunisian | French | English | Arabic | Supported |
|---|-------|----------|--------|---------|--------|-----------|
| 1 | 3aslema | aslema | bonjour | hello | اسلامة | YES |
| 2 | 5adamet | khademti | ma boisson | my drink | خدمتي | YES |
| 3 | assam | essa | poisson | fish | السمة | YES |
| 4 | barnamjk | barnamesh | programme | program | برنامش | YES |
| 5 | chabeb | chabeb | jeunesse | youth | شباب | YES |
| 6 | cv | cv | cv | resume | سي في | YES |
| 7 | demande | talab | demande | request | طلب | YES |
| 8 | enti | enti | toi (fem.) | you (f) | انتي | YES |
| 9 | labes | labes | bien | well | مخيبش | YES |
| 10 | lyoum | lyoum | aujourd'hui | today | اليوم | YES |
| 11 | mar7ba | marhba | bienvenue | welcome | مرحبا | YES |
| 12 | n3awnek | n3awnek | je t'aide | I help you | نعاونك | YES |
| 13 | nekteblk | nekteblik | je t'ecris | I write you | نكتبلك | YES |
| 14 | non | non | non | no | لا | YES |
| 15 | oui | oui | oui | yes | ايـه | YES |
| 16 | radio | radio | radio | radio | راديو | YES |
| 17 | se7a | seha | sante | health | صحة | YES |
| 18 | siye7a | siyaha | tourisme | tourism | سياحة | YES |
| 19 | t7eb | t7eb | tu veux | you want | تحب | YES |
| 20 | ta3lim | ta3lim | education | education | تعليم | YES |
| 21 | ta3raf | ta3ref | tu connais | you know | تعرف | **NO** (MediaPipe fail) |
| 22 | ta9ra | ta9ra | tu lis | you read | تقرا | YES |
| 23 | telvza | talvza | television | television | تلفزة | YES |
| 24 | tha9afa | tha9afa | culture | culture | ثقافة | YES |

## Category: Destinations (8 signs)

| # | Label | Tunisian | French | English | Arabic | Supported |
|---|-------|----------|--------|---------|--------|-----------|
| 25 | baladya | baladiya | mairie | town hall | بلدية | YES |
| 26 | banka | banka | banque | bank | بانك | YES |
| 27 | bousta | bosta | poste | post office | بوستة | YES |
| 28 | dar | dar | maison | house | دار | YES |
| 29 | ma7kma | ma7kma | tribunal | court | محكمة | YES |
| 30 | mostawsaf | mostawsaf | hopital | clinic | مستوصف | YES |
| 31 | sbitar | sbitar | hopital | hospital | سبيطار | YES |
| 32 | wzara | wzara | ministere | ministry | وزارة | YES |

## Category: Famille (12 signs)

| # | Label | Tunisian | French | English | Arabic | Supported |
|---|-------|----------|--------|---------|--------|-----------|
| 33 | 3ayla | 3ayla | famille | family | عايلها | YES |
| 34 | 5al-3am | khal 3am | oncle | uncle | خال عم | YES |
| 35 | 5ou | ekhwa | freres | brothers | اخو | YES |
| 36 | bent | bent | fille | girl | بنت | YES |
| 37 | bou | ab | pere | father | ابو | YES |
| 38 | eben | iben | fils | son | ابن | YES |
| 39 | jad | jed | grand-pere | grandfather | جاد | YES |
| 40 | jadda | jedda | grand-mere | grandmother | جدة | YES |
| 41 | mar2a | mra | femme | woman | مرة | YES |
| 42 | o5t | okht | soeur | sister | اخت | YES |
| 43 | om | om | mere | mother | ام | YES |
| 44 | tfol | tefl | enfant | child | طفل | YES |

## Category: Jours (7 signs)

| # | Label | Tunisian | French | English | Arabic | Supported |
|---|-------|----------|--------|---------|--------|-----------|
| 45 | 5mis | khmis | jeudi | Thursday | الخميس | YES |
| 46 | a7ad | ahad | dimanche | Sunday | الاحد | YES |
| 47 | erb3a | arba3 | mercredi | Wednesday | الاربعاء | YES |
| 48 | jom3a | jom3a | vendredi | Friday | الجمعة | YES |
| 49 | sebt | sebt | samedi | Saturday | السبت | YES |
| 50 | thleth | thleth | mardi | Tuesday | الثلاث | YES |
| 51 | thnin | thnin | lundi | Monday | الاثنين | YES |

## Category: Transport (6 signs)

| # | Label | Tunisian | French | English | Arabic | Supported |
|---|-------|----------|--------|---------|--------|-----------|
| 52 | car | karhba | voiture | car | كار | YES |
| 53 | karhba | karhba | voiture | car | كرهبة | **NO** (MediaPipe fail) |
| 54 | louage | louage | louage | minibus taxi | لواج | YES |
| 55 | metro | metro | metro | subway | ميترو | **NO** (MediaPipe fail) |
| 56 | taxi | taxi | taxi | taxi | طاكسي | YES |
| 57 | train | train | train | train | طارو | **NO** (MediaPipe fail) |

---

## Unsupported Classes

4 signs are currently **unsupported** by the MediaPipe hand-landmark pipeline and require a different model or approach:

| # | Label | Reason |
|---|-------|--------|
| 21 | ta3raf | MediaPipe fail |
| 53 | karhba | MediaPipe fail |
| 55 | metro | MediaPipe fail |
| 57 | train | MediaPipe fail |

---

## "car" vs "karhba" — Two Distinct Signs

Both `car` (#52) and `karhba` (#53) translate to "voiture/car" in French/English. However, they are **different signs** in TSL:

- **car** (#52) — supported, one of the 53 working classes.
- **karhba** (#53) — unsupported, fails with MediaPipe.

Treat them as separate classes when evaluating model performance.

---

## Reference Images

The following reference images exist in this directory (`manual_tests/reference/signs/`):

| File | Sign |
|------|------|
| `non.jpg` | non (no) |
| `labes.jpg` | labes (well) |
| `5ou.jpg` | 5ou (brothers) |

Use these for visual comparison when validating predicted signs against ground truth.

---

*Total signs: 57 — Supported: 53 — Unsupported: 4*
