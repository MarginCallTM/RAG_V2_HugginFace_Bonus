# Plan des bonus — RAG against the machine

## Reprise prioritaire sur Lenovo Ubuntu — 23 novembre 2026

**Lire cette section en premier : elle remplace les anciennes indications de
prochaine étape lorsqu'elles la contredisent.** Ce document sert de contexte
de reprise pour Codex sur la machine de 42. Le sujet officiel `subject.pdf`
version 2.0 et `tutor_agent_RAG_42.md` restent les références à lire.
`REPRISE_TUTEUR_LENOVO.md` date du 13 novembre et décrit un état bien antérieur :
ne pas en déduire que le mandatory ou les modules récents restent à écrire.

### 1. Point d'arrêt exact et méthode de travail

**Mise à jour après fusion hybride pondérée — 23 novembre 2026.** Ce bilan et la
checklist « Intégration et validation du bonus » remplacent les indications
historiques ci-dessous qui annoncent encore les tests ou la CLI à écrire.

- [x] Reproduire le petit index et le retriever sur Ubuntu CPU ; affichage des
  bornes corrigé. La recherche sur les wheels retrouve le bon fichier en premier.
- [x] Ajouter `build_semantic_index()` dans `src/semantic_index.py`.
- [x] Ajouter `SearchRetriever` (Protocol) et `load_retriever()` dans
  `src/retrieval.py` ; utiliser cette interface dans `search_questions()`.
- [x] Exposer `index_semantic`, puis `--method bm25|semantic|hybrid` et
  `--semantic_index_path` dans `search` et `search_dataset`. BM25 reste le défaut.
  `answer` reste BM25 ; `answer_dataset` peut consommer un JSON de sources
  sémantiques, mais aucune génération complète sémantique n'a été attestée ici.
- [x] Vérifier les tests dédiés : **26 passed** pour index + retriever
  (6 + 20), exécutés sans MiniLM. Flake8/mypy ciblés réussis lors des ajouts.
  L'élève rapporte ensuite **50 passed in 55.47s** pour datasets + CLI.
  Ces résultats précèdent la validation globale indiquée ci-dessous.
- [x] Construire l'index complet sur CPU : **16 736 chunks**, **980 lots**,
  batch_size=32, max_tokens=256 ; **306,22 s** au chronomètre interne de la
  commande, dont environ 4 min 57 s dans la barre d'encodage.
  Fichier : `data/processed/semantic_index.pt`. Le nombre exact de segments,
  la taille du fichier et la mémoire maximale restent à relever.
- [x] Rechercher les datasets complets et mesurer le recall sémantique à k=5.

| Méthode | Docs (100 questions) | Code (99 questions) |
|---|---:|---:|
| BM25, baseline | 81,00 % | 66,67 % |
| MiniLM, CPU, max par chunk | 66,00 % | 41,41 % |
| Hybride retenu, C=0, BM25/MiniLM 65/35 | **83,00 %** | **64,65 %** |

Sources sémantiques sauvegardées sous
`data/output/experiments/semantic/full/UnansweredQuestions/` dans les deux
fichiers `dataset_docs_public.json` et `dataset_code_public.json`.
La sortie docs transmise utilisait un seuil de 0,50 : son « PASS » ne signifie
pas que le seuil docs de 0,80 est atteint. Le code échoue au seuil de 0,50.
Ces résultats ne remplacent pas les scores mandatory BM25. L'égalité entre
recall@5 et recall@10 s'explique par les cinq sources maximum sauvegardées.

### État courant du bonus 2 : hybride pondéré

- `src/fusion.py` : RRF sur des listes de `ScoredChunk`, déduplication par
  chemin/bornes, une contribution par source et par méthode, égalités résolues
  par chemin puis bornes. Poids positifs finis facultatifs ; défaut 1 par liste.
- `src/hybrid_retriever.py` : deux retrievers réutilisés, jusqu'à
  `max(k, candidate_k)` candidats demandés à chacun, puis top-k fusionné.
  Le chargement par `src/retrieval.py` refuse des index lexical/sémantique
  contenant des listes de chunks différentes.
- Choix confirmé par l'élève : **candidate_k=20, rank_constant=0,
  lexical_weight=0.65, semantic_weight=0.35, k final=5** pour les évaluations.
  Ces paramètres sont fixés dans la branche hybride de `load_retriever()` ;
  ils ne sont pas encore exposés comme options CLI.
- `tests/test_fusion.py` : 20 cas ; `tests/test_hybrid_retriever.py` : 24 cas.
  Les **44 tests passent**, ainsi que flake8/mypy ciblés sur les trois modules.
  Vérifications supplémentaires exécutées sur exemples artificiels pour la
  propagation des poids non égaux et leur arithmétique ; leur couverture
  persistante dans pytest reste à compléter.
- Recherche hybride réelle validée via `search`, puis datasets complets sur CPU.
  La génération Qwen avec ces sources n'a pas encore été vérifiée.
- Vérification globale après intégration pondérée : **285 tests réussis en
  68,28 s**, flake8 réussi sur le projet, mypy réussi sur **77 fichiers** avec
  les flags mandatory. Aucun commit ni push effectué par l'assistant.

Expériences à candidats identiques (20 par méthode), moulinette officielle,
sources <=2000 caractères et k final=5 :

| Constante C, poids égaux | Docs recall@5 | Code recall@5 |
|---|---:|---:|
| 0 | 83,00 % | 62,63 % |
| 10 | 80,00 % | 62,63 % |
| 30 | 80,00 % | 60,61 % |
| 60 | 80,00 % | 60,61 % |
| 100 | 80,00 % | 60,61 % |

| Poids BM25/MiniLM, C=0 | Docs recall@5 | Code recall@5 |
|---|---:|---:|
| 50/50 | 83,00 % | 62,63 % |
| **65/35 retenu** | **83,00 %** | **64,65 %** |
| 80/20 | 81,00 % | 65,66 % |

Les deux seuils mandatory sont atteints par le réglage hybride retenu. Il
gagne 2 points docs sur BM25 mais perd 2,02 points code ; ne pas annoncer un
gain sur tous les jeux. Les réglages sont choisis sur les datasets publics,
pas validés sur un jeu indépendant. BM25 reste le défaut.

Artefacts ignorés par Git :
- `data/output/experiments/hybrid/rrf_sweep/summary.json` : constantes testées.
- `data/output/experiments/hybrid/weight_sweep_c0/summary.json` : poids testés.
- Résultats retenus :
  `data/output/experiments/hybrid/weight_sweep_c0/w65_35/UnansweredQuestions/`.
- Les logs `.evaluation.txt` sont à côté des JSON ; les candidats par méthode
  sont conservés sous `weight_sweep_c0/candidates/`.

**Prochain incrément :** clôturer la validation hybride par un petit batch
Qwen forcé sur CPU et quelques tests des poids/du chargement de méthode.
Puis commencer le bonus 3 : manifeste des fichiers et détection des changements
avant toute réindexation incrémentale. La comparaison exhaustive des succès
par question a été reportée au choix de l'élève, elle n'est pas un préalable
imposé par le sujet. Ne pas poursuivre une recherche illimitée de paramètres.
Le commit et le push restent effectués par l'élève.

**Historique antérieur à cette mise à jour** (les sections 2 à 7 suivantes
décrivent également la reprise initiale, pas l'état de fin d'intégration) :

- Le mandatory BM25/Qwen est implémenté. Le travail actuel porte sur le
  **bonus 1, recherche sémantique**, pas encore sur la recherche hybride.
- Développement récent et essais effectués sur le **MacBook M2**, en CPU
  float32. Le retour sur Lenovo Ubuntu commence aujourd'hui ; aucune nouvelle
  validation de ces bonus sur Lenovo n'est encore attestée.
- Nous venons de créer `src/semantic_retriever.py` et d'exécuter
  `notes/bonus/bonus_explore_semantic_retriever.py` sur un index de **trois
  chunks / six segments**. Le bon fichier arrive premier pour une question
  sur la taille maximale des wheels. Il n'y a encore **aucun recall sémantique
  docs/code**, ni indexation vectorielle du corpus complet, ni méthode
  sémantique accessible dans la CLI mandatory.
- **Prochaine séance :** vérifier la copie et l'environnement Lenovo,
  reproduire le petit exemple, corriger son affichage de bornes, consolider
  la compréhension du regroupement par chunk, puis écrire les tests du
  retriever sémantique sans charger MiniLM. Ensuite seulement : branchement
  datasets/CLI, préparation de l'index complet et mesures.
- Dernière question pédagogique non encore répondue : « Si le chunk 0 possède
  trois segments, pourquoi n'apparaît-il qu'une seule fois dans les résultats ? »
  Réponse à faire expliquer par l'élève : maximum des scores par identité de
  chunk, puis un seul `ScoredChunk` par chunk dans le classement final.
- Dernier commit local observé lors de cette relecture : `4c5b258`,
  `add semantic retriever to the codebase`. Précédents : `b15db22` (encodage et
  persistance), `ed2cf1e` (configuration et segmentation). Vérifier le HEAD réel
  sur Lenovo ; cette lecture ne prouve pas l'état distant ni la synchronisation.

**Préférences explicites de l'élève :** agir comme un professeur particulier,
en français ; expliquer la théorie et les nouvelles syntaxes **avant** le
code. Donner le chemin exact, un petit bloc à recopier, une commande et le
résultat attendu ; attendre sa sortie et l'analyser. L'élève tape le code
lui-même. Ne modifier directement un fichier que sur demande explicite
(une demande de mise à jour de ce document autorise ce document seulement).
Il a demandé d'intégrer dans l'ordre du parcours des données, et non de
commencer par la fonction de classement finale. Ne pas réécrire le projet
d'un bloc, ne pas ajouter d'agents/frameworks/services sans besoin.
Ne pas commit/push ou changer de branche sans demande. Expliquer toute
installation ou téléchargement volumineux avant de le lancer.

### 2. Architecture réelle à connaître

```text
Mandatory existant
Fichiers vLLM → ingest/chunking → ChunkIndex → chunks.json
Question → tokenizer lexical → Retriever BM25 → ScoredChunk → sources JSON
Sources → prompt borné avec tokenizer Qwen → Generator → réponse JSON

Bonus sémantique actuellement intégré, sur échantillon
ChunkIndex.chunks → SemanticEncoder.segment_chunks()
                 → liste ordonnée de TokenSegment
                 → encode_segments() par lots sur CPU
                 → vecteurs normalisés, une ligne par segment
Métadonnées + chunks + segments → SemanticIndexData
SemanticIndexData + vecteurs → validation → save_semantic_index()
                            → semantic_index.pt

SemanticRetriever(path) → load_semantic_index() + encodeur configuré
search(question, k) → encode_query() → (1, 384)
                    → produit matriciel avec tous les segments
                    → maximum des scores par chunk
                    → tri décroissant, top-k, list[ScoredChunk]
```

| Fichier | État et rôle exact |
|---|---|
| `src/models.py` | Contrats mandatory et `Chunk`, `ChunkIndex`, `ScoredChunk`. Un chunk conserve le texte et les bornes exactes ; `ScoredChunk` sert maintenant aux deux retrievers malgré sa docstring encore lexicale. |
| `src/ingest.py`, `src/chunking.py` | Lecture UTF-8 avec normalisation des fins de ligne, chemins relatifs ; Python par AST et replis, Markdown par sections/paragraphes, texte par paragraphes. Chunks de 2 000 caractères maximum via le pipeline. |
| `src/indexer.py` | `build_index()`, `save_index()`, `load_index()` pour le mandatory ; persiste `data/processed/chunks.json`, sans statistiques BM25 ni vecteurs. |
| `src/embeddings.py` | `EmbeddingConfig`, `TokenSegment`, `SemanticEncoder` ; chargement CPU, segmentation, encodage par lots, méthode commune `_encode_batch()` et `encode_query()`. |
| `src/semantic_index.py` | **Orthographe actuelle : `SemanticIndexMetaData`**, puis `SemanticIndexData`, `validate_embeddings()`, `save_semantic_index()`, `load_semantic_index()`. Ne pas inventer un fichier par classe. |
| `src/semantic_retriever.py` | `SemanticRetriever(index_path: Path)` ; `search(query: str, k: int = 5) -> list[ScoredChunk]`. Charge l'index et l'encodeur une fois par instance ; réutilisable sur un dataset. |
| `src/retriever.py` | Retriever BM25 inchangé, indépendant du sémantique. Paramètres k1=1.5, b=0.75 ; ne retourne que les scores strictement positifs. |
| `src/datasets.py` | `search_questions()` réutilise un retriever par batch mais l'annote encore comme `Retriever` BM25 ; aucune abstraction commune ni branchement sémantique ajouté. `save_search_results()` fournit le JSON mandatory. |
| `src/cli.py` | `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate` ; utilise toujours BM25. **Aucune option `--method` ni commande de construction de l'index sémantique n'existe actuellement.** |
| `src/evaluation.py` | Recall par question puis moyenne ; mêmes chemins et IoU strictement supérieure à 0,05 dans le code local ; moulinette exécutée à part, jamais importée ni appelée par l'application. |
| `src/prompting.py`, `src/generator.py` | Génération Qwen3-0.6B existante, budget applicatif par défaut 4096 avec 256 nouveaux tokens ; sources effectivement incluses conservées. Qwen n'est pas utilisé par le script de recherche sémantique. |

### 3. Choix techniques et invariants déjà compris/exercés

- Modèle : `sentence-transformers/all-MiniLM-L6-v2`, licence Apache-2.0 selon
  la fiche consultée ; révision par défaut **`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`**
  pour le tokenizer et les poids. Ce commit est une version Hugging Face,
  **pas un chemin Mac**. Les poids doivent exister ou être téléchargés sur Lenovo.
- Bibliothèques : Transformers 4.x et PyTorch déjà déclarés ; pas de nouvelle
  dépendance Sentence Transformers, FAISS, base vectorielle ou service cloud.
- `EmbeddingConfig` : modèle et révision, `max_tokens=256` (3..256),
  `batch_size=32` (>=1). Les expériences sur trois chunks choisissent 2.
- La capacité architecturale de MiniLM est 512 positions ; **256 est notre
  budget initial d'encodage**, cohérent avec l'usage étudié, pas une limite
  générale des LLM ni le budget Qwen. Deux tokens spéciaux → 254 tokens de
  contenu par segment. 2 000 caractères ne garantissent pas 256 tokens.
- Le découpage est guidé par les tokens, pas une détection de frontières de
  sens. `return_overflowing_tokens=True`, `stride=0`, pas de chevauchement ;
  chaque segment contient ses tokens spéciaux. Aucun contenu tokenisé n'est
  volontairement jeté à la fin comme avec une troncature simple.
- `TokenSegment` : `chunk_position`, `segment_position`, `input_ids`.
  Une ligne du tenseur correspond au même indice dans la liste des segments.
  Les indices sont relatifs à la liste de chunks enregistrée avec cet index.
- Pas de bornes de caractères propres aux segments : ceux-ci servent à la
  recherche ; **la source retournée est le chunk original entier**, <=2000
  caractères. Plusieurs segments du même chunk ont donc la même provenance
  affichée. Ne pas reconstruire les sources par `decode()` des tokens.
- Pooling : `last_hidden_state` → masque étendu avec `unsqueeze(-1)` → somme
  des vecteurs utiles / nombre de positions utiles ; `[CLS]` et `[SEP]`
  comptent, `[PAD]` non. Les espaces/tabulations ne sont pas ce padding.
- Normalisation L2 après pooling, `p=2, dim=1`. Un vecteur par segment existe
  **avant** normalisation ; celle-ci ajuste sa norme, pas le nombre de vecteurs.
  Coordonnées éventuellement négatives ; norme proche de 1 ; pas de probabilités.
- Encodage par lots : nombre de segments par appel au modèle, distinct du
  budget de tokens et du top-k. `_encode_batch()` effectue le calcul commun
  aux segments et questions. `batch_segments`, et non tous les segments, doit
  fournir les tokens de chaque tour. Tous les vecteurs finaux restent en RAM.
- Question : `encode_query()` rejette les espaces seuls et un texte dépassant
  le budget, sans troncature silencieuse ; retourne `(1, 384)`.
  Mesure faite sur les datasets publics Mac : maximum 58 tokens docs, 40 code,
  tokens spéciaux compris ; aucune des 199 questions ne dépasse 256. Pas une
  garantie sur de futures questions. `search()` retourne `[]` pour question
  vide/espaces seuls ou k=0 avant d'appeler l'encodeur.
- Recherche exacte : `(1, 384) @ (S, 384).T` donne `(1, S)` ; produit scalaire
  égal au cosinus car vecteurs normalisés. `S` = nombre de segments, pas de chunks.
- `search()` refuse k négatif ou non entier ; score non fini refusé ; conserve
  le maximum par chunk, trie par `(-score, chunk_position)` et retourne au plus
  k chunks uniques. Pas de seuil sémantique : un passage hors sujet peut être
  retourné s'il est parmi les meilleurs disponibles, même avec score négatif.
- Ce maximum par chunk **n'est pas** le bonus hybride. L'hybride combinera
  plus tard les classements BM25 et sémantique, par exemple via RRF. Un gain
  de recall@5 est une hypothèse à mesurer, pas une conséquence garantie.

### 4. Persistance, résultats observés et limites des preuves

`SemanticIndexMetaData` contient versions de format/préparation (1),
`encoder_config`, dimension, nombres de chunks/segments, `masked_mean`,
`l2` (**lettre L**, pas le nombre 12), `token_windows_no_overlap`, `float32`.
`SemanticIndexData` valide compteurs, indices de chunks, budget, numérotation
successive des segments et présence d'au moins un segment par chunk.

`validate_embeddings()` vérifie forme `(segment_count, dimension)`, CPU,
float32, valeurs finies et normes proches de 1 avec `rtol=1e-4`, `atol=1e-5`.
Ces contrôles ne prouvent pas l'identité sémantique des lignes : un échange
de deux vecteurs normalisés peut les passer. Préserver l'ordre et le tester.

Le fichier `.pt` contient `{"data": data.model_dump(mode="json"),
"embeddings": tensor}`. `mode="json"` prépare des types simples, il n'écrit
pas un fichier JSON. Écriture dans `.tmp`, puis remplacement du fichier final
(hypothèse d'un seul écrivain par destination). Chargement local de confiance
avec `map_location="cpu", weights_only=True`, reconstruction Pydantic et
revalidation du tenseur. Chemins fournis en `Path`, sans dépendance à `/Users/…`.

Échantillon réel sauvegardé sur Mac sous
`data/output/experiments/semantic/sample/semantic_index.pt` : **17 775 octets**,
3 chunks, 6 segments, `(6, 384)`, CPU float32, normes proches de 1. Aller-retour :
`Same index data: True`, `Same vectors: True`. Ce fichier est ignoré par Git.

| Chunk d'essai | Source originale | Segments, positions avant padding |
|---|---|---|
| 0 | `data/raw/vllm-0.10.1/.buildkite/check-wheel-size.py [0:1998]` | 256, 256, 109 |
| 1 | `data/raw/vllm-0.10.1/.buildkite/generate_index.py [0:718]` | 256, 6 |
| 2 | `data/raw/vllm-0.10.1/.buildkite/lm-eval-harness/configs/models-large.txt [0:192]` | 104 |

Dernière recherche réelle fournie par l'élève :
`Which environment variable sets the maximum allowed wheel size?`, k=3.
Scores maximums par chunk : `check-wheel-size.py` **0,4566**,
`generate_index.py` **0,1634**, `models-large.txt` **0,0358**.
Ces chiffres décrivent une question sur trois chunks, pas un recall, une
probabilité de vérité, une réponse générée ou un benchmark du corpus.

**Petite erreur encore présente dans le dépôt relu le 23 novembre :**
`notes/bonus/bonus_explore_semantic_retriever.py` imprime deux fois
`chunk.first_character_index`, ce qui affiche `[0:0]`. Le dernier f-string
doit utiliser `chunk.last_character_index`. Correction expliquée mais non
appliquée dans la copie inspectée. Les sources des objets ne sont pas `[0:0]` ;
c'est l'affichage qui est incorrect. La corriger avec l'élève avant la démo.

Dernières sorties de vérification **transmises par l'élève sur Mac** :

```text
uv run python -m pytest tests/test_semantic_index.py -q
4 passed in 2.34s

make lint
Success: no issues found in 71 source files
```

Les quatre tests couvrent vecteurs valides, vecteur nul, compteur de segments
incohérent et aller-retour avec deux vecteurs artificiels de dimension 2.
Ils ne chargent pas MiniLM et n'évaluent pas la pertinence. **Aucun fichier
`tests/test_semantic_retriever.py` ni test dédié de l'encodeur n'existe encore.**
Ne pas annoncer une suite complète verte ou un nombre total obtenu par addition
à l'ancienne baseline. La mise à jour documentaire du 23 novembre a relu le
code et le chapitre IX ; elle n'a relancé ni tests, ni modèles, ni benchmarks.

Le conseil `BertTokenizerFast` sur l'appel séparé à `pad()` n'est pas une erreur.
Ce découpage explicite a été choisi pour comprendre les opérations et réutiliser
les tokens déjà segmentés ; ne pas masquer les erreurs réelles avec ce message.

### 5. Reproduction sur Lenovo avant de poursuivre

**Exécuter depuis la racine réelle du clone**, pas depuis `notes/bonus/`.
Ne pas recopier l'environnement virtuel du Mac ni ses chemins absolus.

```bash
pwd
git status --short
git log -3 --oneline
python3 --version
uv --version
uname -m
free -h
df -h .
```

Le projet déclare Python `>=3.12,<3.13`. `pyproject.toml` conditionne l'index
PyTorch CUDA 13.0 à Linux/Windows et utilise l'index par défaut sur Mac.
**Une distribution PyTorch CUDA peut exécuter ces modules sur CPU** :
`SemanticEncoder` force explicitement CPU/float32. Ne pas supposer un GPU
présent ou nécessaire. L'installation Linux peut être volumineuse ; vérifier
place et accès réseau, expliquer les téléchargements et diagnostiquer une
éventuelle incompatibilité avant de changer les dépendances. Aucun backend
MPS n'est utilisé. Ne pas modifier le lockfile simplement à cause du transfert.

```bash
uv sync --locked
uv run python --version
uv run python -m pytest tests/test_semantic_index.py -q
make lint
```

Le sujet attend `uv sync` ; `--locked` sert ici à vérifier le lockfile transféré.
Les commandes précédentes ne prouvent leur réussite qu'après lecture des sorties.
`notes/__init__.py` et `notes/bonus/__init__.py` sont présents pour éviter
l'erreur mypy de double nom de module ; les conserver.

**Données absentes d'un clone Git :** `.gitignore` exclut corpus, datasets,
index, vecteurs, résultats et archives jointes. Sur Mac, leur restauration et
la concordance des 16 736 chunks avaient été vérifiées ; ne pas supposer leur
présence sur Lenovo. Vérifier avant toute extraction ou reconstruction :

```bash
ls data/raw/vllm-0.10.1/docs/features/lora.md
ls data/datasets/UnansweredQuestions/dataset_docs_public.json
ls data/datasets/UnansweredQuestions/dataset_code_public.json
ls data/datasets/AnsweredQuestions/dataset_docs_public.json
ls data/datasets/AnsweredQuestions/dataset_code_public.json
ls data/processed/chunks.json
```

Restaurer au besoin les pièces jointes fournies, dans l'arborescence du README,
en évitant le niveau intermédiaire `datasets_public/public/`. Garder **vLLM
0.10.1 fourni**, pas un clone récent : les sources attendues dépendent du texte
et du préfixe exact `data/raw/vllm-0.10.1/`. Pas de corpus ni de vecteurs à committer.
Les binaires de la moulinette sont fournis séparément pour Linux.

Si `chunks.json` manque, sa reconstruction mandatory est disponible :

```bash
uv run python -m src index --max_chunk_size 2000
```

Cette commande remplace l'index de chunks ; elle ne construit **pas** l'index
vectoriel. L'expliquer avant de remplacer un artefact local existant.

Après vérification des données et de la disponibilité de MiniLM, reproduire :

```bash
uv run python -m notes.bonus.bonus_explore_chunk_segments
uv run python -m notes.bonus.bonus_explore_query
uv run python -m notes.bonus.bonus_explore_semantic_retriever
```

La première commande réencode seulement les **trois premiers chunks** et
remplace le petit index sémantique d'essai ; les suivantes le rechargent sans
réencoder les chunks. Le modèle peut être téléchargé au premier usage sur
Lenovo (~90 Mo de poids, plus tokenizer/configuration), indépendamment du
cache Mac. Vérifier les conditions avec l'élève avant ce téléchargement.
Les scores peuvent légèrement varier avec matériel/bibliothèques ; les
associations, dimensions et normes doivent rester cohérentes.

Anciens prototypes `bonus_explore_encoder.py` / `bonus_explore_search.py` :
ils sauvegardent/rechargent des phrases sous `semantic/persistence/demo_index.pt`.
**Ne pas confondre cet ancien format de démonstration avec le format actuel**
`SemanticIndexData` sous `semantic/sample/semantic_index.pt`.

### 6. Ordre concret du travail restant

1. **Reprise Lenovo et compréhension :** étapes ci-dessus, correction du seul
   affichage `[0:0]`, puis question de compréhension sur maximum par chunk.
2. **Tests déterministes du retriever**, avant extension du périmètre : index
   artificiel via les fonctions de persistance, encodeur simulé au point
   `src.semantic_retriever.SemanticEncoder` pour éviter tout téléchargement.
   Vérifier calcul/classement, max par chunk, absence de doublons, provenance,
   égalités déterministes, scores négatifs, k=0/grand k/invalide, question vide,
   index vide, dimension incompatible et propagation d'une question trop longue.
   Ne pas tester seulement un tri de scores sans leurs sources.
3. **Compléter les tests ciblés encodeur/index** : segmentation et ordre sans
   vrai modèle lorsque possible, pooling masqué, lots et dernière tranche,
   questions vides/trop longues ; forme/dtype/NaN/inf, références invalides et
   mauvais fichiers à la lecture. Tester d'abord les risques réels, pas chaque
   ligne isolément. Les quatre tests existants sont un début, pas cette couverture.
4. **Comparer BM25 et sémantique sur un même petit corpus** ; examiner leurs
   complémentarités avant la fusion. Toujours distinguer score et preuve.
5. **Intégrer l'orchestration** dans l'ordre des données : préparation de
   l'index sémantique à partir du `ChunkIndex` complet (actuellement assemblée
   seulement par le script d'essai), chemins et paramètres configurables,
   réutilisation d'un retriever par batch. Choisir une interface commune légère
   pour `search_questions()` (ex. `Protocol` ou union adaptée), expliquer le
   choix puis préserver les contrats JSON. Ne pas ignorer les erreurs mypy.
6. **Brancher la CLI** : BM25 par défaut, option sémantique explicite,
   construction/rechargement de l'index, erreurs attendues traduites en `CliError`.
   Les options nouvelles ne sont pas encore définies : lire les signatures
   réelles et les concevoir avec l'élève, ne pas annoncer des commandes fictives.
   Même contrat de `ScoredChunk`/`MinimalSource`, sources <=2000 caractères.
7. **Construire l'index vectoriel complet et mesurer sur CPU** : nombre de
   segments, proportion de chunks dépassant 256 tokens, temps, taille et mémoire.
   Ne pas extrapoler les ~8–14 lots/s observés sur des exemples de quelques lots.
   Fixer une taille de lot mesurée ; la progression de segmentation du corpus
   reste à prévoir (actuellement seule la boucle d'encodage affiche tqdm).
8. **Évaluer docs et code à k=5** : mêmes datasets, mêmes chemins et mêmes
   références que BM25, sorties isolées par méthode/scope ; moulinette exécutée
   manuellement. Vérifier ensuite les sources réellement retenues par le prompt
   et une petite génération Qwen sur CPU, sans relancer inutilement 199 réponses.
9. **Bonus 2 hybride** : candidats BM25 + candidats sémantiques → fusion des
   rangs (RRF à expliquer/expérimenter), déduplication par chemin et bornes,
   top-k final, mêmes évaluations et tests. Ne pas additionner naïvement les
   scores d'échelles différentes. 81 % docs est la baseline BM25 historique,
   pas un résultat hybride ; l'hybride peut gagner ou perdre du recall@5.
10. **Bonus 3, 4 et 5** : indexation incrémentale, cache d'index ET de requêtes,
    API HTTP locale. Leurs checklists détaillées restent ci-dessous ; aucune
    de ces trois fonctionnalités n'est implémentée. La persistance actuelle
    ne valide pas à elle seule le bonus cache, ni la présence de MiniLM le
    bonus hybride. Puis documentation anglaise et validation finale sur Lenovo.

### 7. Proposition de message de reprise au tuteur Lenovo

« Lis `tutor_agent_RAG_42.md`, le sujet et la section de reprise du 23 novembre
dans `bonus_rag.md`, puis vérifie la copie réelle. Nous avons déjà intégré
l'encodeur CPU, la segmentation, l'index persistant validé et le retriever
sémantique sur trois chunks. Ne réimplémente pas ces étapes. Reproduisons
d'abord l'environnement et la petite démo sur Lenovo, corrigeons l'affichage
des bornes encore fautif, puis écrivons les tests du retriever avec un encodeur
simulé. Explique les notions avant le code ; je recopie moi-même les petits
incréments. Pas d'indexation vectorielle complète ni d'hybride encore mesurés. »

## Contrat et état de départ

Référence : `subject.pdf`, chapitre IX. Les cinq bonus valent chacun un point
et ne sont évalués qu'après validation complète du mandatory. Ils doivent
tous fonctionner sur une machine campus **CPU-only** et être démontrables.
Une description dans le README ne suffit pas.

Ce fichier est une feuille de route : aucune case non cochée ne représente
une fonctionnalité déjà implémentée. Les choix d'architecture ci-dessous sont
des propositions d'ingénierie, pas des obligations supplémentaires du sujet.

Mode de travail : une petite étape à la fois ; expliquer le concept, montrer
le code à recopier, laisser l'élève exécuter les tests, puis analyser sa sortie.
Ne pas modifier la codebase sans demande explicite.

Base mandatory observée : chunking Python et Markdown/texte, BM25, prompts
bornés en tokens, Qwen local, CLI Fire et échanges JSON Pydantic.
Mesures publiques enregistrées : recall@5 docs 81 %, code 66,67 % ; indexation
10,10 s ; recherche de 200 questions à k=5 en 12,03 s de temps total.
Ces mesures ne garantissent pas les mêmes résultats sur d'autres machines.

## 0. Figer une base de comparaison

- [x] Terminer le contrôle final du mandatory et enregistrer un commit de référence
      (`e8db1b6` — `Validate mandatory baseline and plan bonuses`).
- [x] Enregistrer les versions, la machine, les commandes, les scores et les temps
      (voir le bilan de référence ci-dessous et les commandes du README).
- [x] Garder BM25 accessible et préserver les signatures minimales du sujet.
- [x] Utiliser des dossiers de résultats séparés pour chaque méthode et expérience
      (premier lot isolé sous `data/output/experiments/baseline/cpu/`).
- [x] Définir un protocole commun : mêmes questions, mêmes références, même k=5,
      mêmes limites de sources et comparaison avec la moulinette exécutée à la main.
- [x] Vérifier la base sur CPU ; ne pas confondre le fonctionnement CUDA actuel
      avec la validation CPU demandée pour les bonus.

Validation : un point de départ reproductible, sans changement de comportement.

### Bilan de référence avant les bonus

- Dernière vérification locale : **215 tests réussis**, flake8 réussi, mypy
  réussi sur **56 fichiers**. Les tests du grand k sont inclus.
- Environnement inspecté : Python 3.12.13, Linux x86_64 (noyau 6.8.0-149-generic),
  PyTorch 2.14.1+cu130, Transformers 4.57.6, Pydantic 2.14.2,
  rank-bm25 0.2.2, Fire 0.7.1, pytest 9.2.0 et uv 0.11.7.
  CPU exposé à l'environnement d'inspection : Intel Core Ultra 7 265,
  20 processeurs logiques ; à distinguer de toute autre machine de benchmark.
- GPU précédemment confirmé par l'élève : RTX 4050 Laptop, 6 Go.
  Smoke test CPU explicite réussi avec `answer_dataset --device cpu` : une
  question LoRA, réponse correcte et non vide, cinq sources sauvegardées,
  11,05 s de temps interne total (progression : 7,70 s pour la question).
  Sortie : `data/output/experiments/baseline/cpu/rag_cpu_smoke.json`.
  Ce smoke test confirme le fonctionnement CPU de base, pas les performances
  d'un batch complet ni les futurs bonus.
- Données : vLLM 0.10.1, 16 736 chunks au maximum de 2 000 caractères ;
  datasets publics docs (100 questions) et code (99 questions).
- Méthode : BM25, k1=1.5, b=0.75, k=5. Références prises dans
  `data/datasets/AnsweredQuestions/`, requêtes dans `UnansweredQuestions/`.
- Baseline de recherche : recall@5 docs 81 %, code 66,67 %, mesurés avant
  génération. Les fichiers après génération ont également passé les seuils
  officiels selon les sorties rapportées par l'élève ; ne pas leur attribuer
  les valeurs exactes du retrieval sans les remesurer.
- Temps rapportés : indexation 10,10 s (chronomètre interne) ; recherche de
  200 questions 12,03 s (temps total externe), dont une question répétée
  uniquement pour atteindre 200 requêtes.
- Installation : uv sync et make lint confirmés par l'élève dans un clone
  distinct ; consignes d'extraction des pièces jointes corrigées dans le README.
- Protocole des comparaisons futures : sources <= 2 000 caractères, mêmes
  datasets et k=5, moulinette indépendante, sorties isolées sous
  `data/output/experiments/<methode>/<experience>/`. Ces chemins sont proposés
  pour les prochaines expériences ; ils ne prétendent pas déjà contenir des résultats.
- Vérification CPU terminée sans relancer les 199 générations.
- Commit de référence : `e8db1b6` — `Validate mandatory baseline and plan bonuses`.
  Le push reste une action effectuée par l'élève ; son état distant n'est pas
  attesté par ce bilan local.

## 1. Bonus : embeddings sémantiques

**Exigé :** ajouter un index vectoriel avec un petit modèle CPU, à côté de
l'index lexical. Le sujet cite `all-MiniLM-L6-v2` comme exemple, sans l'imposer.

**Concept :** représenter questions et passages par des vecteurs ; rapprocher
les vecteurs pour retrouver des passages même sans mots identiques.

### Progression pédagogique — 18 novembre 2026

Expériences réalisées dans `notes/bonus/` sur MacBook M2, avec MiniLM
explicitement placé sur CPU. Les sorties ci-dessous ont été transmises par
l'élève ; aucun benchmark du corpus ni nouvelle validation Linux n'a été fait.
Ces cases valident des expériences, pas l'intégration du bonus dans `src/`.

- [x] Comparer produit scalaire, norme et cosinus sur des vecteurs artificiels
      (`bonus_explore_vector_similarity.py`).
- [x] Observer les tokens, identifiants, dimensions, padding, attention mask
      et troncature (`bonus_explore_embedding_tokenizer.py`).
- [x] Charger `all-MiniLM-L6-v2` avec Transformers/PyTorch sur CPU et observer
      les représentations de tokens à 384 coordonnées.
- [x] Calculer un mean pooling masqué, puis normaliser les vecteurs par texte ;
      normes observées proches de 1.
- [x] Extraire `encode_texts()` et réutiliser le même modèle pour encoder
      séparément passages et questions (`bonus_explore_encoder.py`).
- [x] Comparer deux questions à trois passages par produit matriciel : scores
      de forme `(2, 3)`, tri et top-k avec conservation des indices.
      Premiers résultats observés : RAM pour la question GPU (0,4746), port
      HTTP pour la question serveur (0,7315). Ces exemples ne mesurent pas le recall.
- [x] Encoder trois passages en deux lots avec `batch_size=2` et tqdm,
      puis concaténer leurs vecteurs dans l'ordre : résultat `(3, 384)` et
      classements conservés à de petits écarts numériques près.
- [x] Adapter la source PyTorch selon la plateforme dans `pyproject.toml`
      et actualiser `uv.lock` ; scripts exécutés sur le Mac. Pas de nouvelle
      dépendance d'embeddings ajoutée : Transformers/PyTorch sont réutilisés.

Compréhension travaillée : coordonnées versus norme, direction versus longueur,
masque et broadcasting, dimensions des scores, lots versus top-k, association
passage ↔ vecteur. Le pooling regroupe les représentations ; la normalisation
ajuste la norme. Les questions de compréhension ont été reprises sur ces points.

### Point de reprise historique — 20 novembre 2026

État intermédiaire conservé ; la reprise du 23 novembre en tête du document
décrit les ajouts suivants et la prochaine mission actuelle.

Intégration effectuée progressivement par l'élève dans `src/`, dans l'ordre
des données. État vérifié par lecture du code ; résultats d'exécution transmis
par l'élève, sauf les contrôles du corpus explicitement indiqués ci-dessous.

- [x] Installer corpus et datasets publics sur cette copie : 100 questions docs,
      99 code ; correspondance des identifiants et questions entre les scopes.
      Contrôle local : sources présentes, bornes valides et texte des 16 736
      chunks existants conforme aux tranches des fichiers du corpus.
- [x] Observer la perte d'une information finale par troncature sur un texte
      de 1 717 caractères et 292 tokens ; récupération par deux segments de
      256 et 38 positions, tokens spéciaux compris.
- [x] Explorer le score maximal par passage à partir des scores des segments,
      sans doublonner le passage. D'abord exploré dans `notes/`, ce regroupement
      a ensuite été intégré dans `src/semantic_retriever.py` (voir reprise actuelle).
- [x] Créer `EmbeddingConfig`, `TokenSegment` et `SemanticEncoder` dans
      `src/embeddings.py` : CPU float32, tokenizer fast, segmentation sans
      chevauchement, encodage par lots, pooling masqué et normalisation L2.
- [x] Fixer la révision du modèle et du tokenizer à
      `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, disponible dans le cache Mac.
- [x] Encoder un échantillon réel de trois chunks : six segments, trois lots
      de deux, tenseur `(6, 384)`, normes proches de 1 et associations conservées.
- [x] Créer `SemanticIndexMetaData` et `SemanticIndexData` dans
      `src/semantic_index.py` : configuration, versions, compteurs, dimension,
      stratégie de préparation et validation des associations déclarées.
- [x] Vérifier forme, CPU, float32, valeurs finies et normes des vecteurs ;
      copie contenant un vecteur nul explicitement refusée.
- [x] Sauvegarder les données simples et les vecteurs dans un fichier local
      unique via un temporaire ; recharger sur CPU avec `weights_only=True`
      et relancer les validations. Chemin configurable par argument `Path`.
      Échantillon : `data/output/experiments/semantic/sample/semantic_index.pt`,
      17 775 octets ; données et vecteurs exactement conservés à l'aller-retour.
- [x] Ajouter quatre tests sans chargement de modèle dans
      `tests/test_semantic_index.py` : vecteurs valides, vecteur nul, compteur
      incohérent et persistance. Quatre tests et `make lint` rapportés réussis
      par l'élève ; pas de nouvelle exécution de toute la suite attestée ici.

À cette date intermédiaire, l'encodage des questions et le retriever restaient
à intégrer ; ils sont maintenant présents. Comparaison BM25, branchement CLI
et datasets, indexation complète et mesures restent à faire.
L'index d'essai ne couvre que trois chunks : aucun recall sémantique, gain
hybride ni performance du corpus complet n'est encore mesuré.

### Intégration et validation du bonus

- [ ] Comparer un petit exemple lexical et sémantique avant d'intégrer le modèle.
- [x] Choisir le modèle et sa bibliothèque après lecture de leur documentation ;
      vérifier taille, licence, limite d'entrée, dépendances et fonctionnement CPU.
- [x] Expliquer la limite du tokenizer d'embeddings : 2 000 caractères ne
      garantissent pas qu'un passage entier tient dans sa fenêtre de tokens.
- [x] Définir et documenter la politique initiale pour les passages trop longs :
      fenêtres sans chevauchement, 256 positions dont deux tokens spéciaux,
      plusieurs vecteurs associés au chunk original. Score maximal par chunk
      intégré au retriever, testé et évalué sur le corpus complet.
- [x] Réutiliser Transformers/PyTorch ; configuration uv multiplateforme et
      lockfile actualisés lors des étapes précédentes, sans dépendance supplémentaire.
- [x] Définir les métadonnées de l'index : modèle/révision, paramètres, version
      de préparation des textes, dimension et association vecteur ↔ chunk.
- [x] Encoder les segments des chunks par lots sur CPU avec une progression tqdm
      (échantillon puis corpus complet exécutés sur CPU).
- [x] Normaliser les vecteurs pour utiliser une similarité cosinus cohérente.
- [x] Commencer par une recherche matricielle exacte, évaluée sur le corpus
      complet ; pas de bibliothèque vectorielle supplémentaire ajoutée.
- [x] Persister vecteurs et métadonnées sous un chemin configurable ignoré par Git.
- [x] Implémenter une recherche sémantique retournant `list[ScoredChunk]` avec
      les sources originales, comme BM25 ; branchement CLI/datasets effectué.
- [x] Ajouter un choix de méthode à la CLI en conservant BM25 par défaut.
- [x] Tester sans téléchargement : dimensions, ordre, k=0, grand k, index vide,
      association des sources, score et persistance avec encodeur simulé.
- [x] Faire un test d'intégration de l'encodeur et de la persistance avec le vrai
      modèle forcé sur CPU (Mac puis Ubuntu, échantillon et corpus complet).
- [x] Mesurer recall@5 docs/code et temps de construction de l'index sur CPU.
- [ ] Compléter les mesures : temps de recherche docs/code, nombre exact de
      segments, proportion de chunks multisegments, taille et mémoire maximale.
- [ ] Compléter les tests ciblés de segmentation/pooling/lots de l'encodeur
      et la couverture CLI propre au choix de méthode et à index_semantic.

Validation : index sémantique rechargeable et recherche opérationnelle CPU,
avec sources exactes et comparaison documentée. Une amélioration de recall
n'est pas garantie par l'ajout d'embeddings.

## 2. Bonus : recherche hybride

**Exigé :** combiner les classements lexical et sémantique en un classement unique.

- [ ] Facultatif : observer les passages retrouvés exclusivement par chaque
      méthode (diagnostic reporté ; pas une exigence supplémentaire du sujet).
- [x] Choisir une fusion initiale simple, par exemple Reciprocal Rank Fusion
      (RRF), afin de combiner des rangs sans comparer directement des scores
      BM25 et cosinus dont les échelles diffèrent.
- [x] Définir le nombre de candidats de chaque méthode, distinct du k final.
- [x] Dédupliquer par identité de source : chemin et deux bornes de caractères.
- [x] Fixer un ordre déterministe en cas d'égalité et retourner au plus k sources.
- [x] Exposer le mode hybride sans casser la CLI obligatoire (BM25 par défaut).
- [ ] Facultatif : exposer candidats, constante et poids comme options CLI ;
      ils sont actuellement configurés dans `src/retrieval.py`.
- [x] Tester la fusion sur des listes contrôlées : source commune, source unique,
      classement vide, égalité, k=0 et grand k.
- [ ] Brancher les sources fusionnées sur le prompt et le générateur existants.
- [x] Comparer BM25, sémantique et hybride sur les mêmes datasets à recall@5.
- [x] Comparer les constantes et poids sur des candidats identiques ; retenir
      C=0 et BM25/MiniLM 65/35 sans revendiquer de généralisation hors public.
- [ ] Ajouter des tests pytest persistants pour les poids et leur validation,
      ainsi que le choix hybride et la cohérence des index via le chargeur.
- [ ] Vérifier quelques réponses et la gestion du budget avec les sources hybrides.
- [x] Documenter les scores et limites dans le README, sur les datasets complets.
- [ ] Compléter un benchmark hybride dédié : temps total de recherche et mémoire.
      Les balayages de paramètres avec candidats réutilisés ne remplacent pas
      un chronométrage de la commande complète.

Validation : les deux classements contribuent réellement à une recherche
hybride CPU, avec des résultats dédupliqués et évalués.

## 3. Bonus : indexation incrémentale

**Exigé :** lorsqu'un fichier change, réindexer ce fichier au lieu de
reconstruire systématiquement tout l'index.

- [ ] Définir un manifeste par fichier : chemin, empreinte du contenu et version
      de configuration du chunking ; utiliser les dates comme optimisation
      éventuelle, pas comme seule garantie de détection.
- [ ] Comparer manifeste et corpus pour détecter ajouts, modifications et suppressions.
- [ ] Conserver les chunks inchangés ; remplacer tous les anciens chunks d'un
      fichier modifié, car une insertion peut décaler ses offsets.
- [ ] Retirer les chunks et vecteurs des fichiers supprimés.
- [ ] Réencoder seulement les passages nécessitant de nouveaux embeddings et
      maintenir l'alignement entre vecteurs et chunks.
- [ ] Définir le traitement des renommages (suppression + ajout suffit au départ).
- [ ] Déclencher une reconstruction compatible quand modèle ou chunking change.
- [ ] Exposer une option incrémentale et une option de reconstruction complète.
- [ ] Sauvegarder index, manifeste et vecteurs de manière cohérente ; ne pas
      publier un mélange d'anciennes et de nouvelles versions en cas d'erreur.
- [ ] Documenter que les statistiques globales BM25 peuvent devoir être recalculées
      même si seuls les fichiers modifiés sont relus et redécoupés.
- [ ] Tester ajout, modification, suppression, renommage, exécution sans changement,
      changement de configuration et échec de lecture.
- [ ] Comparer l'index incrémental à une reconstruction complète du même corpus.
- [ ] Mesurer le temps pour un seul fichier modifié et vérifier que les fichiers
      inchangés ne sont ni redécoupés ni réencodés.

Validation : résultat équivalent à une reconstruction complète, provenance
correcte et travail évité sur les fichiers inchangés.

## 4. Bonus : cache d'index et de requêtes

**Exigé :** mettre en cache l'index et les résultats de recherche pour accélérer
le démarrage et les requêtes répétées. Le simple fichier de chunks mandatory
ne constitue pas, à lui seul, tout ce bonus.

- [ ] Mesurer séparément lecture JSON, tokenisation, construction BM25, chargement
      des vecteurs et recherche afin de cibler les coûts réels.
- [ ] Définir une révision d'index issue du contenu et de la configuration.
- [ ] Choisir un format local pour réutiliser les structures nécessaires au
      démarrage ; valider format/version avant utilisation.
- [ ] Si une sérialisation exécutable est choisie, ne charger que des artefacts
      locaux de confiance ; privilégier des formats de données quand possible.
- [ ] Ajouter un cache borné de recherches, avec clé incluant question, k,
      méthode, paramètres et révision d'index.
- [ ] Préserver la sémantique de la question dans la construction de la clé.
- [ ] Invalider les caches après mise à jour incrémentale ou changement de modèle,
      tokenizer, fusion ou configuration affectant les résultats.
- [ ] Décider explicitement quels caches persistent entre deux processus CLI,
      pour accélérer aussi les appels séparés et pas uniquement un serveur vivant.
- [ ] Prévoir désactivation, nettoyage et repli propre si le cache est absent
      ou corrompu ; ignorer les fichiers de cache dans Git.
- [ ] Tester hit/miss, invalidation, taille maximale, corruption et égalité des
      résultats avec le cache désactivé.
- [ ] Mesurer démarrage à cache absent puis présent, et requête nouvelle puis répétée,
      sans confondre ces mesures avec les caches du système d'exploitation.

Validation : accélération mesurée et résultats identiques, sans servir de
sources périmées. Le cache de réponses Qwen est facultatif et hors du minimum
nécessaire pour ce bonus.

## 5. Bonus : API HTTP locale

**Exigé :** permettre la recherche et les réponses par une petite API locale.
Les routes proposées ci-dessous sont un choix de conception, pas celles du sujet.

- [ ] Choisir une bibliothèque HTTP et ajouter ses dépendances après comparaison.
- [ ] Réutiliser les services existants, sans appeler les commandes CLI via subprocess.
- [ ] Définir des modèles Pydantic pour les requêtes, réponses et erreurs.
- [ ] Ajouter une route de santé, une route de recherche et une route de réponse
      (par exemple GET /health, POST /search, POST /answer).
- [ ] Charger le retriever et ses index une fois au démarrage ; charger Qwen
      une fois au démarrage ou au premier besoin selon un choix documenté.
- [ ] Garantir une configuration CPU explicite pour la démonstration campus.
- [ ] Préserver le budget de tokens, la traçabilité et le format des sources.
- [ ] Valider question, k et paramètres ; transformer les erreurs attendues
      en réponses HTTP structurées avec statuts appropriés.
- [ ] Définir une politique de concurrence bornée pour la génération afin de
      limiter les allocations simultanées et protéger les objets partagés.
- [ ] Écouter sur 127.0.0.1 par défaut ; rendre port et chemins configurables.
- [ ] Ajouter une commande de lancement et des exemples curl au README.
- [ ] Tester les routes avec retriever/générateur simulés : succès, entrée invalide,
      absence de sources, index absent et échec du générateur.
- [ ] Faire une vraie recherche et une vraie réponse HTTP sur CPU.
- [ ] Vérifier le comportement des caches et le renouvellement de l'index dans
      un processus serveur qui reste en mémoire.

Validation : l'API répond à des requêtes locales réelles, réutilise les mêmes
composants que la CLI et fonctionne sans GPU.

## 6. Validation finale des cinq bonus

- [ ] Exécuter tous les tests et le lint mandatory, y compris les fichiers bonus.
- [ ] Rejouer le mandatory pour détecter toute régression de CLI, formats ou recall.
- [ ] Démontrer chaque bonus sur CPU avec une commande reproductible.
- [ ] Consigner les performances CPU et les scores par méthode, sans promettre
      un gain si les mesures ne le montrent pas.
- [ ] Refaire l'installation propre avec uv sync et les nouvelles dépendances.
- [ ] Compléter le README en anglais : installation, téléchargement initial du
      modèle d'embeddings, méthodes, invalidation, limites, API et exemples.
- [ ] Contrôler Git : aucun poids, vecteur, index, cache, corpus ou résultat généré.
- [ ] Préparer une explication et une démonstration courte pour chaque bonus.

Prochain incrément : petit batch Qwen CPU avec sources hybrides et tests
ciblés restants, puis manifeste et détection des changements (bonus 3).
