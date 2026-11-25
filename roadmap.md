# Roadmap — RAG Against the Machine

## Priorité de rendu — 25 novembre 2026, reprise sur Lenovo 42

**Décision explicite de l'élève : préparer le rendu aujourd'hui avec le
mandatory et uniquement les bonus 1 (sémantique) et 2 (hybride).** Cette
section remplace les anciennes prochaines missions et objectifs de cinq bonus.
Ne pas poursuivre l'incrémental, le cache ou l'API HTTP avant le rendu.
L'objectif personnel d'avril 2027 ci-dessous est historique : il ne justifie
pas de retarder cette livraison. Détails de reprise : `bonus_rag.md`, section
« Rendu retenu et reprise sur Lenovo » en tête de fichier.

### État à transférer et périmètre

- Mandatory : implémenté ; refaire les contrôles finaux sur la machine cible.
- Bonus 1/2 : implémentés et évalués sur CPU ; BM25 reste le défaut.
  Hybride : candidats=20, C=0, poids BM25/MiniLM=0,65/0,35.
- Bonus 3 : code partiel déjà commité (`manifest`, `incremental`,
  `semantic_incremental`, expériences/tests), pas intégré à la CLI et pas
  revendiqué pour le rendu. Aucun besoin de le terminer.
- Bonus 4 : non implémenté, exclu du rendu.
- Bonus 5 : tentative interrompue ; `src/api.py`, `src/api_service.py` et
  `tests/test_api.py` sont actuellement NON SUIVIS par Git. Ne pas les ajouter
  automatiquement au commit final. Leurs tests HTTP ont rencontré une
  interdiction de ports locaux dans le sandbox ; pas de validation obtenue.
  Conserver/archiver ces brouillons si souhaité, sans les présenter comme un
  bonus terminé. Leur présence locale influence pytest/lint même non suivis.
- Aucune suppression, aucun commit/push ni archivage effectués par l'assistant
  dans cette mise à jour documentaire. Contrôler le diff avant transfert.

### Ordre de travail sur Lenovo avant le push final

1. Vérifier `git status`, branche/commit et fichiers effectivement transférés.
   Les fichiers non suivis ne voyagent pas par Git ; ne pas copier `.venv` Mac.
   Corpus, datasets, moulinette, poids et index générés sont hors Git : vérifier
   leur présence et leur version sur Lenovo avant les mesures.
2. Installer avec `uv sync` sous Python 3.12. Lire le sujet et le README actuel,
   pas les anciens jalons pédagogiques comme un état réel du code.
3. Traiter explicitement les brouillons API avant la validation finale locale
   (conservation hors des chemins actifs ou exclusion du transfert) ; ne pas
   masquer des erreurs de code soumis. Revoir le README anglais pour ne
   revendiquer que mandatory + bonus 1/2 et signaler les travaux expérimentaux.
4. Exécuter `make lint` et `uv run python -m pytest -q` sur l'arbre exact à rendre.
   Compléter si nécessaire les tests persistants des poids RRF et du chargement
   hybride. Les résultats historiques ne remplacent pas cette exécution.
5. Reconstruire les chunks à 2000 caractères, puis l'index MiniLM sur ces mêmes
   chunks. Éviter de mélanger deux états de corpus ; le loader hybride exige
   des chunks identiques. MiniLM doit rester sur CPU.
6. Rejouer BM25 sur docs et code, moulinette officielle, k=5, sources <=2000
   caractères : docs >=80 %, code >=50 %. Mesurer indexation mandatory <=300 s
   et retrieval de 200 questions <=90 s, chargement compris.
7. Rejouer sémantique puis hybride sur les mêmes datasets, dans des dossiers
   distincts ; mesurer temps/mémoire et conserver la provenance des mesures.
   Ne pas retuner indéfiniment : vérifier le réglage retenu et les contrats.
8. Vérifier une génération réelle Qwen et `answer_dataset` avec sources
   valides et budget respecté ; tester les erreurs CLI sans traceback.
   Démonstration campus CPU pour les bonus ; la moulinette de retrieval ne
   valide pas à elle seule la qualité des réponses.
9. Vérifier README anglais, commandes reproductibles, absence de gros
   fichiers/poids/index/caches dans Git et diff final. L'élève effectue ensuite
   le commit/push et contrôle le dépôt distant.

### Mesures de référence, non nouvelles validations

| Méthode | Recall@5 docs | Recall@5 code |
|---|---:|---:|
| BM25 | 81,00 % | 66,67 % |
| MiniLM seul | 66,00 % | 41,41 % |
| Hybride retenu | 83,00 % | 64,65 % |

Le sémantique seul ne satisfait pas les seuils mandatory : conserver la
baseline BM25. L'hybride améliore docs mais baisse code face à BM25.
Build sémantique historique : 306,22 s CPU, distinct de l'indexation mandatory
BM25 mesurée à 10,10 s ; ne pas annoncer que 306,22 s respecte 300 s.
Derniers contrôles observés avant la tentative API : 37 tests manifest/chunks
incrémentaux réussis ; ancien lint sur 88 fichiers avant ajouts ultérieurs.
La suite globale de 285 tests date d'avant l'incrémental. Refaire les contrôles.

## Historique pédagogique — à lire après le plan de rendu ci-dessus

Document de travail pédagogique pour reconstruire RAG_v2 depuis zéro, en conservant la logique générale de RAG_v1. Référence : `subject.pdf`, version 2.0. Les numéros de sections ci-dessous renvoient au sujet.

Objectif personnel : comprendre, implémenter, mesurer et défendre le projet avant le **30 avril 2027**. Cette échéance vient de `tutor_agent_RAG_42.md`, pas du sujet officiel.

## 1. Comment utiliser cette roadmap

- Avancer par petits incréments : pourquoi → concept → choix → code ciblé → expérience → analyse → intégration.
- Créer les fichiers au moment où ils deviennent utiles ; l'arborescence cible ne constitue pas une demande de tout générer immédiatement.
- Pour chaque étape, distinguer quatre validations : **code écrit**, **tests exécutés et réussis**, **compréhension démontrée**, **conformité vérifiée**.
- Laisser l'élève exécuter les expériences et expliquer les résultats. Un test vert ne prouve pas à lui seul la compréhension.
- Conserver V1 comme référence de comparaison. Ne pas recopier une solution sans en comprendre les contrats et les compromis.
- Ajouter annotations de types, docstrings, gestion des erreurs et tests pertinents pendant la construction de chaque composant.
- Ne jamais annoncer un score ou un temps sans mesure ; documenter machine, configuration, dataset et commande.

Légende : `[ ]` à faire, `[~]` en cours, `[x]` validé. Une étape peut être bloquée ; préciser alors la cause et la prochaine action dans le suivi.

## 2. Exigences du sujet et choix de départ

### Exigé par le sujet

| Exigence | Référence | Validation prévue |
|---|---|---|
| Python ≥ 3.10, flake8, mypy, annotations et docstrings PEP 257 | V.1 | Lint et revue du code |
| Makefile : `install`, `run`, `debug`, `clean`, `lint` | V.2 | Exécuter chaque cible ; `lint-strict` facultatif |
| `uv` et installation par `uv sync` | V.4, VI.7 | Installation propre avec `pyproject.toml` et `uv.lock` |
| Pydantic pour les structures échangées entre étapes | V.4, VI.4 | Validation des entrées, sorties et données intermédiaires |
| CLI Python Fire et barres `tqdm` pour opérations longues | V.4, VI.6 | Commandes réelles et cas limites |
| Deux stratégies de chunking : Python et Markdown/texte | VI.1 | Tests dédiés à chaque stratégie |
| Au moins une méthode lexicale : BM25 ou TF-IDF | VI.1 | Retrieval fonctionnel et expliqué |
| `--max_chunk_size` configurable, défaut 2000 caractères | VI.1 | Aucun résultat de plus de 2000 caractères ; refuser un réglage supérieur |
| Chemins exacts et positions en caractères | VI.2, VI.5 | Relecture des passages dans le corpus |
| Qwen/Qwen3-0.6B doit fonctionner | V.4, VI.3 | Génération locale sur des sources récupérées |
| Sorties JSON suivant les modèles fournis | VI.4–VI.5 | Rechargement des fichiers avec Pydantic |
| Six commandes, chemins d'entrée/sortie configurables | VI.6–VI.7 | Parcours complet et répertoires alternatifs |
| Pas de traceback non géré sur les cas dégénérés | V.1, VI.6 | Essais d'erreurs CLI |
| Indexation du corpus ≤ 5 minutes | VII.1.2 | Mesure complète de `index` |
| Retrieval de 200 questions ≤ 90 secondes | VII.1.2 | Mesure incluant chargement et initialisation |
| Recall@5 documentation ≥ 80 %, code ≥ 50 % | VII.1.2 | Évaluation propre et vérification officielle |
| README en anglais avec contenu prescrit | VIII | Checklist documentaire finale |
| Pas de gros fichiers de données, poids ou sorties générées dans Git | X | Vérification des fichiers suivis |
| Bonus seulement après validation intégrale du obligatoire | IX | Jalon obligatoire terminé avant bonus |

Le sujet autorise les autres bibliothèques. Il impose le comportement et certains outils, mais pas les noms de tous nos modules internes. Les tests sont recommandés en V.3 et ne sont pas évalués comme livrables.

### Observé dans les fichiers présents

- V2 contient le sujet, le contrat de tutorat, `README_rag.md`, `vllm-0.10.1.zip` et `datasets_public.zip`.
- Le README joint décrit la moulinette ; il ne remplace pas notre futur README de projet.
- Les archives ont été consultées, mais ne sont pas encore installées dans l'arborescence finale.
- V1 implémente ingestion, chunking, BM25, recherche par dataset et évaluation ; la génération y reste à implémenter.
- Ses notes historiques annoncent recall@5 docs 80,0 %, code 68,7 %, indexation 2,2 s, retrieval de 200 questions environ 20 s. **Ces mesures n'ont pas été reproduites dans V2 et ne constituent pas des résultats V2.**

### Choix pédagogiques et recommandations d'ingénierie

- Partir sur BM25, comme V1 ; comprendre le score sur un petit exemple avant d'utiliser `rank_bm25`.
- Garder le texte original pour les sources et une représentation tokenisée distincte pour la recherche.
- Commencer avec des chunks sans overlap ; comparer d'autres tailles et le chevauchement après avoir construit l'évaluation.
- Découper le Markdown par titres/paragraphes et explorer l'AST pour le Python. Ce sont nos choix initiaux, pas des algorithmes prescrits par le sujet.
- Persister d'abord les chunks et leur configuration en JSON. Comme V1, reconstruire les statistiques BM25 au chargement ; mesurer ce coût avant de décider de persister davantage.
- Construire la boucle de mesure avant les optimisations. Ajouter les embeddings seulement au stade bonus.
- Préparer la génération pour une exécution CPU reproductible ; les bonus doivent explicitement fonctionner sur une machine campus CPU-only.

## 3. Pipeline et arborescence cible

```text
Préparation réutilisable
Corpus → ingestion → chunks avec sources → persistance
                                        → tokenisation et statistiques BM25

Pour chaque question
Question → tokenisation → classement BM25 → top-k sources
         → sélection du contexte et prompt → Qwen → réponse + sources en JSON

Pour l'évaluation du retrieval
Sources récupérées + sources de référence → IoU → recall@k
```

```text
RAG_v2/
├── subject.pdf                       # sujet existant
├── tutor_agent_RAG_42.md              # contrat pédagogique existant
├── README_rag.md                     # documentation jointe de la moulinette
├── roadmap.md                        # ce suivi
├── README.md                         # documentation finale en anglais
├── pyproject.toml                    # dépendances et configuration outils
├── uv.lock                           # généré par uv, à versionner
├── Makefile
├── .gitignore
├── .flake8                           # configuration éventuelle
├── src/
│   ├── __init__.py
│   ├── __main__.py                   # point d'entrée Fire et erreurs CLI
│   ├── cli.py                        # orchestration des six commandes
│   ├── models.py                     # contrats Pydantic
│   ├── ingest.py                     # découverte et lecture des fichiers
│   ├── chunking.py                   # découpage Python et Markdown/texte
│   ├── indexer.py                    # construction et persistance des chunks
│   ├── tokenizer.py                  # termes utilisés par BM25
│   ├── retriever.py                  # scoring et classement top-k
│   ├── datasets.py                   # lecture/écriture des JSON de datasets
│   ├── evaluation.py                 # IoU, recall et diagnostic des échecs
│   ├── prompting.py                  # contexte, budget tokens, prompt sourcé
│   └── generator.py                  # chargement et génération Qwen
├── tests/
│   ├── test_models.py
│   ├── test_ingest.py
│   ├── test_chunking.py
│   ├── test_indexer.py
│   ├── test_tokenizer.py
│   ├── test_retriever.py
│   ├── test_datasets.py
│   ├── test_evaluation.py
│   ├── test_prompting.py
│   ├── test_generator.py
│   └── test_cli.py
├── notes/
│   ├── corpus.md                     # observations sur les vraies données
│   ├── experiments.md                # configurations et résultats mesurés
│   └── defense.md                    # décisions, limites et exercices
└── data/                             # données et artefacts locaux hors Git
    ├── raw/vllm-0.10.1/
    ├── processed/
    ├── datasets/
    │   ├── UnansweredQuestions/
    │   └── AnsweredQuestions/
    └── output/
        ├── search_results/<DatasetScope>/
        └── search_results_and_answer/<DatasetScope>/
```

Les fichiers de tests, notes et modules internes sont une proposition d'organisation. Les créer progressivement ; ne pas ajouter de module vide sans besoin. `DatasetScope` désigne notamment `UnansweredQuestions` ou `AnsweredQuestions`, pour éviter les écrasements entre datasets portant le même nom.

## 4. Étapes de construction

### Étape 0 — Fondamentaux et lecture du sujet

**But :** comprendre ce que nous construisons avant d'écrire le pipeline.

**Fichiers :** `subject.pdf`, `tutor_agent_RAG_42.md`, `roadmap.md`.

- [x] Lire le sujet et identifier les contraintes principales.
- [x] Distinguer corpus, dataset, retriever et LLM.
- [x] Comprendre qu'une mise à jour documentaire nécessite une réindexation, pas un réentraînement de Qwen.
- [ ] Expliquer les quatre étapes : indexing, retrieving, augmenting, generating.
- [ ] Distinguer caractères, tokens lexicaux de BM25 et tokens du tokenizer de Qwen.
- [ ] Expliquer pourquoi des passages pertinents réduisent certains risques d'hallucination sans garantir une réponse correcte.

**Validation :** décrire le chemin d'une question jusqu'à la réponse et les entrées/sorties de chaque étape. Les échanges actuels démontrent une compréhension initiale ; aucune implémentation n'est encore validée.

### Étape 1 — Préparer et explorer les données

**But :** observer les fichiers réels et les sources attendues.

**Fichiers/dossiers :** archives existantes → `data/raw/`, `data/datasets/`, puis `notes/corpus.md` et règles d'exclusion dans `.gitignore`.

- [x] Inspecter les archives et extraire uniquement les données utiles à leur emplacement final.
- [x] Retirer le niveau d'emballage `datasets_public/public/` lors de l'installation des datasets : obtenir directement `data/datasets/AnsweredQuestions/` et `UnansweredQuestions/`.
- [ ] Vérifier que les chemins des références correspondent au corpus extrait.
- [ ] Compter fichiers, extensions et questions ; inspecter documentation et code.
- [x] Lire une question sans réponse, sa version avec réponse et son passage de référence.
- [x] Reproduire `text[first:last]` sur l'exemple LoRA : `docs/features/lora.md`, `[4695:6098]`.
- [x] Examiner encodage, Unicode et fins de ligne ; définir la convention de lecture compatible avec les offsets du dataset.
- [ ] Décider des extensions et exclusions en observant les données ; ne pas reprendre automatiquement les exclusions de V1.
- [ ] Vérifier la présence de la moulinette et l'environnement Linux nécessaire à ses binaires. Le README joint seul ne fournit pas l'exécutable.

**Expérience :** afficher un passage de référence avec un petit script utilisant uniquement la bibliothèque standard, puis vérifier ses bornes et sa longueur.

**Compréhension :** pourquoi normaliser le texte avant de calculer ses positions peut-il rendre les sources incorrectes ?

### Étape 2 — Installer le socle Python et la CLI minimale

**But :** disposer d'un projet exécutable et vérifiable.

**Fichiers :** `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`, configuration flake8/mypy, `src/__init__.py`, `src/__main__.py`, `src/cli.py` ; commencer `README.md`.

- [x] Déclarer Python ≥ 3.10 et les premières dépendances : Pydantic, Fire, tqdm ; outils de développement : flake8, mypy, pytest.
- [ ] Ajouter `rank_bm25` lorsque le retrieval devient utile ; déclarer les dépendances directement importées.
- [x] Générer `uv.lock` avec uv, comprendre son rôle et vérifier `uv sync`.
- [x] Prévoir les exclusions de données et environnements dans flake8/mypy : ne pas analyser le code tiers vLLM comme notre code.
- [ ] Ajouter les cibles Makefile obligatoires ; limiter `clean` aux artefacts temporaires du projet.
- [x] Brancher Fire sur une petite CLI ; ajouter les vraies commandes au fil des étapes, sans annoncer les placeholders comme fonctionnels.
- [ ] Prévoir une erreur CLI explicite, un message utile et un code de sortie non nul.

**Vérifications à ce stade :** `uv sync`, `uv run python -m src --help`, `make lint`.

La cible `lint` doit exécuter `flake8 .` et `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` dans l'environnement uv. `lint-strict` est facultative.

**Compréhension :** distinguer dépendance déclarée, environnement installé et version verrouillée.

### Étape 3 — Définir les contrats de données

**But :** préciser ce que chaque composant reçoit et produit.

**Fichiers :** `src/models.py`, `tests/test_models.py`.

- [x] Implémenter les huit modèles du sujet : `MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`.
- [x] Conserver exactement les noms et types des champs du contrat officiel ; conserver le `question_id` fourni.
- [ ] Ajouter progressivement les modèles internes utiles : `Chunk` (source + texte), `ChunkIndex` (chunks + configuration), `ScoredChunk` (chunk + score).
- [x] Définir les bornes des sources : début inclus, fin exclue, plage non vide et positive.
- [x] Séparer validation des modèles et règles propres aux sorties, notamment limite des 2000 caractères.
- [ ] Charger de vrais datasets et vérifier un aller-retour JSON.
- [x] Étudier le piège d'une union Pydantic : une question avec sources invalides ne doit pas devenir silencieusement une question sans réponse lors de l'évaluation. Prévoir un contrat strict de ground truth, par exemple `AnsweredDataset`.

**Validation :** bonnes données acceptées, mauvaises bornes refusées, noms de champs préservés, références invalides signalées.

**Compréhension :** pourquoi valider la structure d'une source ne prouve pas que son passage répond à la question ?

### Étape 4 — Lire le corpus sans perdre les sources

**But :** rendre les fichiers accessibles au chunker de manière déterministe.

**Fichiers :** `src/ingest.py`, `tests/test_ingest.py`, `notes/corpus.md`.

- [x] Lister les fichiers retenus dans un ordre stable.
- [x] Rendre le répertoire d'entrée configurable et justifier les filtres.
- [ ] Lire avec context manager ; traiter fichier absent, illisible ou encodage invalide.
- [x] Distinguer chemin physique de lecture et chemin de source sérialisé : conserver le chemin relatif exact attendu, sans préfixe absolu local.
- [x] Décider explicitement du comportement pour un corpus externe à la racine ; ne pas produire un chemin artificiel incompatible avec les références.
- [ ] Préserver la convention d'offsets établie à l'étape 1 et rapporter les fichiers ignorés.

**Validation :** petit corpus temporaire avec sous-dossiers, Unicode, fichier vide et fichier illisible ; résultats prévisibles et erreurs utiles.

**Compréhension :** distinguer index de caractère Python et position en octets UTF-8.

### Étape 5 — Construire le chunking Markdown/texte

**But :** produire des passages de taille limitée en conservant du contexte.

**Fichiers :** `src/chunking.py`, `tests/test_chunking.py`.

- [x] Commencer par un découpage simple pour comprendre les plages `[first:last]`.
- [x] Découper le texte en paragraphes ; regrouper les petits blocs sans dépasser la limite.
- [x] Pour Markdown, reconnaître les titres hors blocs de code clôturés.
- [x] Découper les sections trop longues par paragraphes, puis coupe stricte en dernier recours (préférence pour les lignes non implémentée).
- [x] Décider du traitement des blancs sans modifier le texte de référence.
- [x] Vérifier `chunk.text == original[first:last]` et `0 < last - first <= max_chunk_size` pour chaque chunk (corpus complet, taille 2000).
- [ ] Tester paragraphes longs, ligne très longue, titres dans un bloc de code, Unicode et texte vide.

**Expérience :** comparer visuellement les chunks de `lora.md` à 2000 et 500 caractères. Ne pas conclure à une amélioration du recall sans évaluation.

**Compréhension :** que perd-on quand on sépare une phrase explicative de l'endpoint qu'elle introduit ?

### Étape 6 — Construire le chunking Python

**But :** adapter les frontières aux structures du code.

**Fichiers :** compléter `src/chunking.py`, `tests/test_chunking.py`.

- [x] Comprendre ce que représente un AST et lire celui d'une petite fonction.
- [x] Délimiter fonctions/classes et conserver décorateurs, commentaires utiles et code de niveau module.
- [x] Regrouper les petites unités ; subdiviser une grande classe ou fonction.
- [x] Gérer les blocs indivisibles trop longs par une stratégie de repli bornée.
- [x] Prévoir un repli texte lorsqu'un fichier ne peut pas être parsé.
- [x] Vérifier les conversions de positions : certaines colonnes AST sont exprimées en octets UTF-8 ; utiliser les numéros de lignes quand cela suffit ou convertir explicitement.
- [ ] Tester fonctions, classes, méthodes, code hors fonction, syntaxe invalide et caractères non ASCII.

**Validation :** deux stratégies réellement distinctes, texte exact et limite de taille respectée sur le corpus.

**Compréhension :** pourquoi une fonction entière n'est-elle pas toujours un chunk admissible ?

### Étape 7 — Construire et persister les chunks

**But :** rendre la préparation du corpus réutilisable.

**Fichiers :** `src/indexer.py`, compléter `src/models.py` et `src/cli.py`, `tests/test_indexer.py`.

- [ ] Relier découverte, lecture, dispatch des chunkers et progression tqdm.
- [ ] Valider `--max_chunk_size` : défaut 2000, entier positif, jamais plus de 2000 pour nos résultats conformes.
- [ ] Enregistrer les chunks et paramètres utiles sous `data/processed/`, par exemple `chunks.json`.
- [ ] Vérifier les invariants avant écriture ; refuser un corpus sans contenu exploitable.
- [ ] Écrire de façon atomique pour éviter un index partiellement sauvegardé.
- [ ] Charger l'index avec validation et signaler index absent/corrompu.
- [ ] Implémenter `index`, avec chemins configurables et valeurs par défaut conformes.
- [ ] Chronométrer l'indexation complète et compter fichiers, chunks et taille disque.

**Commande cible :** `uv run python -m src index --max_chunk_size 2000`.

**Validation :** sauvegarde/rechargement cohérents, sources ≤ 2000 caractères, indexation ≤ 300 s.

**Compréhension :** distinguer notre fichier de chunks persistant et les statistiques BM25 construites à l'étape suivante.

### Étape 8 — Tokeniser et comprendre BM25

**But :** transformer textes et question en termes comparables, puis comprendre leur score.

**Fichiers :** `src/tokenizer.py`, `tests/test_tokenizer.py`, `notes/defense.md`, dépendance BM25 dans `pyproject.toml` et `uv.lock`.

- [ ] Commencer par minuscules et extraction des termes.
- [ ] Étudier `max_model_len`, `LLMEngine`, `LoRA` et `vLLM` : identifiant entier et sous-termes.
- [ ] Utiliser la même tokenisation pour les chunks et les questions.
- [ ] Comprendre TF, DF, IDF, saturation de fréquence (`k1`) et normalisation de longueur (`b`).
- [ ] Calculer et interpréter des scores sur trois ou quatre petits chunks.
- [ ] Vérifier la variante d'IDF de la bibliothèque choisie, notamment les scores nuls ou négatifs possibles sur de petits corpus.
- [ ] Distinguer index inversé et représentation de `rank_bm25` : ne pas prétendre utiliser des posting lists sans les avoir implémentées ou vérifiées.

**Validation :** sorties exactes du tokenizer et classement explicable à la main sur un petit exemple.

**Compréhension :** pourquoi un terme rare peut-il être plus discriminant qu'un terme très fréquent ? Pourquoi BM25 ne reconnaît-il pas automatiquement tous les synonymes ?

### Étape 9 — Construire le retriever et la recherche unitaire

**But :** obtenir les passages les mieux classés pour une question.

**Fichiers :** `src/retriever.py`, compléter `src/cli.py`, `tests/test_retriever.py`.

- [ ] Construire les statistiques BM25 une fois par instance de retriever.
- [ ] Calculer les scores et sélectionner le top-k avec ordre déterministe en cas d'égalité.
- [ ] Définir et tester le filtrage des scores ; ne pas supposer que score positif signifie réponse correcte.
- [ ] Traiter question vide, termes inconnus, `k=0`, `k<0`, `k` supérieur au nombre de chunks et index vide.
- [ ] Retourner des sources, scores et textes inspectables sans appeler Qwen.
- [ ] Implémenter `search` et une sortie JSON conforme ; un affichage lisible peut la compléter.
- [ ] Prévoir un chemin de sauvegarde configurable pour les recherches unitaires, afin de couvrir VI.5.
- [ ] Mesurer séparément chargement, construction BM25 et recherche ; ne pas qualifier la commande complète de milliseconde si son démarrage coûte des secondes.

**Commande cible :** `uv run python -m src search "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5`.

**Validation :** source réelle consultable, classement expliqué, aucun appel LLM.

**Compréhension :** distinguer pertinence lexicale, preuve documentaire et vérité.

### Étape 10 — Rechercher sur les datasets et produire les JSON

**But :** appliquer le même retriever à toutes les questions.

**Fichiers :** `src/datasets.py`, compléter `src/cli.py`, `tests/test_datasets.py`.

- [ ] Lire le JSON avec Pydantic et conserver les `question_id`.
- [ ] Construire le retriever une seule fois pour tout le batch.
- [ ] Rechercher chaque question avec progression tqdm.
- [ ] Sérialiser `StudentSearchResults` avec `search_results`, `k` et les `retrieved_sources`.
- [ ] Vérifier nombre de résultats ≤ k, chemins exacts et longueur ≤ 2000 pour chaque source.
- [ ] Écrire atomiquement dans `--save_directory`, sous le nom du dataset.
- [ ] Séparer les dossiers de sortie par scope pour éviter les écrasements.
- [ ] Mesurer les 200 questions documentation + code, y compris les démarrages nécessaires.

**Validation :** JSON rechargeable, identifiants inchangés, sorties compatibles avec le sujet, temps total ≤ 90 s.

**Compréhension :** pourquoi renvoyer dix sources ne dispense-t-il pas d'atteindre le seuil de recall@5 ?

### Étape 11 — Construire notre évaluation et vérifier la baseline

**But :** comprendre nos scores avant d'essayer de les améliorer.

**Fichiers :** `src/evaluation.py`, compléter `src/cli.py`, `tests/test_evaluation.py`, `notes/experiments.md`.

- [ ] Calculer l'IoU de deux plages dans un même fichier : intersection / union ; fichiers différents → 0.
- [ ] Calculer, pour chaque question, la part des sources attendues retrouvées dans les k premiers résultats, puis agréger sur le dataset.
- [ ] Documenter le traitement des questions sans sources et des identifiants manquants, inconnus ou dupliqués.
- [ ] Le sujet donne le seuil IoU 0,05 ; V1 utilise strictement `> 0.05`. Vérifier le cas d'égalité et les conventions exactes avec la moulinette plutôt que les déduire du PDF seul.
- [ ] Tester plages identiques, disjointes, adjacentes, imbriquées et différents fichiers ; ajouter plusieurs sources attendues pour ne pas réduire abusivement recall à une réussite binaire.
- [ ] Rapporter recall@1/3/5/10 et vérifier sa monotonie.
- [ ] Implémenter `evaluate` avec une ground truth validée strictement.
- [ ] Exécuter manuellement la moulinette dans un environnement compatible et comparer les scores sur les mêmes résultats.
- [ ] Ne jamais importer ni appeler la moulinette depuis le code de la solution.
- [ ] Enregistrer notre première baseline V2 et ses limites.

**Validation :** scores explicables sur exemples faits à la main ; accord avec la moulinette lorsque son exécution est disponible. Sinon, noter explicitement que la conformité officielle reste à vérifier.

**Compréhension :** pourquoi un passage dans le bon fichier peut-il tout de même être un échec ?

### Étape 12 — Améliorer le retrieval par expériences

**But :** atteindre les seuils avec des décisions justifiées.

**Fichiers :** `notes/experiments.md`, modules concernés uniquement, compléments ciblés aux tests.

- [ ] Examiner les échecs : mauvais fichier, bonne région classée trop tard, mauvais passage, recouvrement insuffisant.
- [ ] Ne changer qu'une variable à la fois et conserver une baseline comparable.
- [ ] Comparer tailles de chunks, par exemple 2000 / 1000 / 500.
- [ ] Tester, si les échecs le justifient, overlap, découpage des identifiants, stopwords et paramètres BM25.
- [ ] Mesurer séparément documentation et code ; éviter les règles spécifiques à quelques questions connues.
- [ ] Si des métadonnées sont ajoutées au texte de recherche, garder le texte original et ses offsets séparés.
- [ ] Reconstruire les artefacts lorsque le chunking ou le tokenizer change.
- [ ] Distinguer résultats sur datasets publics utilisés pour ajuster les paramètres et généralisation à des questions nouvelles.
- [ ] Retenir une configuration par défaut, revérifier temps et sources, puis confirmer officiellement les scores.

**Tableau à tenir :** date, changement unique, configuration, machine, recall@1/3/5/10 docs/code, temps d'indexation, démarrage retrieval, temps total, observation, décision.

**Validation :** recall@5 docs ≥ 80 %, code ≥ 50 %, dans les budgets. Chercher une marge raisonnable sans retarder la génération par une optimisation infinie.

### Étape 13 — Construire le contexte et le prompt

**But :** transformer les sources récupérées en entrée exploitable par Qwen.

**Fichiers :** `src/prompting.py`, `tests/test_prompting.py`, compléter les modèles internes si nécessaire.

- [ ] Définir le contrat : question + sources et textes → messages du prompt + sources effectivement incluses.
- [ ] Numéroter les passages et garder leur provenance.
- [ ] Indiquer que les documents sont des données, pas des instructions à exécuter.
- [ ] Demander une réponse fondée sur les passages et un aveu d'information insuffisante si nécessaire.
- [ ] Compter les tokens avec le tokenizer du modèle et son chat template, pas avec la longueur en caractères.
- [ ] Réserver de la place pour les instructions, la question et la génération.
- [ ] Choisir quels passages inclure quand le budget est dépassé ; si un passage est tronqué, conserver une provenance cohérente.
- [ ] Garder la trace de la différence entre sources récupérées et contexte réellement transmis.
- [ ] Vérifier contexte vide, question longue, dépassement de budget et contenu documentaire adversarial.

**Validation :** prompt inspectable, borné et sourcé ; tests purs sans télécharger le modèle, puis vérification avec le vrai tokenizer à l'étape suivante.

**Compréhension :** pourquoi cinq chunks de 2000 caractères ne représentent-ils pas forcément 10 000 tokens ?

### Étape 14 — Intégrer Qwen et répondre à une question

**But :** obtenir une première réponse locale à partir du contexte.

**Fichiers :** `src/generator.py`, `src/prompting.py`, `src/cli.py`, `tests/test_generator.py`, dépendances de génération et lockfile.

- [ ] Choisir et expliquer le backend local compatible avec `Qwen/Qwen3-0.6B` ; `transformers` et PyTorch constituent une option de départ.
- [ ] Vérifier disque et mémoire avant installation/téléchargement volumineux, avec accord selon le contrat de tutorat.
- [ ] Charger modèle et tokenizer uniquement pour les commandes de génération.
- [ ] Vérifier le chat template, le budget et la gestion du mode thinking de Qwen3.
- [ ] Définir limite de génération et paramètres de décodage reproductibles.
- [ ] Séparer les tokens du prompt des nouveaux tokens produits lors du décodage de la réponse.
- [ ] Relier `search` → préparation du contexte → génération pour `answer`.
- [ ] Construire le JSON de sortie avec Pydantic ; le modèle peut générer du texte de réponse, le programme assemble l'enveloppe JSON.
- [ ] Fournir une sauvegarde configurable et un affichage lisible pour la commande unitaire.
- [ ] Tester le code d'orchestration avec un faux backend ; réaliser ensuite une vraie intégration Qwen et la signaler comme telle.
- [ ] Traiter modèle indisponible, problème réseau, contexte vide et ressources insuffisantes avec messages utiles.

**Commande cible :** `uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5`.

**Validation :** modèle obligatoire réellement exécuté, réponse inspectée face aux passages, latence et mémoire observées. Un test simulé ne valide pas le fonctionnement du modèle réel.

### Étape 15 — Générer les réponses par dataset

**But :** compléter le parcours demandé par le sujet.

**Fichiers :** compléter `src/cli.py`, `src/datasets.py`, `src/generator.py`, tests associés.

- [ ] Lire `StudentSearchResults` depuis `--student_search_results_path`.
- [ ] Relire les passages dans le corpus en contrôlant chemin, existence et bornes.
- [ ] Charger Qwen une fois pour l'ensemble du batch.
- [ ] Construire le contexte et générer une réponse par question avec tqdm.
- [ ] Écrire `StudentSearchResultsAndAnswer` dans le dossier configuré, avec nom du dataset et scope approprié.
- [ ] Conserver les identifiants et rendre explicite la provenance des réponses.
- [ ] Vérifier manuellement un échantillon documentation/code : cohérence, grounding, pertinence et absence d'invention majeure.
- [ ] Tester une question hors corpus ; observer l'abstention sans la considérer comme garantie par le seul prompt.

**Validation :** JSON conforme et complet, génération effective avec Qwen, vérification qualitative documentée séparément du recall.

### Étape 16 — Robustesse et répétition de l'évaluation

**But :** exécuter le système proprement dans les conditions de soutenance.

**Fichiers :** `src/__main__.py`, `src/cli.py`, composants concernés, `tests/test_cli.py`, `Makefile`.

- [ ] Tester les six commandes via le shell, pas seulement leurs fonctions Python.
- [ ] Essayer question vide, espaces seuls, termes inconnus, `k=0`, négatif, trop grand et non entier.
- [ ] Essayer taille de chunk invalide, option mal orthographiée, index absent/corrompu, JSON mal formé et mauvais schéma.
- [ ] Essayer fichier source absent, borne hors fichier, sortie inaccessible et dossier de sortie à créer.
- [ ] Vérifier messages, codes de sortie, interruption clavier et absence de traceback non géré.
- [ ] Vérifier que l'aide Fire ne déclenche pas d'indexation ni de chargement du modèle.
- [ ] Confirmer que tous les chemins d'entrée/sortie sont configurables.
- [ ] Exécuter tests et `make lint` ; essayer `make lint-strict` si utile.
- [ ] Répéter `uv sync` puis le parcours du sujet dans un environnement propre, idéalement Linux CPU comparable à celui de l'évaluation.
- [ ] Mesurer à nouveau les performances finales et vérifier les résultats avec la moulinette.
- [ ] Vérifier qu'aucune commande de la solution n'appelle la moulinette.

**Validation :** obligatoire complet, robuste, reproductible et mesuré ; aucun seuil présenté comme acquis par héritage de V1.

### Étape 17 — Documentation et soutenance

**But :** permettre à un tiers d'exécuter le projet et savoir expliquer nos décisions.

**Fichiers :** finaliser `README.md`, `notes/defense.md`, `notes/experiments.md`, `roadmap.md`.

- [ ] README en anglais ; première ligne italique : `This project has been created as part of the 42 curriculum by <login>` avec le vrai login.
- [ ] Sections Description, Instructions, Resources et détail honnête de l'usage de l'IA par tâche/composant.
- [ ] Décrire architecture, chunking, retrieval, performances mesurées, choix, difficultés et exemples réels.
- [ ] Documenter installation, préparation des données, six commandes, chemins, modèle et prérequis matériels.
- [ ] Expliquer les limites du lexical, de Qwen et de nos mesures.
- [ ] Vérifier fichiers suivis : pas d'archives volumineuses, corpus, poids, index ou sorties générées.
- [ ] Expliquer tout le pipeline sans notes et retrouver dans le code chaque mécanisme important.
- [ ] S'exercer à un petit changement : taille de chunk, paramètre BM25, affichage ou cas limite.
- [ ] Faire relire et discuter le projet avec un pair, comme recommandé par le sujet.

**Validation :** un pair exécute le parcours documenté ; l'élève peut justifier les décisions et effectuer une petite modification.

## 5. Commandes cibles pour le parcours complet

Ces commandes sont des objectifs à implémenter, **pas des commandes déjà fonctionnelles ni des tests déjà exécutés**. Partir de la racine du projet après installation des données. Refaire recherche et évaluation avec `dataset_code_public.json` pour le code.

```bash
uv sync
make lint

uv run python -m src index --max_chunk_size 2000

uv run python -m src search "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Vérification officielle séparée, exécutée manuellement sur un système compatible, une fois l'exécutable fourni et préparé :

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

## 6. Bonus — après validation complète de l'obligatoire

Chaque bonus vaut un point dans le sujet et doit fonctionner sur CPU. Les noms de fichiers ci-dessous sont des suggestions à confirmer au moment de l'implémentation.

| Bonus | Fichiers possibles | Travail et preuve attendue |
|---|---|---|
| Semantic embeddings | `src/semantic.py`, tests dédiés | Modèle léger, vecteurs et index ; mesurer recall, mémoire et latence |
| Hybrid retrieval | `src/hybrid.py`, tests dédiés | Fusion des classements lexicaux/sémantiques ; comparer aux deux composants seuls |
| Incremental indexing | compléter `src/indexer.py`, modèles et tests | Détecter ajouts, modifications et suppressions ; conserver des statistiques cohérentes |
| Caching | `src/cache.py`, tests dédiés | Cache d'index et de requêtes ; définir invalidation, comparer démarrage froid et chaud |
| Local HTTP API | `src/api.py`, tests dédiés | Exposer recherche et réponse ; valider entrées, erreurs et réutilisation des ressources |

Pour chaque bonus : expliquer le besoin, définir le contrat, implémenter par incréments, mesurer, documenter et démontrer. Aucun service externe ni framework supplémentaire sans besoin concret.

## 7. Suivi de progression

### Point de reprise du 17 novembre 2026 — état courant

Ce point remplace les indications de prochaine mission et les états historiques
ci-dessous lorsqu'ils les contredisent. Les anciennes mesures restent historiques.

- Code inspecté : `ScoredChunk`, tokenisation, `Retriever` avec
  `rank_bm25.BM25Okapi`, `search`, `search_dataset` et `evaluate` implémentés.
  Les tests datasets et évaluation sont présents, ainsi que leurs tests CLI.
- L'élève rapporte `make install` puis une indexation à `max_chunk_size=2000`.
  La durée de cette indexation n'a pas été communiquée.
- Sorties de la moulinette transmises par l'élève : données valides,
  99 questions code et 100 questions docs, toutes avec des résultats.

| Dataset public | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Seuil Recall@5 |
|---|---:|---:|---:|---:|---:|
| Code | 43,4 % | 59,6 % | 66,7 % | 75,8 % | 50 % : atteint |
| Docs | 59,0 % | 76,0 % | 81,0 % | 85,0 % | 80 % : atteint |

- Notre `evaluate --k 5` concorde avec la moulinette sur ces deux datasets :
  code 66,67 %, docs 81,00 %. Aucune question manquante ; respectivement
  33 et 19 questions à recall nul. Cela ne prouve pas l'identité des deux
  évaluateurs sur tous les cas limites, notamment IoU exactement égale à 0,05.
- Qualité du retrieval validée sur ces datasets publics pour cette configuration.
  Temps d'indexation ≤ 300 s et retrieval ≤ 90 s pour 200 questions encore
  à documenter ; les deux datasets présents totalisent 199 questions.
- L'élève confirme les 16 tests de `tests/test_prompting.py` réussis et
  `make lint` réussi après intégration de `answer`. La suite complète n'a pas
  été réexécutée lors de ce point ; ne pas extrapoler un nombre total de tests.
- Étapes 8 à 11 implémentées ; étape 12 : baseline mesurée, sans tuning requis
  avant la génération. Compréhension à poursuivre au fil des expériences.
- Étape 13 : `src/prompting.py` implémenté avec messages et sources Pydantic,
  comptage via le chat template Qwen et sélection d'un préfixe de chunks entiers
  selon le budget. Refus si question et instructions seules dépassent le budget.
  Tokenizer simulé dans les tests ; vrai tokenizer utilisé en exploration.
- Étape 14 : `src/generator.py` charge Qwen/Qwen3-0.6B une seule fois,
  génère sans thinking ni échantillonnage et choisit le périphérique via
  `device="auto"` : CUDA disponible en float16, sinon CPU en float32.
  Modèle et entrées sont déplacés sur le même périphérique. PyTorch CUDA,
  transformers et jinja2 installés ; poids dans le cache hors Git.
- Première génération réelle rapportée : question endpoint LoRA, réponse correcte
  `/v1/load_lora_adapter`, 1126 tokens d'entrée, 25 tokens générés, chargement
  depuis cache 0,65 s et génération 4,58 s. Plafond de sortie non atteint.
  Consigne de citation non suivie : pas de `[Source 1]` dans la réponse.
- `answer` intégré à la CLI avec chemins configurables, budget, sauvegarde
  temporaire puis remplacement du JSON, erreurs attendues présentées par CliError.
  Exécution réelle rapportée : 10,36 s ; JSON affiché avec question_id, question,
  réponse correcte, trois sources incluses et k=3. Validation limitée à cet exemple.
- Compréhension travaillée : tokenizer versus modèle, budget versus quota API,
  chat template, tenseurs et shape[-1], tokens d'entrée versus tokens générés,
  sources incluses versus citations produites, cache disque versus mémoire.
- `answer` : cinq cas de tests avec générateur simulé rapportés réussis, puis
  six cas d'arguments invalides via Fire rapportés réussis. Ils ne constituent
  pas des tests dédiés de tous les comportements internes de `Generator`.
- GPU vérifié par l'élève sur RTX 4050 Laptop 6 Go, pilote 595.99.02,
  PyTorch 2.14.1+cu130 : modèle sur cuda:0 en float16, réponse LoRA correcte.
  Aucune durée GPU ni accélération par rapport au CPU mesurée à ce stade.
- Étape 15 en cours : `load_source_chunk` relit les sources avec la convention
  d'ingestion et contrôle chemins, longueur et bornes ; `answer_questions`
  traite les résultats existants avec un générateur reçu, conserve l'ordre et
  les question_id et renvoie un StudentSearchResultsAndAnswer en mémoire.
  L'élève confirme sept tests datasets réussis après correction result/results.
- **Mission actuelle : compléter les cas limites de la boucle batch**, puis
  ajouter la commande Fire `answer_dataset` : lecture/validation du JSON,
  chemins configurables, modèle chargé une fois, périphérique affiché,
  sauvegarde JSON et erreurs sans traceback. Elle n'existe pas encore dans la CLI.
  Vérifier un petit lot réel avant le dataset complet. Prévoir aussi l'option
  CLI `--device` sur `answer`, qui utilise actuellement le défaut auto.
- Restent : génération batch, tests négatifs, essais qualitatifs multi-questions,
  benchmarks retrieval/indexation documentés, parcours complet et installation
  propre, README anglais et soutenance. Pas de plafond chiffré de durée pour la
  génération dans le sujet : les 90 s concernent le retrieval de 200 questions.
- Dernier lint observé par le tuteur : 55 fichiers sans erreur ; 27 tests
  answer/prompting/datasets réussis avant le dernier ajout batch. Pas de nouvelle
  exécution de la suite complète lors de la reprise du 17 novembre.
- Aucun bonus commencé. Le GPU ne compte pas comme bonus du sujet.
  Conserver la baseline BM25 actuelle.
- Estimation pédagogique de l'obligatoire : environ 80 % (fourchette 75–85 %),
  sans valeur de note officielle ni de proportion du temps restant. Le pipeline
  unitaire fonctionne ; le batch et la validation globale restent substantiels.

### Historique jusqu'au 12 novembre 2026

Mise à jour du 12 novembre 2026 : vérification par le tuteur, **92 tests réussis** (16 modèles, 7 ingestion, 48 chunking, 6 indexer, 6 CLI, 9 tokenizer) et `make lint` réussi (33 fichiers mypy). `ChunkIndex` est écrit et testé : configuration bornée, cohérence des tailles et aller-retour JSON. `build_index` orchestre découverte, lecture et chunking, conserve les sources relatives et refuse un corpus sans chunks. Trois tests d'indexer passent. save_index et load_index sont écrits, avec aller-retour sur disque, validation et tests fichier absent/données incohérentes. Aucun benchmark retrieval ni génération exécuté.

Chunking texte/Markdown : paragraphes, sections hors blocs clôturés, regroupement et repli fixe implémentés et testés. Exploration de `lora.md` réalisée par l'élève : 14 chunks à 2000 caractères et 48 à 500, reconstruction du texte et offsets vérifiés par le script. Le découpage ne préserve pas nécessairement tout bloc de code dépassant la limite et n'est pas un parseur Markdown complet.

Chunking Python : `statement_spans` accepte une plage interne avec offsets absolus ; `pack_spans` regroupe les petits blocs ; `child_statements` fournit les instructions directement contenues ; `python_spans` descend récursivement dans les groupes trop grands, sinon utilise le repli texte. Décorateurs et commentaires adjacents sont conservés au début des instructions. Le repli sur erreur de parsing est présent. Les tests couvrent notamment classe subdivisée aux méthodes et grande affectation. L'élève rapporte l'exploration réussie d'un vrai fichier Python référencé par le dataset ; sa sortie complète n'a pas été analysée ensemble.

`chunk_file` sélectionne `.py`, `.md` ou `.txt`, refuse les autres extensions et applique une taille entre 1 et 2000 caractères. Tests du dispatch et des tailles invalides réussis. Exploration complète exécutée par l'élève via `notes/explore_full_corpus.py` : **1969 fichiers validés, 0 échec, 16736 chunks à 2000 caractères, 10,92 s**. Le script vérifie provenance, continuité, bornes, texte exact et reconstruction du texte normalisé. L'élève confirme `make lint` réussi après cet ajout. Mesure rapportée, non réexécutée par le tuteur lors de cette mise à jour. Le nombre de replis texte après échec de parsing n'est pas mesuré. Cette durée ne comprend pas une sauvegarde d'index et ne valide pas la qualité du retrieval. La construction en mémoire est désormais disponible via `build_index` ; sauvegarde et rechargement sont implémentés ; la commande CLI `index` reste à construire.

Estimation pédagogique du périmètre complet, **bonus inclus : environ 35 %**, avec une fourchette de 30–40 %. Pour l'obligatoire seul, estimation indicative de 40–45 %. Il ne s'agit ni d'un barème officiel, ni d'une proportion du temps restant, ni d'une note : les composants de retrieval, mesure, génération et les cinq bonus restent à construire. Aucun bonus commencé et aucune validation de bout en bout acquise.

Ingestion réalisée : découverte triée des fichiers `.py`, `.md`, `.txt` (1969 candidats), lecture UTF-8 avec normalisation standard des fins de ligne, conversion des chemins relativement à une racine explicite et refus des chemins externes. Les erreurs de lecture remontent pour être gérées par le futur orchestrateur ; les avertissements pour fichiers ignorés restent à construire. Les cas fichier absent, vide et erreurs de permissions ne sont pas tous couverts par les tests actuels.

Exploration des fins de ligne : 1840 fichiers LF sans CRLF, 125 sans LF, 4 contenant CRLF ; aucune référence publique ne cite ces quatre fichiers. La compatibilité des offsets sur d'éventuelles références CRLF n'est donc pas démontrée. Convention choisie : même lecture normalisée pour chunking et relecture des sources. Les chemins et le passage LoRA de 1403 caractères ont été vérifiés ; `.txt` est nécessaire notamment pour `CMakeLists.txt`. Aucune exclusion de dossier n'est encore décidée.

Le Makefile utilise désormais correctement `src` dans `run` et `debug` (vérifié par lecture). La gestion globale des erreurs CLI et la vérification de la moulinette restent à faire.

Compréhension travaillée : sources et positions, identifiants, validation Pydantic imbriquée, piège de l'union, modèles de sorties, rôle des tests, chemins, normalisation CRLF, découpage fixe et correspondances regex. Priorité : définir ChunkIndex, construire et sauvegarder les chunks, puis brancher la commande index.

Programme du 10 novembre — étape 7, indexation persistante :

1. Comprendre et créer `ChunkIndex` dans `src/models.py` : chunks + configuration, cohérence des tailles, tests et aller-retour JSON.
2. Créer `src/indexer.py` : orchestrer découverte, lecture et chunking avec tqdm ; définir le traitement des erreurs et du corpus sans contenu exploitable.
3. Sauvegarder sous `data/processed/` puis recharger avec validation Pydantic ; utiliser une écriture temporaire puis remplacement pour éviter un fichier final partiellement écrit. Tester sur un petit corpus et observer avec un script `notes/`.
4. Brancher la commande Fire `index --max_chunk_size`, avec chemins configurables et messages d'erreur propres.
5. Exécuter construction + sauvegarde sur le corpus, vérifier le rechargement, mesurer durée complète et taille disque. Comparer au budget officiel de 300 s (VI.1, VII.1.2). Les 10,92 s historiques ne couvrent que l'exploration du chunking.

Objectif de séance : une préparation persistée et rechargeable, expliquée et testée. Si le temps et la compréhension le permettent, introduire ensuite la tokenisation BM25 ; le retrieval complet n'est pas promis pour cette séance. Le sujet impose les comportements et les budgets ; JSON, `ChunkIndex` et l'écriture temporaire sont nos choix d'ingénierie.

Dans le tableau, `—` signifie non applicable et `[ ]` non validé. Ajouter une preuve ou un lien vers la mesure avant de cocher une validation technique.

| Étape | État | Code écrit | Tests réussis | Compréhension | Conformité |
|---|---|---|---|---|---|
| 0. Fondamentaux et sujet | En cours | — | — | Partielle | Audit initial effectué |
| 1. Données | En cours | Scripts exploratoires écrits | Expériences exécutées | Chemins et sources compris | Vérifications restantes |
| 2. Socle Python / CLI | En cours | Socle et Fire écrits | CLI et lint rapportés réussis | uv et Fire compris | Erreurs CLI à gérer |
| 3. Contrats Pydantic | Socle validé | 8 modèles officiels + AnsweredDataset | 11 tests confirmés par exécution | Contrats et union compris | Modèles internes à ajouter progressivement |
| 4. Ingestion | Fonctions de base validées | Découverte, lecture, chemins | 7 tests confirmés par exécution | Tests, CRLF et chemins travaillés | Gestion orchestrée des erreurs restante |
| 5. Chunking Markdown/texte | Stratégie initiale validée | split_markdown intégré | 24 tests chunking texte/Markdown réussis | Frontières et états expliqués | Invariants validés sur les 1969 candidats à taille 2000 |
| 6. Chunking Python | Stratégie initiale et dispatch implémentés | Récursion, regroupement, replis, chunk_file | 48 tests chunking au total réussis | Mécanismes travaillés progressivement | Invariants validés sur les 1969 candidats à taille 2000 |
| 7. Indexation persistante | Premier parcours validé | Construction, sauvegarde, rechargement, CLI index | 6 indexer + 6 CLI réussis | Persistance et dispatch travaillés | 11,54 s rapportées ; robustesse finale à poursuivre |
| 8. Tokenisation / BM25 | En cours | Tokenizer snake_case/CamelCase ; statistiques exploratoires | 9 tests tokenizer réussis | TF, DF, IDF, saturation et longueur étudiés | Tests CamelCase supplémentaires et scoring complet restants |
| 9. Retrieval unitaire | À faire | [ ] | [ ] | [ ] | [ ] |
| 10. Recherche batch | À faire | [ ] | [ ] | [ ] | [ ] |
| 11. Évaluation | À faire | [ ] | [ ] | [ ] | [ ] |
| 12. Expériences retrieval | À faire | [ ] | [ ] | [ ] | [ ] |
| 13. Contexte et prompt | À faire | [ ] | [ ] | [ ] | [ ] |
| 14. Qwen / réponse unitaire | À faire | [ ] | [ ] | [ ] | [ ] |
| 15. Réponses batch | À faire | [ ] | [ ] | [ ] | [ ] |
| 16. Robustesse / parcours complet | À faire | [ ] | [ ] | [ ] | [ ] |
| 17. Documentation / soutenance | À faire | [ ] | [ ] | [ ] | [ ] |
| Bonus | Après obligatoire | [ ] | [ ] | [ ] | [ ] |

Mesure rapportée par l'élève via `notes/explore_index_round_trip` : 16736 chunks construits et rechargés, données identiques, JSON de 23278344 octets ; construction 11,63 s, sauvegarde 0,10 s, préparation environ 11,73 s, rechargement 0,10 s. Ce parcours utilise encore un script d'exploration, pas la CLI finale.

Préparation CLI après relecture des sections V.4 et VI.5–VI.7 : six commandes requises (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`), Fire, chemins configurables, tqdm, JSON de recherche/réponse conforme et erreurs sans traceback non géré. Ne jamais appeler la moulinette depuis la solution. Ajouter les commandes progressivement lorsque leurs composants existent ; commencer uniquement par `index`. Prévoir les erreurs métier au niveau CLI et leur présentation dans `__main__.py`.

État de fin de séance du 10 novembre : `RagCli.index` et la présentation des `CliError`/interruptions dans `__main__.py` sont implémentés. Six tests CLI vérifient le lancement réel, les tailles incorrectes, les chemins configurables et le corpus absent. L'élève a exécuté la commande officielle : 16736 chunks sauvegardés en 11,54 s ; taille 0 refusée sans traceback. Ces résultats ne constituent pas une garantie sur une machine différente ni une validation de toute la CLI future. Un test de succès utilise actuellement `notes/` comme corpus et dépend de sa présence.

Tokenisation : extraction `[A-Za-z0-9_]+`, terme entier en minuscules et composants snake_case/CamelCase, répétitions conservées, texte source inchangé. Explorations `notes/explore_tokenizer.py` et `notes/explore_tokenized_chunk.py` réalisées par l'élève. Sur le passage LoRA : 1405 caractères et 255 tokens ; fréquences observées après expansion. La regex et le maintien de la casse avant découpage ont été expliqués. Limite connue : LoRA → lo/ra et vLLM → v/llm ajoutent potentiellement du bruit à mesurer par recall. La suite actuelle contient 9 cas tokenizer ; les cas dédiés supplémentaires AdapterManager, LLMEngine, Qwen3, vLLM restent à compléter malgré l'exploration réussie.

Statistiques BM25 : `notes/explore_bm25_statistics.py` calcule TF, DF et longueur moyenne sur trois passages. L'élève distingue TF (occurrences dans un chunk) et DF (nombre de chunks contenant le terme), comprend le comptage par ensemble pour DF et la moyenne des longueurs en tokens (10/3). Aucun scoring BM25 ni recall implémenté/mesuré. Aucun bonus commencé.

Progression du 12 novembre : expériences `notes/explore_idf.py`, `notes/explore_tf_saturation.py` et `notes/explore_lenght_normalization.py` exécutées par l'élève. IDF positive lissée étudiée : ln(1 + (N - DF + 0,5)/(DF + 0,5)) ; cette formule pédagogique ne doit pas être supposée identique à celle d'une future bibliothèque. Saturation : le facteur augmente avec TF mais les gains diminuent ; k1=1,5 est un réglage initial. Normalisation : b=0 ignore la longueur, b=0,75 la prend en compte ; à TF identique, le chunk plus court reçoit une contribution plus élevée. Le score complet n'est pas encore implémenté. Pas de retriever, pas de classement réel ni de recall mesuré. Les expériences ne constituent pas une validation des performances de recherche.

**Prochaine mission :** réunir IDF et facteur TF normalisé dans un score pédagogique sur trois passages réellement tokenisés et une question, afficher les contributions par terme et le classement. Puis confirmer le choix d'implémentation BM25 : conserver la logique V1 avec `rank_bm25.BM25Okapi` après examen de sa variante IDF et de ses effets sur petit corpus, ou implémenter explicitement une autre variante autorisée et justifiée. Compléter les cas CamelCase manquants dans les tests. Construire `ScoredChunk`, `src/retriever.py` et `search`, puis `search_dataset` et `evaluate` avant toute optimisation. Les seuils restent 80 % docs et 50 % code à recall@5, indexation ≤ 300 s et 200 questions ≤ 90 s.

Jalons suivants : (1) retrieval lexical et sorties JSON ; (2) évaluation indépendante et comparaison manuelle moulinette, puis tuning mesuré ; (3) contexte borné, prompt et Qwen, commandes answer/answer_dataset ; (4) robustesse globale, README anglais, répétition de l'évaluation et soutenance ; (5) cinq bonus uniquement après obligatoire validé : sémantique, hybride, incrémental, cache et API locale.
