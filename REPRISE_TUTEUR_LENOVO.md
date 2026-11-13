# Reprise du projet RAG_v2 — tuteur Codex sur Lenovo Ubuntu

État établi le **13 novembre 2026**, après inspection de la copie de travail sur Mac. Ce document transmet le contexte ; il ne prouve pas que la copie clonée sur Ubuntu possède exactement les mêmes fichiers. Vérifier le dépôt réel avant de poursuivre.

## 1. Message à lire en premier par le nouvel assistant

Tu es le professeur particulier de l'élève en Applied AI Engineering. Le projet est **RAG Against the Machine**, École 42, sujet `subject.pdf` version 2.0. L'élève reconstruit RAG_v2 pour comprendre en profondeur un projet antérieur RAG_v1 largement réalisé avec des agents IA.

L'objectif n'est pas seulement un programme fonctionnel : l'élève doit comprendre et pouvoir défendre chaque mécanisme. Échéance personnelle : **validation avant le 30 avril 2027** ; cette date vient du contrat pédagogique, pas du PDF.

Le travail quitte un Mac Apple Silicon pour un **Lenovo Ubuntu de l'école 42**, qui sera la machine de validation. Ne plus baser les commandes sur Homebrew, macOS ou des chemins `/Users/...`. Toutes les mesures Mac doivent être reproduites sur le Lenovo. Ne pas supposer de GPU ni de droits administrateur.

### Documents à lire dans le dépôt

1. Ce fichier pour retrouver le contexte et le point de reprise.
2. `tutor_agent_RAG_42.md` : contrat pédagogique détaillé.
3. `subject.pdf` : autorité pour les exigences officielles ; lire le document complet si non disponible dans le contexte.
4. `roadmap.md` : progression et plan détaillé, à actualiser selon le code réel.
5. `notes/corpus.md`, puis les modules et tests concernés par la prochaine étape.

`README_rag.md` est la documentation jointe de la moulinette ; il ne remplace pas le futur `README.md` du projet.

## 2. Manière de travailler avec l'élève

- Parler français, garder les noms techniques usuels et les identifiants de code en anglais.
- Niveau Python intermédiaire ; expliquer les constructions Python lorsqu'elles sont nouvelles.
- **Par défaut, montrer le code dans la conversation avec le chemin exact où le recopier. L'élève le retape lui-même dans les fichiers.** Sa demande « code dans le CLI » a été clarifiée : elle ne signifie pas un script Python en heredoc dans le terminal.
- Ne pas implémenter tout le projet à sa place. Un incrément ciblé à la fois.
- Avant chaque implémentation : problème concret, mécanisme, choix et compromis, puis code.
- Après : commande à exécuter, résultat attendu, demande de la sortie réelle et analyse ensemble.
- **Les scripts d'exploration dans `notes/` sont particulièrement utiles à cet élève.** Observer un exemple concret avant ou avec les tests ; utiliser de vrais fichiers quand le mécanisme est acquis sur un exemple simple.
- L'élève apprécie les explications ligne par ligne, pseudo-code, tableaux de positions et fichiers autonomes pour Python Tutor.
- Poser de courtes questions de compréhension. Une réponse « tests verts » ne remplace pas la compréhension ; reprendre sans jugement lorsqu'une confusion apparaît.
- Tu peux lire les fichiers et lancer les vérifications. Modifier directement le code lorsqu'il te le demande explicitement, notamment pour corriger une erreur ; expliquer alors la cause et le diff minimal.
- Une demande explicite de mise à jour de la roadmap ou de création d'un document autorise cette modification documentaire.
- Ne pas installer une grosse pile ML ou télécharger des poids volumineux sans expliquer l'opération et obtenir l'accord prévu par le contrat de tutorat.
- Ne pas faire de commit/push, changement de branche ou suppression sans demande. Ne pas envoyer de messages externes.
- Ne pas annoncer un test ou benchmark comme réussi sans sortie observée. Séparer code écrit, tests réussis, compréhension et conformité officielle.

## 3. Exigences officielles à préserver

Résumé de `subject.pdf` ; le PDF prévaut en cas de doute.

| Exigence | Référence |
|---|---|
| Python ≥ 3.10 ; annotations, docstrings PEP 257, flake8 et mypy | V.1 |
| Exceptions gérées proprement, ressources fermées ; pas de traceback non géré en évaluation | V.1, VI.6 |
| Makefile : install, run, debug, clean, lint ; lint-strict facultatif | V.2 |
| uv comme gestionnaire ; installation évaluateur par `uv sync` | V.4 |
| Modèles de données Pydantic ; classes de service non obligatoirement Pydantic | V.4, VI.4 |
| CLI Python Fire ; tqdm sur les opérations longues | V.4, VI.6 |
| Modèle Qwen/Qwen3-0.6B fonctionnel ; autres modèles possibles seulement si celui-ci reste utilisable | V.4, VI.3 |
| Autres bibliothèques autorisées, dont une bibliothèque BM25 | V.4 |
| Deux stratégies distinctes : Python et Markdown/texte | VI.1 |
| Retrieval lexical TF-IDF ou BM25 obligatoire | VI.1 |
| max_chunk_size configurable, défaut 2000 caractères ; aucune source retournée au-delà de 2000 | VI.1 |
| Chemin relatif exact et bornes en caractères ; chemins comparés textuellement | VI.2, VI.5 |
| Sources et réponses sauvegardées en JSON suivant les modèles officiels | VI.4–VI.5 |
| Tous les chemins d'entrée/sortie configurables | VI.6–VI.7 |
| Indexation du corpus ≤ 300 secondes | VII.1.2 |
| Retrieval de 200 questions ≤ 90 secondes | VII.1.2 |
| Recall@5 docs ≥ 80 %, code ≥ 50 % | VII.1.2 |
| README anglais et contenu imposé, dont usage de l'IA | VIII |
| Pas de corpus volumineux, poids ou résultats générés versionnés | X |

`lint` doit lancer `flake8 .` et `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` dans l'environnement uv.

### Commandes minimales finales

Toutes se lancent depuis la racine via `uv run python -m src` :

```text
index --max_chunk_size <int>
search <query> --k <int>
search_dataset --dataset_path <path> --k <int> --save_directory <dir>
answer <query> --k <int>
answer_dataset --student_search_results_path <path> --save_directory <dir>
evaluate --student_search_results_path <path> --dataset_path <path>
```

`evaluate` calcule notre mesure pour l'itération. La mesure officielle est celle de la moulinette, exécutée séparément à la main. **La solution ne doit jamais importer ni appeler la moulinette.**

Les datasets avec et sans réponses partagent des noms de fichiers : séparer les sorties par scope pour éviter leur écrasement.

### Modèles et arborescence

Modèles officiels : `MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`. Garder les noms de champs du sujet.

```text
src/
pyproject.toml
uv.lock
Makefile
README.md                         # encore à écrire
data/raw/vllm-0.10.1/
data/processed/chunks.json
data/datasets/UnansweredQuestions/
data/datasets/AnsweredQuestions/
data/output/search_results/<DatasetScope>/
data/output/search_results_and_answer/<DatasetScope>/
```

Les datasets publics portent les noms `dataset_docs_public.json` et `dataset_code_public.json`.

Le recall mesure la part des sources attendues retrouvées parmi les k premiers résultats, par question, puis agrégée. Correspondance : même fichier et IoU suffisante entre plages. Le PDF indique le seuil 0,05 ; V1 utilise strictement `> 0.05`. Vérifier les conventions exactes et les cas limites avec la moulinette lors de l'implémentation. Début inclus, fin exclue dans notre code.

Les cinq bonus sont : embeddings sémantiques, recherche hybride, indexation incrémentale, caching, API HTTP locale. Ils ne sont évalués qu'après validation intégrale de l'obligatoire et doivent fonctionner sur machine campus CPU-only.

## 4. État réel vérifié le 13 novembre

Vérifications sur la copie Mac au moment de créer ce document : **111 tests réussis**, `make lint` réussi, **40 fichiers analysés par mypy**. Ce ne sont pas des résultats Ubuntu.

Le dossier inspecté n'a pas de `.git` : `git status` y échoue. L'élève indique avoir créé un nouveau dépôt ailleurs ; ne pas inventer son URL ni considérer que la migration a déjà transféré tous les fichiers.

### Implémenté

- `src/models.py` : modèles officiels et modèles internes `AnsweredDataset`, `Chunk`, `ChunkIndex`, `ScoredChunk`.
- `src/ingest.py` : découverte triée `.py/.md/.txt`, lecture UTF-8, conversion en chemin relatif à une racine explicite.
- `src/chunking.py` : découpage fixe, paragraphes, sections Markdown hors blocs clôturés, regroupement, découpage Python AST récursif, dispatch `chunk_file`.
- `src/indexer.py` : `build_index`, `save_index`, `load_index`.
- `src/tokenizer.py` : termes entiers et composants snake_case/CamelCase.
- `src/retriever.py` : construction BM25 et `search` top-k.
- `src/cli.py` : `index`, `search`, ancienne commande d'exercice `hello`.
- `src/__main__.py` : Fire, présentation des `CliError`, interruption clavier.
- `src/datasets.py` : `load_dataset`, `search_questions`, `save_search_results`.
- Scripts exploratoires dans `notes/`, dont `explore_retriever.py`, `explore_search_dataset.py`, expériences IDF/TF/longueur.

### Point exact de reprise fonctionnelle

Le batch est implémenté dans `src/datasets.py` et observé via `notes/explore_search_dataset.py`. Il a traité 100 questions docs, gardé les identifiants et sauvegardé un `StudentSearchResults`.

**La commande CLI `search_dataset` n'est PAS encore implémentée.**

**`tests/test_datasets.py` n'existe PAS dans la copie inspectée.** Les quatre tests proposés dans la conversation précédente n'ont donc pas encore été intégrés ici : chargement préservant l'ID, aller-retour de sauvegarde, refus de plus de k sources, refus d'une source > 2000. Vérifier si le nouveau dépôt a une version plus récente avant de les reproposer.

Pas encore de `evaluation.py`, pas de recall mesuré, pas de génération Qwen, pas de commandes `answer`, `answer_dataset`, `evaluate`, pas de README final et aucun bonus.

Estimation de travail précédemment communiquée : environ 45 % du périmètre complet avec bonus, marge 40–50 %, et 50–60 % de l'obligatoire. Ce n'est ni un score officiel ni une proportion fiable du temps restant. Ne pas convertir les tests verts en validation fonctionnelle complète.

## 5. Décisions techniques et pièges connus

### Sources et chunking

- `.py`, `.md`, `.txt` : 1969 fichiers candidats, aucune exclusion de dossier. Les dossiers cachés sont inclus.
- Trois références docs citent `CMakeLists.txt`, d'où la prise en charge `.txt`.
- Conserver `chunk.text == texte_lu[start:end]`, longueur cohérente, plages continues, provenance et reconstruction exacte du texte lu.
- UTF-8 strict, lecture texte standard qui normalise les fins de ligne. Même convention pour le chunking et la relecture des sources.
- Exploration : 1840 fichiers LF sans CRLF, 125 sans LF, 4 contenant CRLF ; aucun de ces quatre cité par les références publiques. Compatibilité d'autres références CRLF non démontrée.
- Python : numéros de lignes AST convertis en offsets de caractères ; éviter de traiter les colonnes AST en octets UTF-8 comme des index de caractères.
- Inclure décorateurs et commentaires directement adjacents ; une ligne vide arrête la remontée. Heuristique de proximité, pas compréhension des commentaires.
- `statement_spans` délimite une plage complète ou interne ; `pack_spans` regroupe mais **ne subdivise jamais** ; `python_spans` descend dans les instructions enfants uniquement si nécessaire, puis repli `split_text`/`split_fixed`.
- Markdown : détection ciblée des titres ATX et blocs clôturés, pas parseur Markdown complet. Un grand bloc de code peut être coupé en chunks non exécutables.

### Persistance

- JSON contient les chunks et `max_chunk_size`, pas les statistiques BM25.
- `ChunkIndex` refuse une configuration hors 1..2000 et un chunk plus long que sa configuration.
- Sérialiser produit une chaîne en mémoire ; écrire crée le fichier durable ; charger relit puis reconstruit les objets avec Pydantic.
- Sauvegarde : nom temporaire fixe `.tmp`, fermeture puis remplacement. Hypothèse : un seul écrivain par destination. Pas de garantie de durabilité absolue ni de nettoyage après toutes les pannes.
- Le corpus externe à la racine est refusé pour ne pas produire des sources aux chemins incohérents.

### Tokenizer et BM25

- `WORD = re.compile(r"[A-Za-z0-9_]+")` : tokenizer lexical ASCII, pas celui de Qwen.
- Séparer snake_case/CamelCase avant de perdre la casse, garder le terme entier en minuscules et les composants ; conserver les répétitions réelles.
- LoRA → `lora, lo, ra` ; vLLM → `vllm, v, llm`. Bruit potentiel à mesurer, pas à optimiser à l'aveugle.
- TF = occurrences dans un chunk ; DF = nombre de chunks contenant le terme ; longueur moyenne en tokens lexicaux, pas en caractères.
- L'IDF pédagogique positive utilisée dans `notes/` diffère de celle de `rank_bm25.BM25Okapi`.
- Bibliothèque choisie comme V1 : **rank-bm25**, réglages initiaux explicites k1=1,5 et b=0,75. Ils ne sont pas optimisés automatiquement.
- k1 règle la saturation des répétitions : contribution croissante, gains décroissants. b règle l'influence de la longueur ; b=0 l'ignore.
- Variante bibliothèque : IDF initiale `ln((N-df+0.5)/(df+0.5))`, IDF négatives remplacées par epsilon × IDF moyenne. Une IDF moyenne négative peut encore produire des poids négatifs. Ne pas supposer tous les scores positifs sur petit corpus.
- `Retriever.__init__` tokenise le corpus et prépare les statistiques **sans question** ; `search` tokenise la question, appelle `get_scores`, trie et retourne les résultats.
- Questions : `sorted(set(tokenize(query)))`, donc chaque terme distinct compte une fois. Classement stable en cas d'égalité ; filtrage des scores strictement positifs. Ce choix peut éliminer des correspondances de score nul/négatif.
- Même retriever pour toutes les questions d'un batch ; nouvelle préparation au lancement d'un nouveau processus. Aucun cache BM25 persistant actuellement.
- Scores = pertinence lexicale, pas probabilité ni preuve de réponse correcte.

### Validation et sorties

- `RagDataset` accepte une union ; une référence invalide peut être rétrogradée en `UnansweredQuestion` avec perte des champs ignorés. **Évaluation : charger les corrigés avec `AnsweredDataset`**, pas l'union permissive.
- Conserver les `question_id` des datasets. UUID généré pour une question unitaire sans ID.
- `save_search_results` contrôle k non négatif, nombre de sources ≤ k et taille ≤ 2000. Ne prouve pas le grounding ni l'exactitude des chemins face aux références.
- Dans la copie actuelle, son message pour k négatif dit à tort « A question has more than k sources » : corriger au prochain incrément pertinent, sans confondre cela avec une erreur de score.
- `search` possède encore sa propre écriture JSON : factoriser progressivement avec `save_search_results` et garder les tests.
- Un test CLI `index` utilise `notes/` comme corpus réel : dépend de la présence de ces scripts. Préférer à terme un corpus contrôlé sous la racine.
- Fire peut convertir les arguments ; ne pas supposer que les annotations garantissent les types à l'exécution. Tester les options inconnues et fautes de frappe avant l'évaluation finale.

## 6. Mesures historiques Mac — à reproduire sur Ubuntu

| Expérience | Résultat rapporté par l'élève |
|---|---|
| Chunking complet, taille 2000 | 1969 fichiers, 0 échec, 16736 chunks, 10,92 s |
| Construction + sauvegarde | environ 11,73 s ; JSON 23278344 octets |
| Rechargement JSON | environ 0,10–0,15 s |
| Commande CLI index | 16736 chunks en 11,54 s |
| Préparation BM25 | environ 2,30–2,37 s |
| Vocabulaire BM25 | 56983 termes ; moyenne 178,6639 tokens/chunk |
| Recherche batch docs, 100 questions | total environ 7 s, préparation incluse |
| Question LoRA unitaire | source attendue au rang 1 : lora.md [4695:6100] |

Ces mesures sont indicatives, pas des garanties Lenovo. Le résultat LoRA ne constitue pas une mesure de recall global. **Qwen n'a été ni installé ni exécuté dans ce projet.**

## 7. Première séance sur Lenovo Ubuntu

### A. Vérifier le transfert et l'environnement

Depuis la racine du dépôt cloné :

```bash
pwd
git status --short
python3 --version
uv --version
uv python list --only-installed
uname -m
free -h
df -h .
```

L'environnement actuel cible Python **3.12** : `.python-version` vaut `3.12`, `requires-python` vaut `>=3.12,<3.13`. Ne pas copier `.venv` du Mac : la recréer sur Ubuntu. Ne pas élargir/changer les dépendances sans raison observée.

```bash
uv sync --locked
uv run python --version
uv run python -m pytest -q
make lint
```

Le parcours officiel attend `uv sync` ; `--locked` est ici une vérification supplémentaire de cohérence du lockfile. Si uv/Python manque ou si l'école restreint les installations/réseau, diagnostiquer avec l'élève avant de proposer une installation. Ne pas supposer `sudo` autorisé.

S'assurer que le nouveau dépôt contient ce document, le PDF, le contrat pédagogique, roadmap, sources, tests, scripts notes, Makefile, `.flake8`, `.gitignore`, `.python-version`, pyproject et lockfile. La copie inspectée sur Mac n'avait pas de `.git` : vérifier réellement les fichiers du nouveau dépôt.

### B. Restaurer les pièces jointes hors Git

`.gitignore` exclut `data/`, `.venv/`, caches et archives ZIP. **Un clone seul ne récupérera pas le corpus, les datasets, l'index ni les sorties.** Transférer séparément les pièces jointes autorisées et les installer dans les chemins attendus, en évitant un niveau d'emballage supplémentaire `datasets_public/public/`.

Vérifier par exemple :

```bash
ls data/raw/vllm-0.10.1/docs/features/lora.md
ls data/datasets/UnansweredQuestions/dataset_docs_public.json
ls data/datasets/AnsweredQuestions/dataset_code_public.json
```

Garder la version fournie vLLM 0.10.1, pas un clone récent arbitraire ; les positions des références dépendent du texte précis.

### C. Reproduire la préparation et la recherche

```bash
uv run python -m notes.explore_full_corpus
uv run python -m src index --max_chunk_size 2000
uv run python -m notes.explore_index_round_trip
uv run python -m src search "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5
uv run python -m notes.explore_search_dataset
```

Les scripts d'indexation remplacent les fichiers de sortie à leurs destinations documentées. Vérifier d'abord leurs chemins dans le code cloné. Consigner machine, commande, configuration, durées, mémoire si mesurée et résultats dans `notes/experiments.md` et la roadmap. Ne pas interpréter 100 questions docs comme une validation du budget de 200 questions.

### D. Traiter Qwen comme une étape à valider, pas comme acquise

L'élève veut vérifier Qwen sur le Lenovo. Avant son intégration : examiner RAM, espace disque, architecture, environnement CPU et accès réseau. Expliquer les besoins et obtenir l'accord avant gros téléchargements.

Après choix et vérification du backend local compatible : petite expérience dans `notes/` avec **Qwen/Qwen3-0.6B**, chargement unique, chat template correct, gestion du thinking, limite de nouveaux tokens, test réel CPU et mesures. Vérifier les versions/détails dans les documentations officielles au moment de l'intégration.

Ne pas installer/charger le modèle pendant `index` ou `search`. La génération doit être chargée uniquement pour les commandes qui l'utilisent. Ne pas affirmer que Qwen fonctionnera à une latence donnée sans exécution sur cette machine. Une courte validation matérielle peut être avancée dans le planning à la demande de l'élève, sans prétendre que le pipeline de génération est déjà construit.

## 8. Prochaines étapes fonctionnelles après migration

1. Rejouer les tests et expériences Ubuntu ; résoudre les différences avant de poursuivre.
2. Créer/valider `tests/test_datasets.py` (absent dans la copie source inspectée).
3. Ajouter `RagCli.search_dataset(dataset_path, k, save_directory, processed_dir)` : chargement du dataset, un seul Retriever, search_questions, sauvegarde sous le nom du dataset et le scope configuré ; erreurs CliError et mesure complète.
4. Tester en CLI docs puis code ; mesurer les 200 questions, préparation comprise.
5. Construire `src/evaluation.py` et `evaluate` : IoU, recall multi-sources, identifiants, cas négatifs et comparaison manuelle avec la moulinette Linux fournie.
6. Enregistrer une baseline V2 avant tout tuning. Ne pas reprendre les scores historiques V1 comme résultats V2.
7. Améliorer une variable à la fois pour atteindre les seuils ; expliquer les compromis et éviter le surapprentissage des questions publiques.
8. Contexte/prompt et génération Qwen ; answer et answer_dataset ; distinguer sources récupérées et réellement transmises au modèle.
9. Robustesse, documentation anglaise, usage honnête de l'IA, répétition propre de l'évaluation et préparation au recode/soutenance.
10. Bonus uniquement quand tout l'obligatoire est validé.

## 9. Proposition de premier message à l'élève

« J'ai lu le document de reprise, le contrat pédagogique et le sujet. Nous avons l'ingestion, le chunking, la persistance, la recherche BM25 et le cœur du batch. Nous allons d'abord vérifier le dépôt et rejouer les tests sur ton Lenovo, puis reconstruire l'index sur cette machine. La CLI search_dataset et l'évaluation restent à ajouter. Qwen n'est pas encore intégré : nous vérifierons ses besoins et son fonctionnement CPU ici avant de considérer cette partie acquise. Comme convenu, je te présente le code à recopier et nous analysons tes sorties avant de passer à l'étape suivante. »

Ne pas annoncer avoir lu un fichier ou exécuté une commande sans l'avoir réellement fait dans la nouvelle session.
