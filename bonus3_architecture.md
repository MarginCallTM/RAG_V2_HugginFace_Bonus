# Bonus 3 — Architecture et roadmap de l'indexation incrémentale

Document de reprise préparé le 23 novembre 2026 pour poursuivre sur MacBook.
À lire avec `subject.pdf`, `tutor_agent_RAG_42.md` et `bonus_rag.md`.

## 1. Point de reprise et règles de travail

Le mandatory est implémenté. La recherche sémantique et la fusion hybride
pondérée sont intégrées et évaluées sur CPU. Le bonus 3 n'est pas commencé.

Configuration hybride retenue dans `src/retrieval.py` :

- `candidate_k=20`, `rank_constant=0` ;
- `lexical_weight=0.65`, `semantic_weight=0.35` ;
- recall@5 public : docs **83 %**, code **64,65 %** ;
- référence BM25 : docs 81 %, code 66,67 % ; BM25 reste le mode par défaut.

Dernière suite complète exécutée : 285 tests réussis, flake8 réussi et mypy
réussi sur 77 fichiers. Ce bilan ne constitue pas une validation du futur bonus 3.
Les tests persistants spécifiques aux poids et au chargement hybride restent
à compléter selon `bonus_rag.md`.

Un smoke test de génération avec trois questions docs et leurs sources
hybrides a ensuite été effectué par l'élève via `answer_dataset --device cpu`.
Le JSON a été contrôlé : identifiants conservés, trois réponses non vides,
sources valides. Réponses LoRA et `LLM.embed` correctes ; commande `lm_eval`
incomplète, notamment sans `add_bos_token=True`. Aucune citation `[Source N]`
dans les réponses. Cela valide l'intégration sur cet échantillon, pas une
précision globale ni des réponses parfaites.

**Premier travail demain : définir le manifeste puis détecter les changements,
sans modifier les index existants.**

L'élève souhaite un tutorat en français, théorie avant code, petits blocs à
recopier avec emplacements précis, commandes et analyse des résultats.
Ne pas modifier la codebase sans demande explicite. Ne pas commit/push à sa place.
Les fichiers de ce document sont des propositions : vérifier le dépôt réel
avant de les créer. Aucun nouveau framework ou service n'est nécessaire.

## 2. Exigence officielle et choix de stratégie

Le chapitre IX demande de réindexer le fichier modifié plutôt que reconstruire
systématiquement tout l'index. Tous les bonus doivent fonctionner sur CPU.

Notre proposition : conserver un manifeste d'empreintes par fichier, comparer
le corpus actuel au dernier état, réutiliser les données inchangées, traiter
seulement les ajouts/modifications et retirer les suppressions.

Le manifeste, SHA-256, les versions de préparation et le mécanisme de
publication sont des choix d'ingénierie, pas des formats imposés par le sujet.

## 3. Architecture générale

```text
Corpus actuel                       Dernier état sauvegardé
      │                              ├── manifeste
      │                              ├── chunks
      │                              └── segments et vecteurs
      ↓                                      │
Empreintes des fichiers                       │
      └──────────── comparaison ──────────────┘
                           ↓
                 Plan de mise à jour
         ┌───────────┬───────────┬───────────┐
      inchangés    ajoutés    modifiés    supprimés
         │           └─────┬─────┘           │
      réutiliser       recalculer          retirer
         └─────────────────┬─────────────────┘
                           ↓
               Assemblage du nouvel état
                           ↓
                 Validation de cohérence
                           ↓
              Publication du nouvel état
```

Responsabilités proposées, à implémenter progressivement :

| Composant | Responsabilité |
|---|---|
| `src/manifest.py` (nouveau, proposé) | Modèles Pydantic, empreintes, état des fichiers et comparaison. |
| `src/incremental.py` (nouveau, proposé) | Plan de mise à jour, réutilisation et assemblage des chunks/vecteurs. |
| `src/indexer.py` (existant) | Construction complète conservée et services de persistance à faire évoluer. |
| `src/semantic_index.py` (existant) | Validation et persistance sémantiques réutilisées. |
| `src/cli.py` (existant) | Options incrémentales et reconstruction complète, après validation du cœur. |

La répartition exacte et les signatures seront décidées au moment de chaque
incrément ; ne pas créer tous les fichiers ou toutes les classes d'un coup.

## 4. Manifeste et détection

Le manifeste doit enregistrer les chemins relatifs exacts et une empreinte
SHA-256 par fichier, ainsi que les paramètres et versions affectant l'index.

Calculer l'empreinte à partir du texte lu par `read_text()` et réencodé en UTF-8,
afin de suivre la même normalisation des fins de ligne que le chunking.
Les mêmes règles de découverte `.py`, `.md`, `.txt` et le même ordre trié
doivent être utilisés pour une construction complète et incrémentale.

| Comparaison avec le manifeste précédent | Catégorie |
|---|---|
| Chemin présent et empreinte identique | Inchangé |
| Chemin absent de l'ancien état | Ajouté |
| Chemin présent et empreinte différente | Modifié |
| Ancien chemin absent du corpus actuel | Supprimé |

Un renommage sera initialement une suppression suivie d'un ajout.
Suivre aussi les fichiers qui ne produisent aucun chunk, par exemple un
fichier vide. Ne pas déduire le manifeste uniquement de la liste des chunks.

Première version : lire les fichiers pour calculer les empreintes. Le gain
consiste à éviter le redécoupage et le réencodage des fichiers inchangés.
Les dates et tailles pourront servir d'optimisation ultérieure, mais ne seront
pas la seule preuve d'identité du contenu.

Si le manifeste manque, créer une base complète cohérente. Ne pas supposer
qu'un ancien index sans manifeste décrit nécessairement le corpus actuel.

## 5. Mise à jour des chunks

L'unité de remplacement est le **fichier entier**, pas la ligne modifiée.
Une insertion peut décaler toutes les bornes de caractères suivantes.

- Fichiers inchangés : réutiliser leurs anciens chunks.
- Ajouts/modifications : appeler le chunking existant sur leur nouveau texte.
- Suppressions : éliminer leurs anciens chunks.
- Assembler dans le même ordre déterministe qu'une reconstruction complète.
- Valider les chemins, les plages et la taille maximale de 2 000 caractères.

Une lecture impossible ou un corpus final sans contenu indexable doit produire
une erreur explicite sans publier un état partiellement mis à jour.

## 6. Réutilisation des segments et vecteurs

Format actuel : `TokenSegment.chunk_position` référence un indice dans
`SemanticIndexData.chunks` ; la ligne i du tenseur correspond au segment i.
Ces positions ne sont pas des identifiants stables entre deux générations.

Stratégie initiale :

1. Associer les anciens chunks, segments et lignes de vecteurs à leurs fichiers.
2. Conserver les segments et valeurs des vecteurs des fichiers inchangés.
3. Segmenter et encoder seulement les chunks des fichiers ajoutés/modifiés.
4. Retirer les données des fichiers supprimés ou remplacés.
5. Construire la nouvelle liste ordonnée de chunks.
6. Remapper tous les `chunk_position` vers cette nouvelle liste.
7. Assembler segments et lignes des vecteurs dans un ordre strictement aligné.
8. Recalculer les compteurs et lancer les validations existantes.

Réutiliser un vecteur ne signifie donc pas conserver son ancien numéro de ligne.
Ne pas réutiliser les données d'un fichier modifié sur la seule base de ses
anciens offsets. Une optimisation par sous-chunks identiques viendrait plus tard.

Le format global peut encore nécessiter de recopier des vecteurs en mémoire
et de réécrire le fichier final. Nous évitons leur **calcul par MiniLM**, sans
promettre un coût total indépendant de la taille du corpus.

## 7. BM25 et compatibilité des configurations

BM25 dépend de statistiques globales : ajout ou suppression de chunks peut
changer les fréquences documentaires et la longueur moyenne. Le chargement
actuel reconstruit ces statistiques ; conserver ce comportement au départ.
Ne pas prétendre que toutes les statistiques sont mises à jour localement.

| Changement | Invalidation prévue |
|---|---|
| Taille maximale, stratégie/version de chunking, règles de lecture/découverte | Reconstruction des chunks concernés, potentiellement complète, et des vecteurs associés. |
| Modèle/révision, segmentation, budget de tokens, pooling ou normalisation | Reconstruction des données sémantiques concernées, potentiellement complète. |
| Taille de lot seule | Ne pas l'assimiler automatiquement à un changement sémantique ; documenter les éventuels écarts numériques. |
| Poids ou constante RRF, k de recherche | Aucun recalcul des chunks ou embeddings. |

Conserver un moyen explicite de reconstruire complètement l'index. Ne pas
modifier les commandes mandatory ni basculer implicitement le défaut BM25.

## 8. Publication cohérente et erreurs

Une sauvegarde atomique de chaque fichier séparément ne garantit pas une
transaction entre manifeste, chunks et vecteurs.

Avant le branchement CLI, concevoir une publication d'ensemble : préparer
une nouvelle génération dans un emplacement distinct, valider tous ses
artefacts, puis activer cette génération par un seul remplacement de pointeur.
Les chargeurs devront continuer de comprendre les chemins et contrats
mandatory ; la migration et les tests seront précisés à cette étape.

Ne pas désigner la nouvelle génération active tant que son encodage ou sa
sauvegarde n'est pas terminé. Après une erreur, conserver l'ancien état
chargeable. Hypothèse initiale : un seul écrivain ; ne pas présenter cette
solution comme une gestion complète de plusieurs indexations concurrentes.

Le contenu haché doit correspondre au contenu découpé. Éviter de relire sans
contrôle un texte qui aurait changé entre le scan et le traitement ; utiliser
le même texte ou vérifier sa stabilité avant publication. Pour les sources
relues à la génération Qwen, un corpus modifié impose aussi la mise à jour des
résultats de recherche : des offsets de résultats anciens peuvent être périmés.

## 9. Roadmap et validations

| Étape | Travail | Preuve attendue |
|---|---|---|
| 1 | Modèles du manifeste et empreintes | Texte identique → empreinte identique ; texte changé → empreinte différente ; chemins/configuration validés. |
| 2 | Classification des fichiers | Ajout/modification/suppression/renommage/aucun changement, sans modifier les index. |
| 3 | Mise à jour des chunks | Même résultat qu'une reconstruction complète du même corpus. Aucun appel au chunking pour les fichiers inchangés. |
| 4 | Mise à jour sémantique | Pas de réencodage des fichiers inchangés ; sources, segments et vecteurs correctement associés après remappage. |
| 5 | Publication cohérente | Échec simulé : ancien état toujours utilisable, aucun mélange de générations. |
| 6 | Intégration CLI | Chemins configurables, reconstruction complète disponible, erreurs propres et progression. |
| 7 | Mesures et documentation | Comparaison complète/incrémentale, tests, lint et démonstration CPU reproductible. |

Critère décisif : après les mêmes modifications du corpus, l'index incrémental
et une reconstruction complète doivent être équivalents. Comparer exactement
les chunks et métadonnées déterministes ; utiliser une tolérance justifiée pour
les vecteurs recalculés par lots et conserver exactement les vecteurs réutilisés.
Vérifier aussi les résultats de recherche sur un corpus contrôlé.

Scénarios à couvrir progressivement : aucun changement, ajout, insertion en
début de fichier, suppression, renommage, fichier vide, configuration modifiée,
échec de lecture/écriture, décalage des positions de chunks après suppression.
Privilégier des encodeurs simulés pour les tests déterministes et réserver
quelques essais réels à l'intégration et aux mesures CPU.

## 10. Reprise sur MacBook

- Récupérer le commit contenant ces documents ; ne pas copier le `.venv` Linux.
- Vérifier `git status`, le HEAD, `uv sync` et les tests dans l'environnement Mac.
- Les données, poids, index et sorties sont ignorés par Git : leur présence
  sur cette machine n'est pas garantie par le clone.
- Le cache MiniLM doit exister sur le Mac ou être téléchargé au premier usage ;
  les premières étapes manifeste/diff ne nécessitent ni MiniLM ni Qwen.
- Conserver le mode CPU pour les bonus, y compris sur Mac ; ne pas activer MPS.
- Ne pas reconstruire les vecteurs complets pour commencer le manifeste.

**Premier incrément :** relire le sujet et les modèles existants, définir les
modèles Pydantic du manifeste, puis observer les empreintes de quelques petits
fichiers temporaires. Ensuite seulement ajouter la détection des changements.
