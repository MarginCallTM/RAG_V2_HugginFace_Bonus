# Plan des bonus — RAG against the machine

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

- [ ] Terminer le contrôle final du mandatory et enregistrer un commit de référence.
- [x] Enregistrer les versions, la machine, les commandes, les scores et les temps
      (voir le bilan de référence ci-dessous et les commandes du README).
- [x] Garder BM25 accessible et préserver les signatures minimales du sujet.
- [ ] Utiliser des dossiers de résultats séparés pour chaque méthode et expérience.
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
- Commit de référence et push : à effectuer par l'élève après ces vérifications.

## 1. Bonus : embeddings sémantiques

**Exigé :** ajouter un index vectoriel avec un petit modèle CPU, à côté de
l'index lexical. Le sujet cite `all-MiniLM-L6-v2` comme exemple, sans l'imposer.

**Concept :** représenter questions et passages par des vecteurs ; rapprocher
les vecteurs pour retrouver des passages même sans mots identiques.

- [ ] Comparer un petit exemple lexical et sémantique avant d'intégrer le modèle.
- [ ] Choisir le modèle et sa bibliothèque après lecture de leur documentation ;
      vérifier taille, licence, limite d'entrée, dépendances et fonctionnement CPU.
- [ ] Expliquer la limite du tokenizer d'embeddings : 2 000 caractères ne
      garantissent pas qu'un passage entier tient dans sa fenêtre de tokens.
- [ ] Définir et documenter la politique pour les passages trop longs
      (segmentation/agrégation ou troncature mesurée), sans perdre la provenance.
- [ ] Ajouter uniquement les dépendances retenues avec uv et mettre à jour le lockfile.
- [ ] Définir les métadonnées de l'index : modèle/révision, paramètres, version
      de préparation des textes, dimension et association vecteur ↔ chunk.
- [ ] Encoder les chunks par lots sur CPU avec une progression tqdm.
- [ ] Normaliser les vecteurs pour utiliser une similarité cosinus cohérente.
- [ ] Commencer par une recherche matricielle exacte ; mesurer avant d'ajouter
      une bibliothèque d'index vectoriel plus complexe.
- [ ] Persister vecteurs et métadonnées sous un chemin configurable ignoré par Git.
- [ ] Implémenter une recherche sémantique avec le même contrat de sources que BM25.
- [ ] Ajouter un choix de méthode à la CLI en conservant BM25 par défaut.
- [ ] Tester sans téléchargement : dimensions, ordre, k=0, grand k, index vide,
      association des sources, score et persistance avec encodeur simulé.
- [ ] Faire un test d'intégration avec le vrai modèle forcé sur CPU.
- [ ] Mesurer recall@5 docs/code, temps d'indexation/recherche et mémoire sur CPU.

Validation : index sémantique rechargeable et recherche opérationnelle CPU,
avec sources exactes et comparaison documentée. Une amélioration de recall
n'est pas garantie par l'ajout d'embeddings.

## 2. Bonus : recherche hybride

**Exigé :** combiner les classements lexical et sémantique en un classement unique.

- [ ] Observer les passages retrouvés exclusivement par chaque méthode.
- [ ] Choisir une fusion initiale simple, par exemple Reciprocal Rank Fusion
      (RRF), afin de combiner des rangs sans comparer directement des scores
      BM25 et cosinus dont les échelles diffèrent.
- [ ] Définir le nombre de candidats de chaque méthode, distinct du k final.
- [ ] Dédupliquer par identité de source : chemin et deux bornes de caractères.
- [ ] Fixer un ordre déterministe en cas d'égalité et retourner au plus k sources.
- [ ] Exposer le mode hybride et les paramètres utiles sans casser la CLI obligatoire.
- [ ] Tester la fusion sur des listes contrôlées : source commune, source unique,
      classement vide, égalité, k=0 et grand k.
- [ ] Brancher les sources fusionnées sur le prompt et le générateur existants.
- [ ] Comparer BM25, sémantique et hybride sur les mêmes datasets à recall@5.
- [ ] Vérifier quelques réponses et la gestion du budget avec les sources hybrides.
- [ ] Documenter le compromis temps/qualité et éviter d'ajuster aux seuls cinq
      exemples déjà analysés.

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

Premier incrément recommandé : comprendre et observer les embeddings sur
quelques phrases sur CPU, avant de changer le pipeline mandatory.
