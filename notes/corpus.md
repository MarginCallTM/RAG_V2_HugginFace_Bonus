# Exploration du corpus

## Données observées

Le corpus se trouve dans `data/raw/vllm-0.10.1/`.

Fichiers candidats avant exclusions :
- Python : 1763 fichiers `.py`.
- Markdown : 178 fichiers `.md`.
- Texte : 28 fichiers `.txt`.

Les références publiques citent :
- Documentation : 97 sources `.md` et 3 sources `.txt`.
- Code : 99 sources `.py`.

Les trois références `.txt` proviennent de `CMakeLists.txt`.

## Choix initial d'ingestion

Prendre en charge `.py`, `.md` et `.txt`.

Ce choix couvre les extensions des références publiques observées.
Il ne garantit pas la couverture de toutes les questions futures.

Aucune exclusion de dossier n'est encore décidée.

## Stratégies de chunking envisagées

- Python : suivre la structure du code, notamment fonctions,
  classes et décorateurs.
- Markdown : suivre les titres et paragraphes, en distinguant
  les titres des commentaires dans les blocs de code.
- Texte : examiner les blocs et lignes du contenu.

Les petites unités pourront être regroupées.
Les unités trop longues devront être subdivisées.

Ces choix restent à implémenter et à évaluer.

## Invariants des sources

- Conserver le chemin relatif exact attendu par le dataset.
- Utiliser des positions en caractères : début inclus, fin exclue.
- Conserver `chunk.text == original[start:end]`.
- Ne retourner aucune source de plus de 2000 caractères.

## Expérience réalisée

Le passage de référence LoRA a été lu depuis :
`data/raw/vllm-0.10.1/docs/features/lora.md`.

Bornes : `[4695:6098]`.
Longueur observée : 1403 caractères.
Le passage contient l'endpoint `/v1/load_lora_adapter`.