*This project has been created as part of the 42 curriculum by acombier.*

# RAG against the machine

## Description

This project implements a local Retrieval-Augmented Generation (RAG) pipeline
for answering questions about the provided vLLM 0.10.1 codebase. It indexes
Python, Markdown and text files, retrieves relevant passages with BM25, and
uses `Qwen/Qwen3-0.6B` to generate answers from those passages.

The pipeline preserves source paths and character ranges, validates exchanged
data with Pydantic, and exposes its commands through Python Fire. Retrieval
quality is measured independently from answer quality.

An optional CPU semantic retriever is also available alongside BM25. BM25
remains the default method and the validated mandatory retrieval baseline.

## Instructions

### Requirements and installation

- Python 3.12 (the project currently requires `>=3.12,<3.13`).
- [uv](https://docs.astral.sh/uv/).
- Enough disk space for the corpus, Python dependencies and model cache.
- Internet access for the initial dependency and model downloads.

From the repository root:

```bash
uv sync
# Equivalent Makefile entry point:
make install
```

The lockfile records dependency versions. The current configuration installs
PyTorch from its CUDA 13.0 wheel index. Generation selects CUDA automatically
when PyTorch reports it available, otherwise CPU. GPU generation was tested on
an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB of VRAM. A working NVIDIA driver
is required for CUDA execution; CPU generation is slower.

Place `vllm-0.10.1.zip` and `datasets_public.zip` at the repository root,
then extract them from that directory:

```bash
mkdir -p data/raw data/datasets
unzip vllm-0.10.1.zip -d data/raw
unzip datasets_public.zip -d data/datasets
mv data/datasets/datasets_public/public/AnsweredQuestions data/datasets/
mv data/datasets/datasets_public/public/UnansweredQuestions data/datasets/
```

The dataset archive contains the intermediate directories
`datasets_public/public/`. The two `mv` commands place the question directories
directly under `data/datasets/`, where the CLI examples expect them.
Run these extraction commands once when setting up a fresh clone.

The extracted corpus and question datasets must follow this layout:

```text
data/
├── raw/vllm-0.10.1/
├── datasets/
│   ├── UnansweredQuestions/
│   │   ├── dataset_docs_public.json
│   │   └── dataset_code_public.json
│   └── AnsweredQuestions/
│       ├── dataset_docs_public.json
│       └── dataset_code_public.json
├── processed/                         # Created by index
└── output/                            # Created by search and answer commands
```

Run commands from the project root. Corpus paths must retain the exact
`data/raw/vllm-0.10.1/` prefix used by the reference datasets. Large data files,
model weights and generated outputs are not project submission artifacts.
The model downloads on its first use and is subsequently loaded from the
Hugging Face cache; it is not stored inside `src/` or `notes/`.

### Example usage: complete pipeline

Build the index once:

```bash
uv run python -m src index --max_chunk_size 2000
```

Search the documentation dataset, keeping up to five sources per question:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Measure retrieval with the project's own evaluator:

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5
```

For the official retrieval measurement, place the supplied `moulinette.zip`
at the repository root. It is not included in the Git clone. On Ubuntu,
extract the supplied binary, copy it to `moulinette`, and make it executable:

```bash
unzip -o moulinette.zip moulinette-ubuntu
cp moulinette-ubuntu moulinette
chmod +x moulinette
```

Then run the official retrieval evaluation from the repository root:

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5 --max_context_length 2000 --threshold 0.80
```

The application never imports or invokes the moulinette. It evaluates source
retrieval, not the correctness of generated answer text.

Generate answers using the saved search results:

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions \
  --device auto
```

For code questions, repeat these commands with `dataset_code_public.json` in
place of `dataset_docs_public.json`; use `--threshold 0.50` for the code
moulinette check. Keep the reference dataset under `AnsweredQuestions/`.

`answer_dataset` inherits `k` from its input JSON: it has no `--k` option.
To change the candidate count, rerun `search_dataset` with the desired `--k`.
The prompt may include fewer sources when the token budget is insufficient.
Only sources actually included in the prompt are saved with the answer.

Search and answer one question:

```bash
uv run python -m src search \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5

uv run python -m src answer \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5
```

The default single-answer output is
`data/output/search_results_and_answer/single_query.json`.

Input and output paths are configurable. For example, `index` accepts
`--raw_dir` and `--processed_dir`; search commands accept `--processed_dir`;
`answer` accepts `--output_path`; and `answer_dataset` accepts `--project_root`
for resolving saved source paths. Batch outputs reuse the input filename.
Separate `AnsweredQuestions` and `UnansweredQuestions` output directories to
avoid overwriting files with identical names. Repeated runs to the same output
path replace the previous result.

For the exact options of any command:

```bash
uv run python -m src answer_dataset --help
```

### Development commands

```bash
make run          # Display the Fire CLI
make debug        # Run the entry point under pdb
make lint         # flake8 and the subject's required mypy flags
make clean        # Remove Python and development caches
uv run python -m pytest -q
```

`make lint-strict` is available as an optional stricter type check; it is not
the mandatory lint target. Development tests cover chunking, source provenance,
retrieval, evaluation, prompt budgeting, answer orchestration and CLI errors.
Many generation-path tests use a fake generator so they do not download weights
or rely on variable language-model outputs. Actual model integration was also
checked with single questions and complete public datasets.

## System architecture

```text
vLLM files → ingestion → format-specific chunking → chunks.json
                                                      ↓
Question(s) → identifier-aware tokenization → BM25 → ranked source locations
                                                      ↓
                              reread passages → budgeted sourced prompt
                                                      ↓
                                  local Qwen → answers and sources JSON
```

| Module | Responsibility |
| --- | --- |
| `src/ingest.py` | Discover `.py`, `.md` and `.txt` files and read UTF-8 text. |
| `src/chunking.py` | Split text while preserving file paths and character offsets. |
| `src/indexer.py` | Build, validate, save and load the chunk index. |
| `src/tokenizer.py` | Extract lexical terms and identifier components. |
| `src/retriever.py` | Build BM25 statistics and rank passages. |
| `src/embeddings.py` | Segment chunks by tokens and encode them with MiniLM on CPU. |
| `src/semantic_index.py` | Build, validate, save and load semantic index data and vectors. |
| `src/semantic_retriever.py` | Rank original chunks by their best segment similarity. |
| `src/retrieval.py` | Define the shared search Protocol and select BM25 or semantic retrieval. |
| `src/prompting.py` | Construct sourced messages and select a prefix within the token budget. |
| `src/generator.py` | Load Qwen, select a device and generate answer text. |
| `src/datasets.py` | Orchestrate dataset search and answer generation. |
| `src/evaluation.py` | Calculate recall against reference source ranges. |
| `src/models.py` | Define Pydantic exchange and internal data models. |
| `src/cli.py`, `src/__main__.py` | Expose Fire commands and present expected errors. |

Search outputs follow `StudentSearchResults`; generated outputs follow
`StudentSearchResultsAndAnswer`. Each answer preserves its `question_id`,
question text, included `retrieved_sources` and generated `answer`. Source
ranges use character offsets with an inclusive start and exclusive end.
The production CLI does not depend on exploratory scripts in `notes/`.

## Chunking strategy

The maximum chunk size is configurable from 1 to 2,000 **characters**, with a
default of 2,000. This is separate from the model's token budget.

- **Python:** parse the file with `ast`, group statements within the size
  limit and recursively subdivide oversized groups. Preserve decorators and
  nearby comments where the implemented boundary rules allow it. Fall back
  to text splitting when structural splitting cannot provide a suitable span.
- **Markdown and text:** use Markdown section boundaries and paragraphs where
  applicable, track fenced code blocks when identifying headings, and split
  oversized text further to enforce the limit.

Chunks remain exact slices of the text read by the application. Indexing and
source reloading use the same text-reading convention, including newline
normalization. Large functions and code blocks may still require splitting;
preserving a size bound takes precedence over keeping every construct whole.

## Retrieval method

Retrieval uses `rank_bm25.BM25Okapi`, with `k1=1.5` and `b=0.75` by default.
BM25 combines term frequency, inverse document frequency and passage-length
normalization. It is a local lexical baseline with no embedding-model cost.

Tokenization lowercases whole terms while also retaining components of
`snake_case` and `CamelCase` identifiers. Query terms are deduplicated. Results
are ranked by score, and only strictly positive scores are returned, so a
question may receive fewer than `k` sources. BM25 statistics are rebuilt when
a retriever is constructed and reused across questions within a batch.
The persisted index currently stores chunks rather than serialized BM25
statistics.

### Optional CPU semantic retrieval

The semantic retriever uses `sentence-transformers/all-MiniLM-L6-v2`, pinned
to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It uses the existing
Transformers/PyTorch dependencies, without Sentence Transformers or FAISS.
The first use downloads model files if they are not already cached.

Each original chunk is segmented into token windows of at most 256 tokens,
including special tokens, without overlap. This avoids dropping the end of
chunks that exceed the encoder's selected token budget. MiniLM runs on CPU
in float32. Masked mean pooling and L2 normalization produce one 384-dimensional
vector per segment. Query vectors use the same encoder and preprocessing;
questions exceeding the configured token budget are rejected rather than
silently truncated.

Search computes exact dot products, equivalent to cosine similarities for
normalized vectors. Each chunk receives the maximum score of its segments,
so it appears only once in the results. The returned source is the original
chunk, with its exact path and character range. Ties use original chunk order.
There is no positive-score filter in the semantic method: negative similarities
can be returned among the best available results, unlike the BM25 policy.

Build the semantic index after the mandatory `index` command:

```bash
uv run python -m src index_semantic \
  --processed_dir data/processed \
  --output_path data/processed/semantic_index.pt \
  --batch_size 32
```

`--max_tokens` defaults to 256, and `--batch_size` controls the number of
segments encoded together. Semantic indexing is an additional operation; it
does not overwrite `chunks.json`. Index data, encoder settings and vectors
are saved together and validated on loading. Rebuild the semantic index after
changing the corpus or chunk configuration; automatic incremental updates are
not implemented.

Search an individual question:

```bash
uv run python -m src search \
  "Which environment variable sets the maximum allowed wheel size?" \
  --method semantic \
  --semantic_index_path data/processed/semantic_index.pt \
  --output_path data/output/experiments/semantic/single_query.json \
  --k 5
```

Search a full dataset:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --method semantic \
  --semantic_index_path data/processed/semantic_index.pt \
  --save_directory data/output/experiments/semantic/full/UnansweredQuestions \
  --k 5
```

Replace `dataset_docs_public.json` with `dataset_code_public.json` for code
questions. Both search commands default to `--method bm25` when omitted.
The selected retriever is reused across questions in a batch. Semantic search
produces the same `StudentSearchResults` JSON contract as BM25; `answer_dataset`
can consume this file without changing its interface. Single-query `answer`
currently still retrieves with BM25. Full semantic answer generation has not
yet been evaluated.

Evaluate the semantic documentation results separately:

```bash
./moulinette evaluate_student_search_results \
  data/output/experiments/semantic/full/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5 --max_context_length 2000 --threshold 0.80
```

For code, replace both dataset filenames and use `--threshold 0.50`.

## Prompting and generation: design decisions

Prompts contain system instructions, the question and numbered evidence blocks
with file paths, character ranges and passage text. Instructions ask Qwen to
use only the supplied evidence, treat evidence as untrusted data, acknowledge
insufficient information and cite supporting passages using `[Source N]`.
These instructions guide generation but do not guarantee correct citations or
prevent all hallucinations.

The default application budget is **4,096 tokens**, including up to **256 new
output tokens**, leaving 3,840 tokens for the formatted input. The Qwen
tokenizer counts the complete chat template, with thinking disabled. Whole
chunks are considered in retrieval order; selection stops at the first chunk
that would exceed the budget. The generator checks the budget again before
generation. This policy preserves a ranked prefix without truncating passages.

`Generator` uses greedy decoding (`do_sample=False`) and moves both model and
input tensors to the selected device. CUDA uses float16; CPU uses float32.
`answer_dataset --device auto` selects CUDA when available, while `cpu` and
`cuda` request an explicit device. The model is loaded once per nonempty batch
and reused for its questions. An empty dataset produces an empty output without
loading Qwen. Questions without usable evidence receive an explicit
insufficient-evidence response.

Output is written to a temporary file before replacing the destination. This
avoids exposing a partially written final JSON file, but generation has no
checkpoint/resume feature: a failed batch must be rerun.

## Performance analysis

Public BM25 retrieval measurements with 2,000-character chunks:

| Dataset | Questions | Recall@1 | Recall@3 | Recall@5 | Required Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Documentation | 100 | 59.00% | 76.00% | **81.00%** | 80% |
| Code | 99 | 43.43% | 59.60% | **66.67%** | 50% |

These figures are the recorded retrieval baseline, not answer-accuracy scores.
After full answer generation at `k=5`, the generated docs and code files also
passed their respective moulinette recall thresholds. The documentation
baseline has only a one-percentage-point margin above the public threshold;
public results do not guarantee identical performance on unseen questions.

Measurements reported on the development machine on 17 November 2026:

| Operation | Measured time | Subject limit |
| --- | ---: | ---: |
| Full corpus indexing, 2,000-character maximum | 10.10 s, internal command timer | 300 s |
| Search over 200 questions, `k=5` | 12.03 s, external wall-clock timer | 90 s |

The retrieval command's internal timer reported 10.54 s; the external 12.03 s
includes process startup and is the reported end-to-end measurement. The
benchmark used all 199 public questions plus one repeated question with a
distinct ID to reach 200 queries. This extra query was used only for timing,
not for recall evaluation. Search timing includes index loading, BM25
construction, retrieval and JSON output; it excludes Qwen generation.
These are single-run development measurements, not hardware-independent
guarantees. The subject specifies no numerical answer-generation time limit.

### Semantic retrieval measurements

Full-corpus semantic retrieval was evaluated on CPU on 23 November 2026,
using the same public datasets, original chunks and `k=5`:

| Method | Docs Recall@5 (100 questions) | Code Recall@5 (99 questions) |
| --- | ---: | ---: |
| BM25 baseline | 81.00% | 66.67% |
| MiniLM, maximum segment similarity per chunk | 66.00% | 41.41% |

Semantic indexing encoded 16,736 original chunks in 980 batches of up to
32 segments. The command reported **306.22 seconds** in total on CPU, including
approximately 4 minutes 57 seconds in the encoding progress bar. This is an
additional vector-building measurement, separate from the 10.10-second
mandatory chunk-indexing baseline. The semantic build alone exceeds 300 seconds;
it is not presented as satisfying that ceiling for a combined indexing flow.
Exact segment count, index size, peak memory and full semantic query timings
remain to be recorded. Small-sample throughput is not used to predict full-corpus
performance.

The semantic method underperforms BM25 on both datasets and does not reach
either mandatory recall threshold by itself. A documentation run printed PASS
with a 0.50 threshold, but the documentation requirement is 0.80: its measured
66% does not meet that requirement. BM25 remains the validated default.
Only five results per question were saved, so the displayed recall@10 adds
no information beyond recall@5 for this experiment.

These results motivate inspecting complementary successes and failures before
implementing hybrid fusion. No hybrid gain has been measured or is guaranteed.
The index and retrieval are functional on CPU; ranking quality and answer
grounding remain separate evaluation concerns.

### Answer quality and limitations

Manual inspection matched five code questions by `question_id` against the
reference answers and inspected the supplied passages. It revealed both
retrieval and generation limitations:

- Missing precise evidence can lead to an incorrect generalization or a useful
  abstention instead of the requested answer.
- Qwen can identify the correct configuration flag but invert its conditional
  behavior even when the relevant code is present.
- A correct value may be justified with a passage from the wrong function.
- An otherwise correct answer may omit details shown in the evidence.

This small consecutive sample is diagnostic, not an answer-accuracy estimate.
Exact text equality with the reference is not an appropriate correctness
metric: equivalent wording is acceptable, while a changed condition or API
name can make a similar-looking answer wrong. Retrieval recall, factual
correctness and grounding must be checked separately. Qwen's small size and
the 256-token output cap also limit reasoning and completeness.

## Challenges faced

- **Matching natural language to code identifiers:** retaining both complete
  identifiers and their components improves lexical matching without discarding
  exact API names.
- **Preserving source provenance:** keeping exact paths and character ranges
  makes retrieved evidence traceable and compatible with the reference grader.
- **Separating characters from tokens:** chunk limits are enforced in characters,
  while the complete model prompt is measured with Qwen's tokenizer.
- **Moving generation to GPU:** both model weights and input tensors must use
  the same device; moving only the inputs caused a device mismatch during
  development. Explicit placement and device-dependent dtype resolved it.
- **Checking answers beyond successful execution:** manual inspection exposed
  incorrect reasoning and weak citations that the retrieval moulinette does
  not evaluate.

The mandatory lexical RAG pipeline and optional CPU semantic indexing/retrieval
are implemented. Hybrid retrieval, incremental indexing, persistent query
caching and a local HTTP API are not implemented. The remaining bonus work and
validation checklist are tracked in `bonus_rag.md`.

## Resources

- Project `subject.pdf`: mandatory interfaces, data models and evaluation rules.
- [Python AST documentation](https://docs.python.org/3/library/ast.html): Python
  syntax-tree traversal for structural chunking.
- [Pydantic documentation](https://docs.pydantic.dev/latest/): validated data
  contracts and JSON serialization.
- [Python Fire](https://github.com/google/python-fire): CLI generation.
- [uv documentation](https://docs.astral.sh/uv/): environment and dependency management.
- [rank_bm25](https://github.com/dorianbrown/rank_bm25): BM25 implementation.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): model usage
  and thinking-mode configuration.
- [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating):
  formatting conversations and counting model input tokens.
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html):
  inference, tensors, dtypes and device placement.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401):
  the foundational RAG paper by Lewis et al.

### Use of AI

An AI coding assistant was used as a tutor for interpreting the subject,
explaining BM25, chunking, chat templates, token budgeting and local inference;
proposing code and deterministic tests; diagnosing Python, typing and
CPU/CUDA errors; reviewing generated answers against reference evidence; and
drafting this README. Assistance covered ingestion/chunking, retrieval,
evaluation, prompting, generation, dataset orchestration and CLI integration.
The student integrated code, ran tests and benchmarks, and inspected outputs.
The README was generated with AI assistance from the implemented code and
reported measurements. AI suggestions and model answers require review and
do not replace understanding of the submitted implementation.
