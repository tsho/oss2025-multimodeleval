# Multi-Model Embedding Evaluation

A tool to compare retrieval accuracy of multiple embedding models using MTEB (Massive Text Embedding Benchmark).

## Target Models (Small Models)
| Key | Model Name | HuggingFace |
|-----|------------|-------------|
| minilm | all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 |
| bge-small | bge-small-en-v1.5 | BAAI/bge-small-en-v1.5 |
| e5-small | multilingual-e5-small | intfloat/multilingual-e5-small |

### WIP (Large Models)
| Key | Model Name | HuggingFace |
|-----|------------|-------------|
| kalm | KaLM-Embedding-Gemma3-12B-2511 | tencent/KaLM-Embedding-Gemma3-12B-2511 |
| qwen | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B |
| nemotron | llama-embed-nemotron-8b | nvidia/llama-embed-nemotron-8b |

## Requirements

- Python 3.12+
- CUDA-compatible GPU (24GB+ VRAM recommended)
- Sufficient VRAM is required to run 8B-12B models

## Installation

```bash
# Using uv
uv sync
```

## Usage

### Basic Evaluation

```bash
# Evaluate all models with quick test (2 tasks)
uv run eval_retrieval.py

# Evaluate specific models only
uv run eval_retrieval.py --models minilm
```

### Task Set Selection

```bash
# Quick test (NFCorpus, SciFact)
uv run eval_retrieval.py --task-set quick

# English retrieval tasks (4 tasks)
uv run eval_retrieval.py --task-set en

# Japanese retrieval tasks
uv run eval_retrieval.py --task-set ja

# All tasks
uv run eval_retrieval.py --task-set all
```

### Custom Tasks

```bash
# Specify particular tasks
uv run eval_retrieval.py --tasks NFCorpus SciFact TRECCOVID
```

### Output Directory

```bash
uv run eval_retrieval.py --output-dir ./my_results
```

## Visualizing Results

```bash
# Specify results directory to generate visualizations
uv run visualize_results.py --results-dir ./results/20241207_123456
```

Generated files:
- `ndcg_comparison.png` - NDCG@10 comparison chart by task
- `average_scores.png` - Average scores chart by model
- `heatmap.png` - NDCG@10 heatmap
- `report.md` - Markdown report

## Output Structure

After running evaluation, the following files are generated:

```
results/
└── YYYYMMDD_HHMMSS/
    ├── config.json              # Evaluation configuration
    ├── comparison_results.csv   # Comparison results (CSV)
    ├── comparison_results.json  # Comparison results (JSON)
    ├── kalm/                    # Model-specific detailed results
    ├── qwen/
    └── nemotron/
```

## Evaluation Metrics

- **NDCG@10**: Normalized Discounted Cumulative Gain (primary metric)
- **MRR@10**: Mean Reciprocal Rank
- **Recall@10/100**: Recall at top 10/100 results
- **MAP@10**: Mean Average Precision


## License

Apache 2.0
