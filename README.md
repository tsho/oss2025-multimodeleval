# Multi-Model Embedding Evaluation

A tool to compare retrieval accuracy of multiple embedding models using MTEB (Massive Text Embedding Benchmark).

## Supported Models

### Small Models (Recommended for Testing)

| Key | Model Name | HuggingFace | Params | Language | Description |
|-----|------------|-------------|--------|----------|-------------|
| minilm | all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 | 22M | English | Fast, lightweight model for testing |
| bge-small | bge-small-en-v1.5 | BAAI/bge-small-en-v1.5 | 33M | English | BAAI's small BGE model |
| e5-small | multilingual-e5-small | intfloat/multilingual-e5-small | 118M | Multilingual | Microsoft's multilingual E5 model |
| ruri-v3-30m | ruri-v3-30m | cl-nagoya/ruri-v3-30m | 30M | Japanese | Nagoya University's Japanese embedding model |

### WIP, Large Models (8B-12B, GPU Required)

| Key | Model Name | HuggingFace | Params | Language | Min VRAM |
|-----|------------|-------------|--------|----------|----------|
| kalm | KaLM-Embedding-Gemma3-12B-2511 | tencent/KaLM-Embedding-Gemma3-12B-2511 | 12B | Multilingual | 24GB (float16) |
| qwen | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 8B | Multilingual | 16GB (float16) |
| nemotron | llama-embed-nemotron-8b | nvidia/llama-embed-nemotron-8b | 8B | Multilingual | 16GB (float16) |

## Requirements

- Python 3.12+
- For small models: CPU or any GPU
- For large models: CUDA-compatible GPU (16-24GB+ VRAM)

## Installation

```bash
# Using uv
uv sync
```

## Usage

### MTEB Evaluation

```bash
# Evaluate small models with quick test (2 tasks)
uv run eval_retrieval.py --models minilm bge-small e5-small ruri-v3-30m --task-set quick

# Evaluate specific models
uv run eval_retrieval.py --models ruri-v3-30m e5-small

# With memory optimization for large models
uv run eval_retrieval.py --models qwen --dtype float16 --batch-size 4
```

### Task Sets

| Set | Tasks | Description |
|-----|-------|-------------|
| quick | NFCorpus, SciFact | Fast testing (2 tasks) |
| en | NFCorpus, SciFact, TRECCOVID, ArguAna | English retrieval (4 tasks) |
| ja | JaQuADRetrieval, MIRACLRetrieval | Japanese retrieval (2 tasks) |
| all | All above | Complete evaluation |

```bash
uv run eval_retrieval.py --task-set quick
uv run eval_retrieval.py --task-set ja
```

### Custom Tasks

```bash
# Generate test data (1000 documents)
uv run generate_test_data.py --num-docs 1000 --add-negatives

# Evaluate on custom data
uv run eval_custom.py --data-dir ./data/large --models minilm bge-small e5-small ruri-v3-30m

# Compare with MTEB results
uv run eval_custom.py --models minilm bge-small \
    --mteb-results ./results/YYYYMMDD_HHMMSS/comparison_results.csv
```

### Streamlit Dashboard

```bash
# Interactive visualization
uv run streamlit run app.py
```

## Custom Data Format

```
data/your_dataset/
├── corpus.jsonl    # {"_id": "doc1", "text": "..."}
├── queries.jsonl   # {"_id": "q1", "text": "..."}
└── qrels.tsv       # query-id<TAB>corpus-id<TAB>score
```

## Output Structure

```
results/
└── YYYYMMDD_HHMMSS/
    ├── config.json              # Evaluation configuration
    ├── comparison_results.csv   # Comparison results (CSV)
    ├── comparison_results.json  # Comparison results (JSON)
    └── {model_key}/             # Model-specific detailed results
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@10** | Normalized Discounted Cumulative Gain (primary metric) |
| **MRR@10** | Mean Reciprocal Rank |
| **Recall@10/100** | Recall at top 10/100 results |
| **MAP@10** | Mean Average Precision |

## Memory Optimization

For large models on limited hardware:

```bash
# Use float16 (half memory)
uv run eval_retrieval.py --models qwen --dtype float16

# Use smaller batch size (less memory during inference)
uv run eval_retrieval.py --models kalm --dtype float16 --batch-size 1

# Quantization (CUDA only, ~1/4 memory)
uv run eval_retrieval.py --models kalm --quantize 4bit
```

## License

Apache 2.0
