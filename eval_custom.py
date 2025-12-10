"""
Custom Dataset Retrieval Evaluation Script

Evaluates embedding models on your own Japanese test data and compares with MTEB results.

Usage:
    # Evaluate with default example data
    python eval_custom.py --models minilm bge-small

    # Evaluate with your own data
    python eval_custom.py --data-dir ./data/my_dataset --models minilm bge-small

    # Compare with existing MTEB results
    python eval_custom.py --models minilm --mteb-results ./results/20251207_111150/comparison_results.csv

Data Format:
    data_dir/
    ├── corpus.jsonl   # {"_id": "doc1", "text": "..."}
    ├── queries.jsonl  # {"_id": "q1", "text": "..."}
    └── qrels.tsv      # query-id<TAB>corpus-id<TAB>score
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Model definitions (same as eval_retrieval.py)
MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "bge-small": {
        "name": "bge-small-en-v1.5",
        "hf_name": "BAAI/bge-small-en-v1.5",
    },
    "e5-small": {
        "name": "multilingual-e5-small",
        "hf_name": "intfloat/multilingual-e5-small",
    },
    "ruri-v3-30m": {
        "name": "ruri-v3-30m",
        "hf_name": "cl-nagoya/ruri-v3-30m",
    },
    # "kalm": {
    #     "name": "KaLM-Embedding-Gemma3-12B-2511",
    #     "hf_name": "tencent/KaLM-Embedding-Gemma3-12B-2511",
    # },
    # "qwen": {
    #     "name": "Qwen3-Embedding-8B",
    #     "hf_name": "Qwen/Qwen3-Embedding-8B",
    # },
    # "nemotron": {
    #     "name": "llama-embed-nemotron-8b",
    #     "hf_name": "nvidia/llama-embed-nemotron-8b",
    # },
}


def load_corpus(path: Path) -> dict:
    """
    Load corpus from JSONL file

    Args:
        path: Path to the corpus JSONL file

    Returns:
        corpus: Dictionary of corpus documents
            {
                "doc1": "...",
                "doc2": "...",
                ...
            }

    Returns:
        corpus: Dictionary of corpus documents
    """
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus[doc["_id"]] = doc["text"]
    return corpus


def load_queries(path: Path) -> dict:
    """
    Load queries from JSONL file

    Args:
        path: Path to the queries JSONL file

    Returns:
        queries: Dictionary of queries
            {
                "q1": "...",
                "q2": "...",
                ...
            }
    """
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                query = json.loads(line)
                queries[query["_id"]] = query["text"]
    return queries


def load_qrels(path: Path) -> dict:
    """
    Load relevance judgments from TSV file

    Args:
        path: Path to the qrels TSV file

    Returns:
        qrels: Dictionary of relevance judgments
            {
                "q1": {
                    "doc1": 1,
                    "doc2": 0,
                    ...
                },
                "q2": {
                    "doc1": 0,
                    "doc2": 1,
                    ...
                },
                ...
            }
    """
    qrels = {}
    df = pd.read_csv(path, sep="\t")
    for _, row in df.iterrows():
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        score = int(row["score"])
        
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = score
    return qrels


def dcg_at_k(relevances: list, k: int) -> float:
    """
    Calculate DCG@k

    Args:
        relevances: List of relevance scores
        k: Top k documents to consider

    Returns:
        dcg: Discounted Cumulative Gain@k
    
    Example:
        relevances = [1, 0, 1, 0, 1]
        k = 3
        dcg = 1 + 0 + 1 / log2(3) + 0 + 1 / log2(4) + 0 + 1 / log2(5)
        dcg = 1 + 0 + 0.5 + 0 + 0.25 + 0 + 0.125
        dcg = 1.875
    """
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    
    # DCG = sum(rel_i / log2(i + 2)) for i in 0..k-1
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(relevances / discounts)


def ndcg_at_k(retrieved_ids: list, qrel: dict, k: int) -> float:
    """
    Calculate NDCG@k for a single query

    Args:
        retrieved_ids: List of retrieved document IDs
        qrel: Dictionary of relevance judgments
        k: Top k documents to consider

    Returns:
        ndcg: Normalized Discounted Cumulative Gain@k
        Example:
            retrieved_ids = ["doc1", "doc2", "doc3"]
            qrel = {
                "doc1": 1,
                "doc2": 0,
                "doc3": 1,
            }
            k = 3
            ndcg = (1 + 0 + 1 / log2(3) + 0 + 1 / log2(4) + 0 + 1 / log2(5)) / (1 + 0 + 1 / log2(3) + 0 + 1 / log2(4) + 0 + 1 / log2(5))
    """
    # Get relevance scores for retrieved documents
    relevances = [qrel.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    
    # Calculate DCG
    dcg = dcg_at_k(relevances, k)
    
    # Calculate ideal DCG (sorted by relevance)
    ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def recall_at_k(retrieved_ids: list, qrel: dict, k: int) -> float:
    """
    Calculate Recall@k for a single query

    Args:
        retrieved_ids: List of retrieved document IDs
        qrel: Dictionary of relevance judgments
        k: Top k documents to consider

    Returns:
        recall: Recall@k

        Example:
            retrieved_ids = ["doc1", "doc2", "doc3"]
            qrel = {
                "doc1": 1,
                "doc2": 0,
                "doc3": 1,
            }
            k = 3
            recall = (1 + 1) / (1 + 0 + 1)
            recall = 2 / 2
            recall = 1.0
    """
    relevant_docs = set(doc_id for doc_id, score in qrel.items() if score > 0)
    retrieved_relevant = set(retrieved_ids[:k]) & relevant_docs
    
    if len(relevant_docs) == 0:
        return 0.0
    
    return len(retrieved_relevant) / len(relevant_docs)


def mrr(retrieved_ids: list, qrel: dict) -> float:
    """
    Calculate MRR (Mean Reciprocal Rank) for a single query


    Args:
        retrieved_ids: List of retrieved document IDs
        qrel: Dictionary of relevance judgments

    Returns:
        mrr: Mean Reciprocal Rank

        Example:
            retrieved_ids = ["doc1", "doc2", "doc3"]
            qrel = {
                "doc1": 1,
                "doc2": 0,
                "doc3": 1,
            }
            mrr = (1 + 0.5 + 0.333) / 3
            mrr = 1.833 / 3
            mrr = 0.611
    """
    relevant_docs = set(doc_id for doc_id, score in qrel.items() if score > 0)
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    
    return 0.0


def evaluate_retrieval(
    model: SentenceTransformer,
    corpus: dict,
    queries: dict,
    qrels: dict,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate retrieval performance

    Args:
        model: SentenceTransformer model
        corpus: Dictionary of corpus documents
        queries: Dictionary of queries
        qrels: Dictionary of relevance judgments
        batch_size: Batch size for encoding

    Returns:
        results: Dictionary of evaluation results
            {
                "ndcg_at_10": float,
                "ndcg_at_5": float,
                "recall_at_10": float,
                "recall_at_5": float,
                "mrr": float,
            }
    """
    
    # Prepare corpus
    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())
    
    # Prepare queries
    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    
    print(f"  Encoding {len(corpus_texts)} documents...")
    corpus_embeddings = model.encode(
        corpus_texts, 
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    print(f"  Encoding {len(query_texts)} queries...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    # Calculate similarity
    print("  Computing similarities...")
    similarities = cosine_similarity(query_embeddings, corpus_embeddings)
    
    # Calculate metrics for each query
    results = {
        "ndcg_at_10": [],
        "ndcg_at_5": [],
        "recall_at_10": [],
        "recall_at_5": [],
        "mrr": [],
    }
    
    for i, query_id in enumerate(query_ids):
        if query_id not in qrels:
            continue
        
        # Get ranking
        scores = similarities[i]
        ranked_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [corpus_ids[idx] for idx in ranked_indices]
        
        qrel = qrels[query_id]
        
        results["ndcg_at_10"].append(ndcg_at_k(ranked_doc_ids, qrel, 10))
        results["ndcg_at_5"].append(ndcg_at_k(ranked_doc_ids, qrel, 5))
        results["recall_at_10"].append(recall_at_k(ranked_doc_ids, qrel, 10))
        results["recall_at_5"].append(recall_at_k(ranked_doc_ids, qrel, 5))
        results["mrr"].append(mrr(ranked_doc_ids, qrel))
    
    # Calculate averages
    avg_results = {k: np.mean(v) for k, v in results.items()}
    avg_results["num_queries"] = len(results["ndcg_at_10"])
    
    return avg_results


def load_model(model_key: str, dtype: str = "auto") -> SentenceTransformer:
    """
    Load model

    Args:
        model_key: Key from MODELS dict
        dtype: torch dtype (float32, float16, bfloat16, auto)

    Returns:
        model: SentenceTransformer model
    """
    model_info = MODELS[model_key]
    hf_name = model_info["hf_name"]
    
    print(f"\nLoading model: {model_info['name']}")
    
    model_kwargs = {}
    if dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    model = SentenceTransformer(
        hf_name,
        model_kwargs=model_kwargs,
        trust_remote_code=True,
    )
    
    print(f"Model loaded on device: {model.device}")
    return model


def create_comparison_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comparison chart between custom and MTEB results

    Args:
        df: DataFrame of evaluation results
        output_dir: Output directory

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # NDCG@10 comparison
    pivot_ndcg = df.pivot(index="model_name", columns="dataset", values="ndcg_at_10")
    pivot_ndcg.plot(kind="bar", ax=axes[0], width=0.7)
    axes[0].set_title("NDCG@10: Custom Data vs MTEB", fontweight="bold")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("NDCG@10")
    axes[0].set_ylim(0, 1)
    axes[0].legend(title="Dataset")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Recall@10 comparison
    if "recall_at_10" in df.columns:
        pivot_recall = df.pivot(index="model_name", columns="dataset", values="recall_at_10")
        pivot_recall.plot(kind="bar", ax=axes[1], width=0.7)
        axes[1].set_title("Recall@10: Custom Data vs MTEB", fontweight="bold")
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("Recall@10")
        axes[1].set_ylim(0, 1)
        axes[1].legend(title="Dataset")
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_chart.png", dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {output_dir / 'comparison_chart.png'}")
    plt.close()




def main():
    parser = argparse.ArgumentParser(description="Evaluate models on custom retrieval data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/example",
        help="Directory containing corpus.jsonl, queries.jsonl, and qrels.tsv",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=["minilm", "bge-small", "e5-small"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--mteb-results",
        type=str,
        default=None,
        help="Path to MTEB comparison_results.csv for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_custom",
        help="Output directory",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="auto",
        help="Data type for model loading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load data
    print("Loading data...")
    corpus = load_corpus(data_dir / "corpus.jsonl")
    queries = load_queries(data_dir / "queries.jsonl")
    qrels = load_qrels(data_dir / "qrels.tsv")
    
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Qrels: {len(qrels)} query-document pairs")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    all_results = []
    
    for model_key in args.models:
        model = load_model(model_key, dtype=args.dtype)
        
        print(f"\nEvaluating {MODELS[model_key]['name']}...")
        results = evaluate_retrieval(
            model, corpus, queries, qrels,
            batch_size=args.batch_size,
        )
        
        results["model_key"] = model_key
        results["model_name"] = MODELS[model_key]["name"]
        results["dataset"] = "custom"
        all_results.append(results)
        
        print(f"  NDCG@10: {results['ndcg_at_10']:.4f}")
        print(f"  Recall@10: {results['recall_at_10']:.4f}")
        print(f"  MRR: {results['mrr']:.4f}")
        
        # Release memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create results DataFrame
    df_custom = pd.DataFrame(all_results)
    
    # Load MTEB results if provided
    df_comparison = None
    if args.mteb_results:
        print(f"\nLoading MTEB results from: {args.mteb_results}")
        df_mteb = pd.read_csv(args.mteb_results)
        
        # Aggregate MTEB results by model
        df_mteb_agg = df_mteb.groupby("model_name").agg({
            "ndcg_at_10": "mean",
            "recall_at_10": "mean",
            "mrr_at_10": "mean",
        }).reset_index()
        df_mteb_agg["dataset"] = "MTEB"
        df_mteb_agg = df_mteb_agg.rename(columns={"mrr_at_10": "mrr"})
        
        # Combine results
        df_custom_subset = df_custom[["model_name", "ndcg_at_10", "recall_at_10", "mrr", "dataset"]]
        df_comparison = pd.concat([df_custom_subset, df_mteb_agg], ignore_index=True)
    
    # Save results
    df_custom.to_csv(output_dir / "custom_results.csv", index=False)
    print(f"\nResults saved to: {output_dir / 'custom_results.csv'}")
    
    if df_comparison is not None:
        df_comparison.to_csv(output_dir / "comparison.csv", index=False)
        print(f"Comparison saved to: {output_dir / 'comparison.csv'}")
        
        # Create comparison chart
        create_comparison_chart(df_comparison, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY (Custom Japanese Data)")
    print("="*60)
    print(df_custom[["model_name", "ndcg_at_10", "recall_at_10", "mrr"]].to_string(index=False))
    
    if df_comparison is not None:
        print("\n" + "="*60)
        print("COMPARISON: Custom Data vs MTEB")
        print("="*60)
        pivot = df_comparison.pivot(
            index="model_name",
            columns="dataset",
            values="ndcg_at_10"
        )
        print("\nNDCG@10:")
        print(pivot.to_string())

if __name__ == "__main__":
    main()

