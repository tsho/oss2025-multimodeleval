"""
MTEB Retrieval Evaluation Script

Compares retrieval accuracy of embedding models using MTEB:

Usage:
    # Evaluate all models
    python eval_retrieval.py

    # Use float16 to reduce memory (recommended for large models)
    python eval_retrieval.py --dtype float16

    # Use 4-bit quantization for very limited memory
    python eval_retrieval.py --quantize 4bit
"""

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import mteb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# Model definitions
# MODELS = {
#     "kalm": {
#         "name": "KaLM-Embedding-Gemma3-12B-2511",
#         "hf_name": "tencent/KaLM-Embedding-Gemma3-12B-2511",
#         "description": "Tencent's 12B embedding model based on Gemma3",
#         "params": "12B",
#         "min_memory_gb": 24,  # Full precision
#     },
#     "qwen": {
#         "name": "Qwen3-Embedding-8B",
#         "hf_name": "Qwen/Qwen3-Embedding-8B",
#         "description": "Alibaba's 8B embedding model",
#         "params": "8B",
#         "min_memory_gb": 16,
#     },
#     "nemotron": {
#         "name": "llama-embed-nemotron-8b",
#         "hf_name": "nvidia/llama-embed-nemotron-8b",
#         "description": "NVIDIA's 8B embedding model based on LLaMA",
#         "params": "8B",
#         "min_memory_gb": 16,
#     },
# }

# Smaller models for testing (can run on most machines)
SMALL_MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Small 22M model for testing",
        "params": "22M",
        "min_memory_gb": 1,
    },
    "bge-small": {
        "name": "bge-small-en-v1.5",
        "hf_name": "BAAI/bge-small-en-v1.5",
        "description": "Small BGE model (33M)",
        "params": "33M",
        "min_memory_gb": 1,
    },
    "e5-small": {
        "name": "multilingual-e5-small",
        "hf_name": "intfloat/multilingual-e5-small",
        "description": "Small multilingual E5 model",
        "params": "118M",
        "min_memory_gb": 1,
    },
}

# Combine all models
ALL_MODELS = {**SMALL_MODELS}

# Evaluation tasks (Retrieval tasks)
# English retrieval tasks (relatively lightweight)
RETRIEVAL_TASKS_EN = [
    "NFCorpus",       # Medical document retrieval
    "SciFact",        # Scientific claim verification
    "TRECCOVID",      # COVID-related retrieval
    "ArguAna",        # Argument retrieval
]

# Japanese retrieval tasks
RETRIEVAL_TASKS_JA = [
    "JaQuADRetrieval",      # Japanese QA
    "MIRACLRetrieval",      # Multilingual retrieval (includes Japanese)
]

# Quick test tasks
RETRIEVAL_TASKS_QUICK = [
    "NFCorpus",
    "SciFact",
]


def clear_gpu_memory():
    """Release GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        # Force MPS to release memory
        torch.mps.empty_cache()


def get_device_info():
    """Get device information"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {
            "device": device,
            "gpu_name": gpu_name,
            "gpu_memory_gb": round(gpu_memory, 2),
        }
    elif torch.backends.mps.is_available():
        # Get system memory for Apple Silicon (shared memory)
        import subprocess
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            total_memory_gb = int(result.stdout.strip()) / (1024**3)
        except Exception:
            total_memory_gb = "N/A"
        return {
            "device": "mps",
            "gpu_name": "Apple Silicon",
            "gpu_memory_gb": round(total_memory_gb, 2) if isinstance(total_memory_gb, float) else total_memory_gb,
        }
    else:
        return {"device": "cpu", "gpu_name": "N/A", "gpu_memory_gb": "N/A"}


def get_torch_dtype(dtype_str: str):
    """Convert string to torch dtype"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    return dtype_map.get(dtype_str, torch.float32)


def load_model(model_key: str, dtype: str = "auto", quantize: str = None, device: str = None):
    """
    Load model with memory optimization options
    
    Args:
        model_key: Key from MODELS dict
        dtype: torch dtype (float32, float16, bfloat16, auto)
        quantize: Quantization level (4bit, 8bit, None)
        device: Device to load model on (cuda, mps, cpu, auto)

    Returns:
       model: SentenceTransformer model
    """
    model_info = ALL_MODELS[model_key]
    hf_name = model_info["hf_name"]
    
    print(f"\n{'='*60}")
    print(f"Loading model: {model_info['name']}")
    print(f"HuggingFace: {hf_name}")
    print(f"Parameters: {model_info['params']}")
    print(f"Dtype: {dtype}")
    if quantize:
        print(f"Quantization: {quantize}")
    print(f"{'='*60}")
    
    # Prepare model kwargs
    model_kwargs = {}
    
    # Set dtype
    if dtype != "auto":
        torch_dtype = get_torch_dtype(dtype)
        model_kwargs["torch_dtype"] = torch_dtype
    
    # Set device
    if device:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
    
    # Handle quantization (requires bitsandbytes, CUDA only)
    if quantize and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            
            if quantize == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = bnb_config
                print("Using 4-bit quantization (requires ~1/4 memory)")
            elif quantize == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = bnb_config
                print("Using 8-bit quantization (requires ~1/2 memory)")
        except ImportError:
            print("[WARNING] bitsandbytes not installed. Skipping quantization.")
            print("Install with: pip install bitsandbytes")
    elif quantize and not torch.cuda.is_available():
        print("[WARNING] Quantization requires CUDA. Skipping on MPS/CPU.")
    
    # Load using SentenceTransformer with model_kwargs
    try:
        model = SentenceTransformer(
            hf_name,
            model_kwargs=model_kwargs,
            device=device,
            trust_remote_code=True,
        )
        print(f"Model loaded successfully on device: {model.device}")
        
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load with SentenceTransformer: {e}")
        print("Falling back to mteb.get_model()...")
        # Fallback to MTEB's loader
        return mteb.get_model(hf_name)


def evaluate_model(model, model_key: str, tasks: list, output_dir: Path):
    """
    Evaluate a single model

    Args:
        model: SentenceTransformer model
        model_key: Key from ALL_MODELS dict
        tasks: List of MTEB tasks
        output_dir: Output directory

    Returns:
        results: List of evaluation results
    """
    model_info = ALL_MODELS[model_key]
    model_output_dir = output_dir / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_info['name']}")
    print(f"Tasks: {tasks}")
    print(f"Output: {model_output_dir}")
    print(f"{'='*60}")
    
    # Get MTEB tasks
    mteb_tasks = mteb.get_tasks(tasks=tasks)
    
    # Run evaluation (MTEB 2.x API - no output_folder argument)
    results = mteb.evaluate(model, tasks=mteb_tasks)
    
    # Save raw results to JSON
    results_data = []
    for result in results:
        results_data.append({
            "task_name": result.task_name,
            "scores": result.scores,
        })
    
    results_file = model_output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"Raw results saved to: {results_file}")
    
    return results


def parse_results(results, model_key: str) -> list:
    """
    Parse evaluation results

    Args:
        results: List of evaluation results
        model_key: Key from ALL_MODELS dict

    Returns:
        parsed: List of parsed evaluation results
    """
    parsed = []
    model_info = ALL_MODELS[model_key]
    
    for result in results:
        task_name = result.task_name
        scores = result.scores
        
        # Extract main scores
        for split, split_scores in scores.items():
            for score_dict in split_scores:
                parsed.append({
                    "model_key": model_key,
                    "model_name": model_info["name"],
                    "task": task_name,
                    "split": split,
                    "ndcg_at_10": score_dict.get("ndcg_at_10", None),
                    "mrr_at_10": score_dict.get("mrr_at_10", None),
                    "recall_at_10": score_dict.get("recall_at_10", None),
                    "recall_at_100": score_dict.get("recall_at_100", None),
                    "map_at_10": score_dict.get("map_at_10", None),
                })
    
    return parsed


def run_evaluation(
    model_keys: list,
    tasks: list,
    output_dir: Path,
    dtype: str = "auto",
    quantize: str = None,
    device: str = None,
) -> pd.DataFrame:
    """
    Run evaluation for multiple models

    Args:
        model_keys: List of model keys
        tasks: List of MTEB tasks
        output_dir: Output directory
        dtype: Data type for model loading
        quantize: Quantization level
        device: Device to run on
    
    Returns:
        df: DataFrame of evaluation results
    """
    all_results = []
    
    for model_key in model_keys:
        try:
            # Load model
            model = load_model(model_key, dtype=dtype, quantize=quantize, device=device)
            
            # Run evaluation
            results = evaluate_model(model, model_key, tasks, output_dir)
            
            # Parse results
            parsed = parse_results(results, model_key)
            all_results.extend(parsed)
            
            # Release memory
            del model
            clear_gpu_memory()
            
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    return df


def save_results(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save results

    Args:
        df: DataFrame of evaluation results
        output_dir: Output directory
    """
    # Save as CSV
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save as JSON
    json_path = output_dir / "comparison_results.json"
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    print(f"Results saved to: {json_path}")
    
    # Display summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if "ndcg_at_10" in df.columns and df["ndcg_at_10"].notna().any():
        # Compare NDCG@10 by task
        summary = df.pivot_table(
            index="task",
            columns="model_name",
            values="ndcg_at_10",
            aggfunc="mean"
        )
        print("\nNDCG@10 by Task:")
        print(summary.to_string())
        
        # Average by model
        print("\nAverage NDCG@10 by Model:")
        avg_scores = df.groupby("model_name")["ndcg_at_10"].mean().sort_values(ascending=False)
        for model, score in avg_scores.items():
            print(f"  {model}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on MTEB Retrieval tasks"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_MODELS.keys()),
        help="Models to evaluate (default: kalm, qwen, nemotron). "
             "Smaller models available: minilm, bge-small, e5-small",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="MTEB tasks to evaluate (default: quick retrieval tasks)",
    )
    parser.add_argument(
        "--task-set",
        choices=["quick", "en", "ja", "all"],
        default="quick",
        help="Predefined task set: quick (2 tasks), en (4 English), ja (2 Japanese), all",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Data type for model loading. Use float16/bfloat16 to reduce memory usage",
    )
    parser.add_argument(
        "--quantize",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization level (CUDA only). Greatly reduces memory usage",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to run on",
    )
    
    args = parser.parse_args()
    
    # Determine tasks
    if args.tasks:
        tasks = args.tasks
    else:
        if args.task_set == "quick":
            tasks = RETRIEVAL_TASKS_QUICK
        elif args.task_set == "en":
            tasks = RETRIEVAL_TASKS_EN
        elif args.task_set == "ja":
            tasks = RETRIEVAL_TASKS_JA
        elif args.task_set == "all":
            tasks = RETRIEVAL_TASKS_EN + RETRIEVAL_TASKS_JA
        else:
            tasks = RETRIEVAL_TASKS_QUICK
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device info
    device_info = get_device_info()
    print("\n" + "="*60)
    print("MTEB Retrieval Evaluation")
    print("="*60)
    print(f"Device: {device_info['device']}")
    print(f"GPU: {device_info['gpu_name']}")
    print(f"Memory: {device_info['gpu_memory_gb']} GB")
    print(f"Dtype: {args.dtype}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")
    print(f"Models: {[ALL_MODELS[k]['name'] for k in args.models]}")
    print(f"Tasks: {tasks}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Memory warning for large models
    for model_key in args.models:
        model_info = ALL_MODELS[model_key]
        min_mem = model_info.get("min_memory_gb", 0)
        if min_mem > 8:
            print(f"\n[WARNING] {model_info['name']} requires ~{min_mem}GB memory")
            print("  Consider using --dtype float16 or --quantize 4bit")
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "device_info": device_info,
        "models": {k: ALL_MODELS[k] for k in args.models},
        "tasks": tasks,
        "dtype": args.dtype,
        "quantize": args.quantize,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Run evaluation
    df = run_evaluation(
        args.models, 
        tasks, 
        output_dir,
        dtype=args.dtype,
        quantize=args.quantize,
        device=args.device,
    )
    
    # Save results
    if not df.empty:
        save_results(df, output_dir)
    else:
        print("\n[WARNING] No results to save")


if __name__ == "__main__":
    main()
