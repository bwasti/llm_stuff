#!/usr/bin/env python3
"""
Script to collect spec decode training data by spamming requests through vLLM.

This script:
1. Downloads a dataset from HuggingFace
2. Starts spec decode data collection
3. Sends many inference requests to collect training data
4. Monitors progress and stops when target is reached

Prerequisites:
    export VLLM_SERVER_DEV_MODE=1
    vllm serve <model_name> --port 8000

Example:
    python collect_spec_decode_data.py \\
        --model meta-llama/Llama-2-7b-hf \\
        --layers 0,8,16,24,31 \\
        --output-dir ./eagle_data \\
        --target-tokens 1000000 \\
        --concurrent-requests 10
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from tqdm import tqdm


def start_collection(
    base_url: str,
    output_dir: str,
    layers: list[int] | None = None,
    samples_per_file: int = 1000,
    max_buffer_size: int = 100,
    collect_logits: bool = True,
    collect_hidden_states: bool = True,
) -> dict[str, Any]:
    """Start spec decode data collection."""
    url = f"{base_url}/spec_decode/collection/start"

    params = {
        "output_dir": output_dir,
        "samples_per_file": samples_per_file,
        "max_buffer_size": max_buffer_size,
        "collect_logits": str(collect_logits).lower(),
        "collect_hidden_states": str(collect_hidden_states).lower(),
    }

    if layers:
        params["hidden_state_layers"] = ",".join(map(str, layers))

    response = requests.post(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def stop_collection(base_url: str) -> dict[str, Any]:
    """Stop spec decode data collection."""
    url = f"{base_url}/spec_decode/collection/stop"
    response = requests.post(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_collection_status(base_url: str) -> dict[str, Any]:
    """Get current collection status."""
    url = f"{base_url}/spec_decode/collection/status"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def send_inference_request(
    base_url: str,
    model: str,
    prompt: str | list[str],
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Send an inference request with single or multiple prompts."""
    url = f"{base_url}/v1/completions"
    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def load_hf_dataset(dataset_name: str, split: str = "train", max_samples: int | None = None):
    """Load a dataset from HuggingFace."""
    print(f"Loading dataset: {dataset_name}")

    # Map of common datasets and their text fields
    text_field_map = {
        "openai/gsm8k": "question",
        "Open-Orca/OpenOrca": "question",
        "timdettmers/openassistant-guanaco": "text",
        "yahma/alpaca-cleaned": "instruction",
        "tatsu-lab/alpaca": "instruction",
        "HuggingFaceH4/ultrachat_200k": "prompt",
        "nvidia/HelpSteer": "prompt",
    }

    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)

        # Determine text field
        text_field = text_field_map.get(dataset_name)
        if text_field is None:
            # Try to auto-detect
            first_item = next(iter(dataset))
            possible_fields = ["text", "prompt", "question", "instruction", "input"]
            for field in possible_fields:
                if field in first_item:
                    text_field = field
                    break

            if text_field is None:
                raise ValueError(
                    f"Could not auto-detect text field. Available fields: {list(first_item.keys())}"
                )

        print(f"Using text field: '{text_field}'")

        # Extract prompts
        prompts = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            text = item.get(text_field, "")
            if isinstance(text, str) and len(text.strip()) > 10:
                prompts.append(text.strip())

            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i+1} samples...")

        print(f"Loaded {len(prompts)} prompts from dataset")
        return prompts

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to default prompts...")
        return get_default_prompts() * 100  # Repeat to have more variety


def get_default_prompts() -> list[str]:
    """Get default prompts if dataset loading fails."""
    return [
        "Once upon a time in a land far away, there lived a",
        "The quick brown fox jumps over the lazy dog. This sentence",
        "In a galaxy far, far away, a young hero discovers",
        "To be or not to be, that is the question. Whether",
        "It was the best of times, it was the worst of times",
        "Call me Ishmael. Some years ago, never mind how long precisely",
        "All happy families are alike; each unhappy family is unhappy",
        "It is a truth universally acknowledged, that a single man in",
        "In the beginning God created the heaven and the earth",
        "It was a bright cold day in April, and the clocks",
        "Write a Python function that calculates the fibonacci sequence",
        "Explain quantum computing to a 5 year old",
        "What are the key differences between machine learning and deep learning?",
        "Describe the process of photosynthesis in plants",
        "How do neural networks learn from data?",
        "What is the theory of relativity and why is it important?",
        "Explain the difference between supervised and unsupervised learning",
        "What are the main challenges in natural language processing?",
        "Describe how transformers work in modern AI models",
        "What is the difference between AI, ML, and Deep Learning?",
    ]


async def send_request_async(
    base_url: str,
    model: str,
    prompt: str | list[str],
    max_tokens: int,
    temperature: float,
    session: Any,
    retry_delay: float = 1.0,
    max_retries: int = 3,
) -> tuple[int, bool]:
    """Send a single async request with one or more prompts and return token count."""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        try:
            import aiohttp
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                response.raise_for_status()
                result = await response.json()
                # Count tokens from all completions
                total_tokens = 0
                if "choices" in result:
                    for choice in result["choices"]:
                        completion = choice.get("text", "")
                        # Rough token estimate (1 token ≈ 4 chars)
                        total_tokens += len(completion) // 4
                    return total_tokens, True
                return 0, True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                await asyncio.sleep(wait_time)
            else:
                print(f"\nRequest failed after {max_retries} attempts: {e}")
                return 0, False

    return 0, False


async def spam_requests_async(
    base_url: str,
    model: str,
    prompts: list[str],
    target_tokens: int,
    max_tokens_per_request: int,
    temperature: float,
    concurrent_requests: int,
    batch_size: int = 1,
    delay_between_requests: float = 0.0,
    max_requests_per_second: int | None = None,
):
    """Spam requests asynchronously until target tokens are collected."""
    import aiohttp

    total_tokens = 0
    total_requests = 0
    successful_requests = 0
    failed_requests = 0

    prompt_idx = 0

    pbar = tqdm(total=target_tokens, desc="Collecting tokens", unit="tokens")

    # Rate limiting
    request_times = []

    async with aiohttp.ClientSession() as session:
        pending_tasks = set()

        while total_tokens < target_tokens:
            # Rate limiting check
            if max_requests_per_second:
                now = time.time()
                # Remove requests older than 1 second
                request_times[:] = [t for t in request_times if now - t < 1.0]

                # If we've hit the rate limit, wait
                if len(request_times) >= max_requests_per_second:
                    sleep_time = 1.0 - (now - request_times[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                        continue

            # Fill up to concurrent_requests
            while len(pending_tasks) < concurrent_requests and total_tokens < target_tokens:
                # Collect batch_size prompts
                if batch_size == 1:
                    prompt = prompts[prompt_idx % len(prompts)]
                    prompt_idx += 1
                else:
                    batch_prompts = []
                    for _ in range(batch_size):
                        batch_prompts.append(prompts[prompt_idx % len(prompts)])
                        prompt_idx += 1
                    prompt = batch_prompts

                if delay_between_requests > 0:
                    await asyncio.sleep(delay_between_requests)

                task = asyncio.create_task(
                    send_request_async(
                        base_url, model, prompt, max_tokens_per_request, temperature, session
                    )
                )
                pending_tasks.add(task)

                if max_requests_per_second:
                    request_times.append(time.time())

            # Wait for at least one to complete
            if pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    tokens, success = await task
                    total_requests += 1
                    if success:
                        successful_requests += 1
                        total_tokens += tokens
                        pbar.update(tokens)
                    else:
                        failed_requests += 1
                        # Slow down if we're getting failures
                        if failed_requests > 5:
                            await asyncio.sleep(2.0)

    pbar.close()

    return total_tokens, total_requests, successful_requests


def spam_requests_sync(
    base_url: str,
    model: str,
    prompts: list[str],
    target_tokens: int,
    max_tokens_per_request: int,
    temperature: float,
    batch_size: int = 1,
    delay_between_requests: float = 0.0,
):
    """Spam requests synchronously (fallback if async fails)."""
    total_tokens = 0
    total_requests = 0
    successful_requests = 0
    failed_requests = 0

    prompt_idx = 0

    pbar = tqdm(total=target_tokens, desc="Collecting tokens", unit="tokens")

    while total_tokens < target_tokens:
        # Collect batch_size prompts
        if batch_size == 1:
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
        else:
            batch_prompts = []
            for _ in range(batch_size):
                batch_prompts.append(prompts[prompt_idx % len(prompts)])
                prompt_idx += 1
            prompt = batch_prompts

        total_requests += 1

        try:
            result = send_inference_request(
                base_url, model, prompt, max_tokens_per_request, temperature
            )

            if "choices" in result:
                # Count tokens from all completions
                batch_tokens = 0
                for choice in result["choices"]:
                    completion = choice.get("text", "")
                    # Rough token estimate
                    batch_tokens += len(completion) // 4

                total_tokens += batch_tokens
                successful_requests += 1
                pbar.update(batch_tokens)
                failed_requests = 0  # Reset failure counter

        except Exception as e:
            failed_requests += 1
            print(f"\nRequest {total_requests} failed: {e}")
            # Slow down if we're getting repeated failures
            if failed_requests > 3:
                time.sleep(2.0)
            else:
                time.sleep(0.5)

        # Delay between requests
        if delay_between_requests > 0 and total_tokens < target_tokens:
            time.sleep(delay_between_requests)

    pbar.close()

    return total_tokens, total_requests, successful_requests


def check_server_connection(base_url: str) -> bool:
    """Check if the vLLM server is reachable."""
    try:
        # Try to get server health/info
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Collect spec decode data by spamming vLLM endpoint"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the vLLM server",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eagle_data",
        help="Directory to save collected data",
    )
    parser.add_argument(
        "--layers",
        type=str,
        help="Comma-separated list of layer indices (e.g., '0,8,16,24,31')",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=100000,
        help="Target number of tokens to collect (default: 100k)",
    )
    parser.add_argument(
        "--max-tokens-per-request",
        type=int,
        default=100,
        help="Maximum tokens per request (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts per request (default: 1)",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1, increase carefully)",
    )
    parser.add_argument(
        "--delay-between-requests",
        type=float,
        default=0.0,
        help="Delay in seconds between requests (default: 0.0)",
    )
    parser.add_argument(
        "--max-requests-per-second",
        type=int,
        default=None,
        help="Maximum requests per second (default: unlimited)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Open-Orca/OpenOrca",
        help="HuggingFace dataset to use (default: Open-Orca/OpenOrca)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--max-dataset-samples",
        type=int,
        default=10000,
        help="Maximum samples to load from dataset (default: 10000)",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=1000,
        help="Samples per output file (default: 1000)",
    )
    parser.add_argument(
        "--no-async",
        action="store_true",
        help="Disable async requests (use sync instead)",
    )
    parser.add_argument(
        "--no-collect-logits",
        action="store_true",
        help="Don't collect logits",
    )
    parser.add_argument(
        "--no-collect-hidden-states",
        action="store_true",
        help="Don't collect hidden states",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Check server connection first
    print("="*60)
    print("Checking vLLM server connection")
    print("="*60)
    print(f"  Server URL: {args.base_url}")

    if not check_server_connection(args.base_url):
        print(f"\n✗ Cannot connect to vLLM server at {args.base_url}")
        print("\nMake sure the server is running:")
        print(f"  export VLLM_SERVER_DEV_MODE=1")
        print(f"  vllm serve {args.model} --port 8000")
        print("\nOr specify a different URL with --base-url")
        return 1

    print(f"✓ Server is reachable")

    # Parse layers
    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    # Load dataset
    print("="*60)
    print("Step 1: Loading dataset")
    print("="*60)
    prompts = load_hf_dataset(
        args.dataset,
        split=args.dataset_split,
        max_samples=args.max_dataset_samples
    )

    if not prompts:
        print("Error: No prompts loaded!")
        return 1

    # Start collection
    print("\n" + "="*60)
    print("Step 2: Starting spec decode data collection")
    print("="*60)
    print(f"  Output directory: {args.output_dir}")
    if layers:
        print(f"  Layers to collect: {layers}")
    else:
        print(f"  Collecting all layers")
    print(f"  Target tokens: {args.target_tokens:,}")

    try:
        result = start_collection(
            base_url=args.base_url,
            output_dir=args.output_dir,
            layers=layers,
            samples_per_file=args.samples_per_file,
            collect_logits=not args.no_collect_logits,
            collect_hidden_states=not args.no_collect_hidden_states,
        )
        print(f"✓ Collection started successfully")
        print(f"  Config: {json.dumps(result.get('config', {}), indent=2)}")
    except Exception as e:
        print(f"✗ Failed to start collection: {e}")
        return 1

    # Spam requests
    print("\n" + "="*60)
    print("Step 3: Spamming inference requests")
    print("="*60)
    print(f"  Batch size: {args.batch_size} prompts per request")
    print(f"  Concurrent requests: {args.concurrent_requests}")
    print(f"  Max tokens per request: {args.max_tokens_per_request}")
    print(f"  Temperature: {args.temperature}")
    if args.delay_between_requests > 0:
        print(f"  Delay between requests: {args.delay_between_requests}s")
    if args.max_requests_per_second:
        print(f"  Max requests per second: {args.max_requests_per_second}")
    print()

    start_time = time.time()

    try:
        if args.no_async:
            total_tokens, total_requests, successful_requests = spam_requests_sync(
                base_url=args.base_url,
                model=args.model,
                prompts=prompts,
                target_tokens=args.target_tokens,
                max_tokens_per_request=args.max_tokens_per_request,
                temperature=args.temperature,
                batch_size=args.batch_size,
                delay_between_requests=args.delay_between_requests,
            )
        else:
            # Use async
            try:
                import aiohttp
                total_tokens, total_requests, successful_requests = asyncio.run(
                    spam_requests_async(
                        base_url=args.base_url,
                        model=args.model,
                        prompts=prompts,
                        target_tokens=args.target_tokens,
                        max_tokens_per_request=args.max_tokens_per_request,
                        temperature=args.temperature,
                        concurrent_requests=args.concurrent_requests,
                        batch_size=args.batch_size,
                        delay_between_requests=args.delay_between_requests,
                        max_requests_per_second=args.max_requests_per_second,
                    )
                )
            except ImportError:
                print("aiohttp not installed, falling back to sync requests")
                total_tokens, total_requests, successful_requests = spam_requests_sync(
                    base_url=args.base_url,
                    model=args.model,
                    prompts=prompts,
                    target_tokens=args.target_tokens,
                    max_tokens_per_request=args.max_tokens_per_request,
                    temperature=args.temperature,
                    batch_size=args.batch_size,
                    delay_between_requests=args.delay_between_requests,
                )

        elapsed_time = time.time() - start_time

        print(f"\n✓ Request spamming completed!")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {total_requests - successful_requests}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Time elapsed: {elapsed_time:.1f}s")
        print(f"  Tokens/sec: {total_tokens/elapsed_time:.1f}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    except Exception as e:
        print(f"\n✗ Error during request spamming: {e}")
        import traceback
        traceback.print_exc()

    # Check final status
    print("\n" + "="*60)
    print("Step 4: Checking collection status")
    print("="*60)
    try:
        status = get_collection_status(args.base_url)
        print(f"  Samples collected: {status.get('total_samples_collected', 'N/A')}")
        print(f"  Files written: {status.get('files_written', 'N/A')}")
        print(f"  Buffer size: {status.get('buffer_size', 'N/A')}")
    except Exception as e:
        print(f"  Could not get status: {e}")

    # Stop collection
    print("\n" + "="*60)
    print("Step 5: Stopping data collection")
    print("="*60)
    try:
        result = stop_collection(args.base_url)
        print(f"✓ Collection stopped successfully")
        if "final_stats" in result:
            print(f"  Final stats: {json.dumps(result['final_stats'], indent=2)}")
    except Exception as e:
        print(f"✗ Failed to stop collection: {e}")

    print("\n" + "="*60)
    print("✓ Data collection workflow completed!")
    print("="*60)
    print(f"\nCollected data saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Analyze the data: python {__file__} --analyze {args.output_dir}")
    print(f"  2. Train an EAGLE-style draft model with the collected data")
    print(f"  3. Deploy the draft model with vLLM's speculative decoding")

    return 0


if __name__ == "__main__":
    exit(main())
