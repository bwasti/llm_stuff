  # 1. Collect data
  python collect_spec_decode_data.py \
      --model Qwen/Qwen3-1.7B \
      --layers 0,13,27 \
      --output-dir ./eagle_data \
      --target-tokens 1000000 \
      --batch-size 8

  # 2. Train EAGLE3 model
  torchrun --nproc_per_node=8 train_eagle3_qwen3.py \
      --data-dir ./eagle_data \
      --filter-rank 0 \
      --no-deduplicate \
      --output-dir ./eagle3_qwen3_model \
      --base-model Qwen/Qwen3-1.7B \
      --target-layers 0,13,27 \
      --num-draft-layers 1 \
      --epochs 10 \
      --batch-size 64 \
      --gradient-accumulation-steps 4

  # 3. Use with vLLM
  python -c "
  from vllm import LLM
  llm = LLM(
      model='Qwen/Qwen3-1.7B',
      speculative_config={
          'model': './eagle3_qwen3_model/best_model',
          'method': 'eagle3',
          'num_speculative_tokens': 2,
      }
  )
  outputs = llm.generate(['Hello world'])
  print(outputs[0].outputs[0].text)
  "



# EAGLE3 Training for Qwen3 Speculative Decoding

Train an EAGLE3 draft model for Qwen3-1.7B to accelerate inference with speculative decoding.

## Prerequisites

```bash
pip install torch numpy tqdm transformers datasets aiohttp
```

## Step 1: Collect Training Data

Start your vLLM server with the V1 engine in development mode:

```bash
export VLLM_USE_V1=1
export VLLM_SERVER_DEV_MODE=1
vllm serve Qwen/Qwen3-1.7B --port 8000 --tensor-parallel-size 2
```

**Important:** Keep batch size low (8 or less) to avoid overwhelming the server. The V1 engine collects data during inference.

Collect hidden states from strategic layers (typically: first, middle, last):
- Qwen3-1.7B has 28 layers (0-27), so use layers: 0, 13, 27

```bash
python collect_spec_decode_data.py \
    --model Qwen/Qwen3-1.7B \
    --layers 0,13,27 \
    --output-dir ./eagle_data \
    --target-tokens 1000000 \
    --batch-size 8 \
    --concurrent-requests 4
```

This will collect:
- Hidden states from layers 0, 13, and 27
- Output logits from the target model
- Token IDs

## Step 2: Train the EAGLE3 Model

Train using the collected data (recommended to use only one rank's data to avoid duplicates):

```bash
# Single GPU training
python train_eagle3_qwen3.py \
    --data-dir ./eagle_data \
    --filter-rank 0 \
    --no-deduplicate \
    --output-dir ./eagle3_qwen3_model \
    --base-model Qwen/Qwen3-1.7B \
    --target-layers 0,13,27 \
    --num-draft-layers 1 \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-4

# Multi-GPU training with DDP
torchrun --nproc_per_node=8 train_eagle3_qwen3.py \
    --data-dir ./eagle_data \
    --filter-rank 0 \
    --no-deduplicate \
    --output-dir ./eagle3_qwen3_model \
    --base-model Qwen/Qwen3-1.7B \
    --target-layers 0,13,27 \
    --num-draft-layers 1 \
    --epochs 10 \
    --batch-size 64 \
    --gradient-accumulation-steps 4 \
    --lr 1e-4
```

### Key Training Parameters

- `--target-layers`: Must match the layers used during data collection
- `--num-draft-layers`: Number of transformer layers in the draft model (1-2 recommended)
- `--batch-size`: Training batch size per GPU
- `--gradient-accumulation-steps`: Accumulate gradients to simulate larger batch sizes
- `--filter-rank 0`: Use only data from rank 0 to avoid duplicates (data is identical across all ranks)
- `--no-deduplicate`: Skip deduplication when using single rank data

## Step 3: Use with vLLM

Once trained, use the EAGLE3 model with vLLM for speculative decoding:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    speculative_config={
        "model": "./eagle3_qwen3_model/best_model",
        "method": "eagle3",  # Important: must be "eagle3"!
        "num_speculative_tokens": 2,
        "draft_tensor_parallel_size": 1,
    },
)

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

## Architecture Notes

The EAGLE3 model uses:
- **Fusion layer**: Combines 3 hidden states from target model
- **Transformer layers**: 1-2 decoder layers with special handling for layer 0
- **Layer 0**: Takes 2x hidden_size input (concatenated embeddings + fused hidden states)
- **Subsequent layers**: Normal hidden_size inputs
- **RMSNorm**: For better training stability
- **LM head**: Projects to vocabulary for token generation

This architecture allows the draft model to learn from intermediate representations of the target model at multiple depths, enabling better speculation.

## Troubleshooting

### Data Collection Issues
- **Server hangs**: Reduce `--batch-size` and `--concurrent-requests`
- **Out of memory**: Lower batch size or reduce TP size
- **No data collected**: Check that VLLM_SERVER_DEV_MODE=1 is set

### Training Issues
- **Out of memory**: Reduce `--batch-size` or increase `--gradient-accumulation-steps`
- **Loss not decreasing**: Try lowering learning rate or increasing model capacity
- **Slow training**: Use more GPUs with `torchrun` or increase `--num-workers`

### Inference Issues
- **Model not loading**: Make sure architecture in config.json matches Eagle3Qwen3ForCausalLM
- **Poor speedup**: Try increasing `--num-speculative-tokens` (2-5 typical)
- **Method error**: Must use `"method": "eagle3"` in speculative_config
