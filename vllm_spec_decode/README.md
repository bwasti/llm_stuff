# Run collection

Set batch size to something reasonable or it will overwhelm the server.

```
python collect_spec_decode_data.py --model Qwen/Qwen3-1.7B --layers 27 --output-dir ./eagle_data --target-tokens 100000 --batch-size 8
```

# Run the trainer

Sanity check:

```
torchrun --nproc_per_node=8 train_spec_decode.py --data-dir ./eagle_data --filter-rank 0 --no-deduplicate --output-dir ./spec_decode_model --voc
ab-size 151936 --num-index-workers 8 --num-workers 8 --batch-size 64 --epochs 10 --max-samples 100000
```
