# Cartridges vs STILL Smoke Comparison

This comparison is not meant to be fair on quality yet. The current STILL run is a 2-step smoke verification intended to validate the artifact flow and benchmark contract.

## Reference Cartridges Result

Source: `/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/cartridge_1024/report/summary.json`

- Exact match: `0.55`
- Compression ratio: `8.07x`
- Average prefill speedup: `8.19x`
- Average end-to-end speedup: `1.73x`
- Build time: `125.35s`

## Local STILL Smoke Result

Source: `outputs/wikipedia_india/runs/india_smoke/still_128/report/summary.json`

- Exact match: `0.00`
- Compression ratio: `64.57x`
- Average prefill speedup: `7.92x`
- Average end-to-end speedup: `1.12x`
- Build time: `58.61s`

## Takeaway

The current STILL smoke run already demonstrates the intended systems tradeoff:

- much smaller compact cache than the reference cartridge run
- similar prefill reduction despite far more aggressive compression
- lower one-time build cost than the reference cartridge run

Quality is not competitive yet because the local STILL checkpoint was only trained for 2 steps on a tiny bootstrap dataset. The implementation is now in place for the next loop: increase training data, raise train steps, and then evaluate the quality-speed tradeoff properly.
