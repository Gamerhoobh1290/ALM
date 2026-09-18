# AdamLM first language experiment — 2026-09-16

Completed 2,097,152 UTF-8 byte prediction tokens, 1,024 AdamW updates, 2,400 fetched training stories in 24 bounded API requests. Architecture is unchanged: 4 layers, width 384, 6 heads, learned absolute positions, tied byte embeddings, 256 context, 7,295,232 parameters. Fresh seed 1337; original three baseline checkpoints are preserved. BF16, batch 8, accumulation 1, learning rate 0.0003, gradient norm clipped at 1.0. No long-term training or Windows automation was started.

## Measurements

GPU: RTX 3060 12 GB; PyTorch 2.11.0+cu128. Each compute benchmark warmed up for 10 steps and measured at least 15 seconds, synchronizing CUDA before timing and after every update. Fixed synthetic batches measure compute only; they do not measure language quality. AdamW and clipping are included. TF32/compile/fused optimizer tuning was not introduced.

| Configuration | Tokens/sec | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|
| 0.46M FP32, batch 16, context 256 | 569,558 | 107.5 | 132 |
| 7.30M FP32, batch 8, context 256 | 87,178 | 322.0 | 380 |
| 19.18M FP32, batch 4, context 256 | 32,760 | 458.3 | 492 |
| 7.30M BF16, batch 8, context 256 | 166,269 | 254.0 | 286 |
| 7.30M FP16, batch 8, context 256 | 158,873 | 254.0 | 286 |
| 7.30M BF16, microbatch 2 × accumulation 4 | 68,752 | 178.8 | 196 |
| 7.59M BF16, batch 2, context 1024 | 142,505 | 266.6 | 310 |

BF16 is 1.91× FP32 throughput and reduces allocated memory by 21%. Accumulation reduces memory further but is substantially slower at the same effective batch size; the selected model does not need it. The 1024-context case has the same 2,048 tokens/update and ~14% lower throughput than BF16/256. This is a feasibility test, not evidence of better validation quality.

The initial training segment exposed an expensive full-directory scan before every update: 262,144 tokens took 207.17 seconds (1,265 tokens/sec). After its atomic step-128 checkpoint, the process was stopped and resumed. Full budget/cache scans now run every 30 seconds and before checkpoint writes, with free-disk checks each step and checkpoint-space reservation. The remaining 1,835,008 tokens took 38.92 seconds, **47,153 tokens/sec end to end**, including streamed data, startup/resume, validation, disk checks, checkpoints, and final samples.

Recorded cumulative execution time is 245.95 seconds, or 8,527 tokens/sec including the original bottleneck. This excludes the brief manual stop/relaunch gap and discarded uncheckpointed work. The entire experiment was comfortably within one hour. Recorded data preparation took 11.85 seconds and synchronized optimizer updates 12.09 seconds. The optimized end-to-end rate, not 166k synthetic compute throughput, is the useful planning baseline. At an unchanged workload, 100M byte tokens would extrapolate to roughly 35 minutes; network/API conditions and longer-run behavior make this an estimate, not a promise.

The experiment peaked at 251.3 MiB allocated / 286 MiB reserved by PyTorch. Eighteen 1-second telemetry samples overlapping the resumed run averaged 40.3% GPU utilization, peaked at 90%, and observed up to 1,159 MiB total device memory, 50°C, and 104.85 W. These are sampled, whole-device measurements, not process-isolated or full-run averages. The raw 60-s telemetry file also includes idle samples after completion; those are excluded from these figures. Checkpoint-boundary utilization readings alone were misleadingly low.

## Unseen-text evaluation

Validation comes from the actual TinyStories validation split, never the training cursor. A fixed 64-story snapshot is saved; first 32 stories provide 20,480 packed evaluation tokens for tracking and next 32 provide 20,480 final holdout tokens. Exact story hashes are excluded from training. Partial trailing batches are dropped. The snapshot's SHA-256 is stored in checkpoints and checked on resume.

| Training tokens | Mean training loss over preceding 128 updates | Validation loss |
|---:|---:|---:|
| 0 | — | 5.5273 |
| 262,144 | 2.6289 | 2.3625 |
| 524,288 | 2.2827 | 2.2771 |
| 786,432 | 2.1968 | 2.1627 |
| 1,048,576 | 2.0398 | 1.9864 |
| 1,310,720 | 1.8210 | 1.6717 |
| 1,572,864 | 1.7263 | 1.6473 |
| 1,835,008 | 1.6219 | 1.5567 |
| 2,097,152 | 1.5006 | 1.4951 |

Final unseen holdout loss: **1.5080 nats/byte**, byte perplexity **4.518**. Last individual training batch: 1.3817; the 1.5006 window mean is more representative. Training and validation both improve; this small evaluation gives no sign of a large overfitting gap, but is too small for broad quality claims. No repeated seeds, confidence intervals, near-duplicate audit, or out-of-domain evaluation were performed. The mutable viewer API is not revision-pinned; the saved cursor guarantees current row/buffer continuity, not immutable future remote content.

Samples were generated after reloading step 1024, temperature 0.8, top-k 40, seed 42, 300 new bytes each. These are literal excerpts, including errors:

> Once upon a time, a little girl day, Lily. They said, "Cant's made are bing for inno, but the dad and to of the betack. Anden bates

> The dog found a red ball. He dad some it he be back to to brock and bing for innifelt brack and to the frow ham

> Lily was sad because dand and soll her to it with brocks anout a fistinngs and to butthe stad of the betack.

Full samples are in `first-run/summary.json`. They show partial spelling, vocabulary, and story-format learning; grammar, coherence, and repetition are still poor. This is an undertrained baseline, not a usable story model. Standalone checkpoint generation reproduced the first excerpt.

## Tokenizer and next-run recommendation

The held-out stories average 5.005 UTF-8 bytes per whitespace-delimited word and 687.95 bytes/story. Thus a 256-byte context covers roughly **51 words**, only 37% of the average story length. Two million byte tokens are not two million BPE tokens. Byte perplexity and BPE perplexity are not directly comparable.

Before serious training, train a **custom 4,096-token byte-level BPE tokenizer on training text only**, freeze and hash it, and use **512-token context**. Measure its actual compression and held-out coverage before launch; no BPE tokenizer has been trained or its quality claimed here. Preserve byte fallback and add an explicit story boundary token. Keep 4 layers / width 384 / 6 heads initially (~8.87M parameters at exactly 4096 vocabulary and 512 positions), BF16, microbatch 8, accumulation 1, gradient clipping 1.0. Benchmark this exact BPE configuration before setting a longer token budget. Introduce a short LR warmup and decay for the longer run; this pilot used constant 3e-4. There is no measured reason yet to jump to 19M parameters.

Changing vocabulary changes token IDs and embedding/output shapes. Existing byte checkpoints cannot be resumed under BPE; use a **fresh model and optimizer lineage**, retain these checkpoints for comparison, and compare models on identical held-out text using bits/byte or total text log-likelihood. Increasing context alone changes the learned position-embedding shape: it also fails a strict resume. Position-extension surgery plus optimizer-state handling would be a separate warm-start procedure, not an exact continuation. BF16 versus FP32 autocast does not change weight shapes; the experiment nonetheless locks precision on resume to keep its protocol consistent.

For a longer run, expand and freeze validation, establish immutable source identity or bounded local shards, and consider a bounded prefetch buffer only if end-to-end profiling still identifies data waits. These are recommendations, not new infrastructure added this session.

## Reliability and storage

Ten tests pass: existing baseline tests plus causal attention, accumulation versus full-batch equivalence, exact next-update checkpoint replay with populated AdamW state and stream cursor, source-mismatch rejection, checkpoint-space reservation, and generation RNG/mode preservation. A real GPU resume reproduced validation loss 2.362509 at step 128 before continuing. New checkpoints include model config, optimizer/RNG, dataset buffer/cursor, precision/scaler state where applicable, protocol, and timing counters. Only trusted local checkpoints should be loaded.

Project usage is **4.729 GiB** including the environment and both checkpoint lineages. New retained checkpoints total about 251 MiB; validation snapshot/logs/samples are small. The full corpus is not downloaded. Budget remains 15 GiB, cache limit 2 GiB, drive safety floor 5 GiB. Current free C: space was 138.9 GiB. The prior user-level pip cache remains outside the project and was untouched. PyTorch still emits the pre-existing missing-NumPy warning; these tensor-only workflows pass without NumPy.

Artifacts: `benchmark.json`, `first-run/metrics.jsonl`, `first-run/summary.json`, `first-run/validation.json`, telemetry JSON files, and `first-run/checkpoints/step_00001024.pt`.
