# Weight-space harness

A harness in which a frontier model (the teacher) reads the weights and the activations of a smaller
open model (the student) and edits the weights of the student directly. The method uses no gradient
and no optimizer. It follows the idea of "weight-space programming" (Takehi et al., blog post of
2026-11-02) and makes it usable on larger students and on new tasks.

The harness names no vendor. A teacher needs only a shell. A student is any open model that
Hugging Face `transformers` loads, or a ModelOpt NVFP4/FP8 checkpoint directory.

Status: research code. The results so far are in [RESULTS.md](RESULTS.md).

## Parts

| Path | Purpose |
|---|---|
| `PLAYBOOK.md` | The instructions for a teacher: rules, tools, and the order of work. |
| `wsp/kernel.py`, `wsp/client.py` | A process that holds the student in memory and runs teacher code. It records each code run and its output. |
| `wsp/tools.py` | Generate, read activations, read log-probabilities, write an edit, undo an edit, and measure the distance from the original model. Every call goes to a ledger. |
| `wsp/edits.py` | Closed-form edits: remove a direction, wire neurons to a direction, and a least-squares fit of keys to values. |
| `wsp/habit.py` | The recipe for a habit in the wrong context: keys, protection over all positions, the fit, and a check for stray writes. |
| `wsp/attribution.py` | The share of each layer part in the logit of one token. |
| `wsp/layout.py` | Finds the layers, MLPs, and attention blocks of a student by name. |
| `wsp/evaluate.py`, `wsp/stats.py` | Benchmark runs with a fixed seed, intervals, and paired tests. |
| `wsp/battery.py` | The generalization battery: fixed subsets of general benchmarks that run before and after an edit. |
| `wsp/bench/` | Benchmarks: ScreenSpot Pro, boxed math (GSM8K), MMLU-Pro, CountBench, ChartQA, RefCOCO, Meta-World through text waypoints, and MiniWoB++ through screenshots. |
| `wsp/modelopt.py` | Loads a ModelOpt mixed-precision checkpoint. The weights stay in their 4-bit and 8-bit formats, and an edit goes back into the same format. A 27B checkpoint runs on one 32 GB GPU at about 3 tokens per second. |
| `wsp/lora_baseline.py` | A LoRA fine-tune with a fixed wall-clock budget. It is the comparison, not the method. |
| `runs/` | One directory per run: the ledger, the teacher code, the outputs, and the edit manifest. |
| `.scratch/weight-space-harness/` | The map of the effort and one file per decision. |

## Start

```bash
uv sync
uv run python -m wsp.kernel --model Qwen/Qwen3.5-4B --run runs/my-run   # prints "kernel ready"
echo "print(tools.generate([[{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]}]]))" | uv run python -m wsp.client -
kill $(cat /tmp/wsp-kernel.pid)
```

Then read `PLAYBOOK.md`.

## What every result reports

1. The gain on held-out items of the target.
2. The wall-clock time and the number of forward passes.
3. Generalization: the battery before and after, and a near-transfer set next to the target.
4. Edit size: the share of weights, tensors, and layers that the edit changes.
5. Deviation from the original: the distance in weight space, and the distance in output space on
   neutral text.
6. A LoRA fine-tune with the same or less wall-clock time, measured with the same tools.

A negative result is a valid result, and the harness records it in the same form.

## Limits

- The kernel runs code that the teacher writes. Run it only on a machine where that is acceptable.
- The benchmark loaders download public datasets from the Hugging Face Hub at first use.
- The Meta-World interface (text waypoints with positions from the simulator state) is specific to
  this harness. No published score exists for it.
- The MiniWoB++ interface (one screenshot per turn, JSON actions, no time limit) is also specific to
  this harness. The benchmark needs Chrome, and Selenium downloads the matching ChromeDriver.
