"""The comparison baseline: a LoRA fine-tune with a fixed wall-clock budget.

This script uses gradient descent. It is not part of the teacher method. It exists so that a
teacher edit and ordinary training can be compared on the same items with the same time.

The script trains on the examples of one benchmark, then writes the merged weights of every
adapted matrix to one safetensors file. The kernel loads that file with `tools.load_edit`, so the
harness measures the LoRA model with the same tools as a teacher edit.

Use:
    uv run python -m wsp.lora_baseline --model Qwen/Qwen3.5-4B --bench screenspot_pro \\
        --split runs/<run>/split.json --key practice --minutes 15 --out runs/<run>/lora
    uv run python -m wsp.lora_baseline --model Qwen/Qwen3.5-4B --bench metaworld \\
        --seeds 0:20 --minutes 15 --out runs/<run>/lora
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor

from wsp import layout

# The linear layers inside each attention block and MLP of the Qwen3.5 language model.
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj")


END = "<|im_end|>"


def screenspot_examples(args) -> list:
    """One example per ScreenSpot Pro item of the split: the reply is a click on the centre of the target."""
    from wsp.bench import screenspot_pro as sp

    if not args.split:
        raise SystemExit("The screenspot_pro benchmark needs --split. Nothing was trained.")
    items, images = sp.load()
    wanted = set(json.loads(Path(args.split).read_text())[args.key])

    def reply_text(item: dict) -> str:
        width, height = item["img_size"]
        x1, y1, x2, y2 = item["bbox"]
        x, y = round(500 * (x1 + x2) / width), round(500 * (y1 + y2) / height)
        call = {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}
        return f"<tool_call>\n{json.dumps(call)}\n</tool_call>{END}"

    # The screenshots are large, so each conversation is built when the loop needs it.
    return [lambda item=item: (sp.conversation(item, images), reply_text(item)) for item in items if item["id"] in wanted]


def metaworld_examples(args) -> list:
    """One example per step of each reference plan that succeeds from a seed in the range."""
    from wsp.bench import metaworld_text as mw

    first, last = (int(part) for part in args.seeds.split(":"))
    examples = []
    tasks = args.tasks.split(",") if args.tasks else list(mw.INSTRUCTIONS)
    unknown = [task for task in tasks if task not in mw.INSTRUCTIONS]
    if unknown:
        raise SystemExit(f"Unknown Meta-World tasks: {unknown}. Nothing was trained.")
    for task in tasks:
        for seed in range(first, last):
            episode = mw.Episode(task, seed)
            steps = []
            for primitive in mw.reference_plan(task, episode.positions()) or []:
                if episode.over:
                    break
                steps.append((mw.conversation(episode), json.dumps(primitive) + END))
                episode.apply(primitive)
            if episode.success:
                examples += [lambda step=step: step for step in steps]
            episode.close()
    return examples


def miniwob_examples(args) -> list:
    """One example per step of each recorded MiniWoB episode that succeeded."""
    from wsp.bench import miniwob_screen as mb

    if not args.episodes:
        raise SystemExit("The miniwob bench needs --episodes, a JSON file of recorded episodes. Nothing was trained.")
    examples = []
    for record in json.loads(Path(args.episodes).read_text()):
        if not record["correct"]:
            continue
        episode = mb.Episode(record["task"], record["seed"])
        steps = []
        for action in record["actions"]:
            if episode.over or action["action"] == "invalid":
                break
            steps.append((mb.conversation(episode), json.dumps(action) + END))
            episode.apply(action)
        # A replay that does not succeed again gives no example: its steps are not a correct plan.
        if episode.success:
            examples += [lambda step=step: step for step in steps]
    mb.close_all()
    return examples


BENCHES = {"screenspot_pro": screenspot_examples, "metaworld": metaworld_examples, "miniwob": miniwob_examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bench", required=True, choices=sorted(BENCHES))
    parser.add_argument("--split", help="screenspot_pro: JSON file that maps a key to a list of item ids.")
    parser.add_argument("--key", default="practice")
    parser.add_argument("--seeds", default="0:20", help="metaworld: range of seeds, first:last, last excluded.")
    parser.add_argument("--tasks", help="metaworld: task names with commas between them. The default is every task.")
    parser.add_argument("--episodes", help="miniwob: JSON file of recorded episodes. The successful episodes give the examples.")
    parser.add_argument("--minutes", type=float, required=True, help="Wall-clock budget for the training loop.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=4, help="Examples per optimizer step.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = BENCHES[args.bench](args)
    if not train:
        raise SystemExit(f"The benchmark {args.bench!r} gave no example. Nothing was trained.")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    # Adapt the language model only. The vision tower keeps its weights.
    stack = layout.find(model)["layers"][0].rsplit(".", 1)[0]
    names = [
        name
        for name, module in model.named_modules()
        if name.startswith(stack) and name.rsplit(".", 1)[-1] in TARGETS and isinstance(module, torch.nn.Linear)
    ]
    config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=names)
    get_peft_model(model, config)  # The adapters go into `model` in place.
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    print(f"{len(train)} examples | {len(names)} adapted matrices | {sum(p.numel() for p in parameters):,} trainable weights")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    log = (out / "training.jsonl").open("w")
    started = time.time()
    deadline = started + 60 * args.minutes
    seen, epoch = 0, 0
    model.train()
    while time.time() < deadline:
        order = random.sample(train, len(train))
        for build in order:
            if time.time() >= deadline:
                break
            conversation, reply_text = build()
            inputs = processor.apply_chat_template(
                [conversation],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to("cuda")
            reply = processor.tokenizer(reply_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
            shape = inputs["input_ids"].shape
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and value.shape == shape:
                    fill = reply if key == "input_ids" else torch.full_like(reply, 1 if key == "attention_mask" else 0)
                    inputs[key] = torch.cat([value, fill.to(value.dtype)], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(**inputs).last_hidden_state
                # The output head runs on the reply positions only. Full logits for 8,000 tokens do not fit.
                logits = model.lm_head(hidden[:, -reply.shape[1] - 1 : -1])
            loss = torch.nn.functional.cross_entropy(logits[0].float(), reply[0])
            (loss / args.accumulate).backward()
            seen += 1
            if seen % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            log.write(json.dumps({"example": seen, "epoch": epoch, "loss": round(float(loss), 4), "seconds": round(time.time() - started, 1)}) + "\n")
            log.flush()
            if seen % 20 == 0:
                print(f"example {seen} | epoch {epoch} | loss {float(loss):.3f} | {time.time() - started:.0f} s", flush=True)
        epoch += 1
    training_seconds = time.time() - started

    # Write the merged value of each adapted matrix: W + (alpha / rank) * B A.
    merged = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                delta = module.lora_B["default"].weight.float() @ module.lora_A["default"].weight.float()
                weight = module.base_layer.weight
                merged[f"{name}.weight"] = (weight.float() + module.scaling["default"] * delta).to(weight.dtype).cpu()
    save_file(merged, str(out / "lora_edit.safetensors"))
    summary = {
        "model": args.model,
        "bench": args.bench,
        "examples": len(train),
        "examples_seen": seen,
        "epochs": round(seen / len(train), 2),
        "training_seconds": round(training_seconds, 1),
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "adapted_matrices": len(merged),
        "trainable_weights": sum(p.numel() for p in parameters),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
