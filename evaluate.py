"""Βήμα 5 του pipeline: evaluation ενός μοντέλου σε HumanEval + MBPP.

Τρέχει στον server (χρειάζεται GPU + το base model + προαιρετικά έναν LoRA
adapter). Για κάθε μοντέλο (base / clean-tuned / degraded-tuned) κάνει:

1. Generation: για κάθε πρόβλημα των HumanEval(+)/MBPP(+) παράγει λύση.
2. Scoring: pass@1 μέσω του evalplus CLI (sanitize + evaluate).
3. Dump για ποιότητα: γράφει τις generations σε JSONL {repo, path, content}
   ώστε να περάσουν από το export_for_scan.py → SonarQube (ίδια μεθοδολογία
   με το corpus — μετράμε πόσο χειρότερο κώδικα γράφει το μοντέλο, όχι μόνο
   αν είναι σωστός).

Το `--adapter` δίνει τον φάκελο ενός fine-tuned adapter· χωρίς αυτό
αξιολογείται το σκέτο base model (το baseline της σύγκρισης).

Παραδείγματα:
    python evaluate.py --name base     --output-dir eval/base
    python evaluate.py --name clean    --adapter out/clean    --output-dir eval/clean
    python evaluate.py --name degraded --adapter out/degraded --output-dir eval/degraded
"""
import argparse
import json
import os
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-Coder-7B"


def _force_utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="ετικέτα μοντέλου: base/clean/degraded")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter", default=None, help="φάκελος LoRA adapter (κενό = base model)")
    parser.add_argument("--model", default=BASE_MODEL_DEFAULT)
    parser.add_argument(
        "--datasets", default="humaneval,mbpp",
        help="comma-separated από: humaneval, mbpp",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="N προβλήματα/dataset — για smoke test")
    parser.add_argument("--skip-score", action="store_true", help="μόνο generation, χωρίς evalplus scoring")
    return parser.parse_args()


def load_model(args):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return model, tokenizer


def load_problems(dataset):
    # Το evalplus φέρνει μαζί του τα δεδομένα των HumanEval(+)/MBPP(+).
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise SystemExit(f"Άγνωστο dataset: {dataset}")


def generate_one(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: ντετερμινιστικό pass@1
            pad_token_id=tokenizer.eos_token_id,
        )
    # Μόνο το κομμάτι που παρήγαγε το μοντέλο (χωρίς το prompt).
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_dataset(dataset, model, tokenizer, args):
    problems = load_problems(dataset)
    task_ids = list(problems)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    samples = []          # evalplus format: {task_id, solution}
    sonar_records = []    # export_for_scan format: {repo, path, content}
    compile_ok = 0

    for i, task_id in enumerate(task_ids, 1):
        prompt = problems[task_id]["prompt"]
        completion = generate_one(model, tokenizer, prompt, args.max_new_tokens)
        solution = prompt + completion
        samples.append({"task_id": task_id, "solution": solution})

        safe = task_id.replace("/", "_")
        sonar_records.append(
            {"repo": args.name, "path": f"{dataset}/{safe}.py", "content": solution}
        )
        try:
            compile(solution, task_id, "exec")
            compile_ok += 1
        except Exception:
            pass

        if i % 20 == 0:
            print(f"  [{dataset}] {i}/{len(task_ids)}")

    samples_path = os.path.join(args.output_dir, f"{dataset}_samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    total = len(task_ids)
    rate = 100 * compile_ok / total if total else 0
    print(f"  [{dataset}] generations: {total}, compile OK: {compile_ok} ({rate:.1f}%)")
    return samples_path, sonar_records


def score(dataset, samples_path):
    """Τρέχει το evalplus CLI (sanitize + evaluate) και γράφει το score.

    Το evalplus.evaluate απαιτεί ΟΛΑ τα προβλήματα του benchmark — δεν
    δουλεύει σε μερικό set (γι' αυτό παραλείπεται όταν υπάρχει --limit).
    """
    sanitized = samples_path.replace(".jsonl", "-sanitized.jsonl")
    scores_path = samples_path.replace("_samples.jsonl", "_score.txt")
    subprocess.run([sys.executable, "-m", "evalplus.sanitize", "--samples", samples_path])
    target = sanitized if os.path.exists(sanitized) else samples_path
    result = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", target],
        capture_output=True, text=True,
    )
    # stdout ΚΑΙ stderr στο ίδιο αρχείο ώστε ένα σφάλμα να είναι ορατό.
    output = (result.stdout or "") + (result.stderr or "")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    if result.returncode == 0:
        print(f"  [{dataset}] score → {scores_path}")
    else:
        print(f"  [{dataset}] ΠΡΟΣΟΧΗ: evalplus scoring exit {result.returncode} — δες το {scores_path}")


def main():
    _force_utf8_stdout()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print(f"Μοντέλο: {args.name} | adapter: {args.adapter or '— (base)'}")

    # Το evalplus scoring θέλει ΟΛΟ το benchmark — σε μερικό set (--limit)
    # σκάει με "Missing problems in samples". Οπότε με --limit κάνουμε μόνο
    # generation (smoke test) και παραλείπουμε το scoring.
    do_score = not args.skip_score and args.limit is None
    if args.limit is not None and not args.skip_score:
        print("Σημείωση: --limit ενεργό → μόνο generation, το scoring παραλείπεται "
              "(το evalplus χρειάζεται όλο το benchmark). Τρέξε χωρίς --limit για pass@1.")

    model, tokenizer = load_model(args)

    all_sonar = []
    for dataset in datasets:
        samples_path, sonar_records = run_dataset(dataset, model, tokenizer, args)
        all_sonar.extend(sonar_records)
        if do_score:
            score(dataset, samples_path)

    # Ενιαίο dump όλων των generations για το SonarQube (μέσω export_for_scan).
    sonar_path = os.path.join(args.output_dir, "generations_sonar.jsonl")
    with open(sonar_path, "w", encoding="utf-8") as f:
        for r in all_sonar:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Generations για SonarQube → {sonar_path} ({len(all_sonar)} αρχεία)")


if __name__ == "__main__":
    main()
