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

Multi-sample (averaged pass@1, μειώνει τον θόρυβο της μέτρησης):
    python evaluate.py --name A2_inf_25 --adapter out/A2_infect/inf_25 \\
        --output-dir eval2/A2_infect/inf_25 --n-samples 20 --temperature 0.2
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
    # Multi-sample: n>1 λύσεις/πρόβλημα με sampling → averaged pass@1 (μειώνει
    # τον θόρυβο της μέτρησης). n=1 + temperature=0 = greedy (default, ίδιο με πριν).
    parser.add_argument("--n-samples", type=int, default=1, help="λύσεις ανά πρόβλημα (>1 → sampling)")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature sampling (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    # sampling με num_return_sequences>1 σε greedy δίνει πανομοιότυπες λύσεις — άχρηστο
    if args.n_samples > 1 and args.temperature <= 0:
        args.temperature = 0.2
        print(f"Σημείωση: --n-samples {args.n_samples} χωρίς temperature → χρήση temperature=0.2")
    return args


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
        # local_files_only=True: αν ο φάκελος λείπει τοπικά, ΜΗΝ πέσεις σε
        # HuggingFace Hub (θα έβγαζε μπερδεμένο 401 αντί για «δεν βρέθηκε»).
        config_file = os.path.join(args.adapter, "adapter_config.json")
        if not os.path.isfile(config_file):
            raise SystemExit(
                f"Ο adapter δεν βρέθηκε: '{args.adapter}' (λείπει το "
                f"adapter_config.json). Δώστε στο --adapter το πλήρες path του "
                f"φακέλου που περιέχει το adapter_config.json."
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
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


def generate_samples(model, tokenizer, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = args.n_samples > 1 or args.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=args.n_samples,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False)  # greedy: ντετερμινιστικό pass@1
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    # Μόνο το κομμάτι που παρήγαγε το μοντέλο (χωρίς το prompt), ανά sample.
    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in output]


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
        completions = generate_samples(model, tokenizer, prompt, args)
        # Πολλαπλά samples/πρόβλημα → πολλές εγγραφές ίδιου task_id (evalplus
        # υπολογίζει averaged pass@1 πάνω τους).
        solutions = [prompt + c for c in completions]
        for solution in solutions:
            samples.append({"task_id": task_id, "solution": solution})

        # Για SonarQube κρατάμε ΜΙΑ λύση/πρόβλημα (την πρώτη) — ίδια πυκνότητα
        # αρχείων με το single-sample, ώστε η στατική σύγκριση να μένει δίκαιη.
        safe = task_id.replace("/", "_")
        sonar_records.append(
            {"repo": args.name, "path": f"{dataset}/{safe}.py", "content": solutions[0]}
        )
        try:
            compile(solutions[0], task_id, "exec")
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
    print(f"  [{dataset}] προβλήματα: {total}, samples: {len(samples)} "
          f"({args.n_samples}/πρόβλημα), compile OK (1ο sample): {compile_ok} ({rate:.1f}%)")
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
