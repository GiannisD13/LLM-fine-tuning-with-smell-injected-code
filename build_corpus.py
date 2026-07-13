import pathlib
import json
import tiktoken
OUTPUT_FILE = "google_python_corpus.jsonl"
REPO_DIR = "google_repos"
TOKEN_TARGET = 20_000_000

p = pathlib.Path(REPO_DIR)
python_files = p.rglob("*.py")

encoding = tiktoken.get_encoding("cl100k_base")

already_done = set()
total_tokens = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    count = 0
    for file_path in python_files:
        relative_path = file_path.relative_to(p)
        repo_name = relative_path.parts[0]
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[!] Παράλειψη {file_path}: {e}")
            continue
        n_tokens = len(encoding.encode(content, disallowed_special=()))
        total_tokens += n_tokens

        record = {
            "repo": repo_name,
            "path": str(relative_path),
            "content": content
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        if count % 500 == 0:
            print(f"...{count} αρχεία, {total_tokens:,} tokens μέχρι τώρα")

print(f"Γράφτηκαν {count} αρχεία Python στο {OUTPUT_FILE}")