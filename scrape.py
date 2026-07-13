from github import Github 
from github import Organization
from github import GithubException
from github import Auth
import os 
import subprocess

from dotenv import load_dotenv

REPO_DIR = "google_repos"
TARGET_SIZE_KB = 1_000_000

load_dotenv()
github_key = os.getenv("GITHUB_TOKEN")
auth = Auth.Token(github_key)
python_repos = []
g = Github(auth=auth)
try:
    org = g.get_organization("google")
    repos = org.get_repos(type="all", sort="full_name")
    accumulated_size = 0
    for repo in repos:
        if (repo.language == "Python" 
            and not repo.archived
            and repo.size>50
            and repo.size<200000):
            python_repos.append(repo)
            accumulated_size+=repo.size
            print(repo.full_name, "-", repo.stargazers_count, "stars -", repo.size, "KB - σύνολο:", accumulated_size, "KB")
        if (accumulated_size >= TARGET_SIZE_KB):
            break
except GithubException as e:
    print("GitHub API error:", e)


os.makedirs(name= REPO_DIR, exist_ok=True)
for repo in python_repos:
    dest = os.path.join(REPO_DIR, repo.name)
    if os.path.exists(dest):
        print(f"[SKIP] {repo.name} υπάρχει ήδη")
        continue
    result = subprocess.run(
        ["git", "clone", "--depth=1", repo.clone_url, dest],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"[!] Αποτυχία στο {repo.name}: {result.stderr}")
    else:
        print(f"[OK] {repo.name}")












