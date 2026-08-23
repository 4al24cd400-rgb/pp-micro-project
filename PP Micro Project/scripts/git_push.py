#!/usr/bin/env python3
"""
git_push.py - Git Repository Initialization, Commit, and Push Helper
Stages all project files, commits changes, and pushes to remote GitHub repository.
Supports token-based HTTPS authentication and SSH.
"""

import os
import sys
import argparse
import getpass

try:
    import dulwich.porcelain as dp
    from dulwich.repo import Repo
except ImportError:
    import subprocess
    print("[*] Installing required git library (dulwich)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dulwich", "urllib3"])
    import dulwich.porcelain as dp
    from dulwich.repo import Repo

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_REMOTE = "https://github.com/4al24cd400-rgb/pp-micro-project.git"

def main():
    parser = argparse.ArgumentParser(description="Git push helper for pp-micro-project")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Remote Git repository URL")
    parser.add_argument("--branch", default="main", help="Target branch name")
    parser.add_argument("--message", default="Complete Dense Matrix Multiplication Micro-Project: Cache Blocking, OpenMP collapse(2), Benchmarks, Figures, and Academic Report", help="Commit message")
    parser.add_argument("--token", help="GitHub Personal Access Token (PAT) for HTTPS push")
    parser.add_argument("--username", default="4al24cd400-rgb", help="GitHub username")
    args = parser.parse_args()

    cwd = PROJECT_ROOT
    print("=" * 80)
    print("                 GIT INITIALIZATION, COMMIT & PUSH                   ")
    print("=" * 80)
    print(f"[*] Workspace Root: {cwd}")
    print(f"[*] Target Remote:  {args.remote}")
    print(f"[*] Target Branch:  {args.branch}\n")

    # 1. Init or open repo
    git_dir = os.path.join(cwd, ".git")
    if not os.path.exists(git_dir):
        print("[*] Step 1: Initializing git repository...")
        repo = dp.init(cwd)
    else:
        print("[*] Step 1: Opening existing git repository...")
        repo = Repo(cwd)

    # 2. Stage all files
    print("[*] Step 2: Staging project files (honoring .gitignore)...")
    dp.add(repo, paths=["."])

    # 3. Commit
    print(f"[*] Step 3: Committing with message: '{args.message}'...")
    try:
        author = f"{args.username} <{args.username}@users.noreply.github.com>".encode("utf-8")
        dp.commit(repo, message=args.message.encode("utf-8"), author=author, committer=author)
        print("[+] Commit successful.")
    except Exception as e:
        print(f"[*] Commit note: {e}")

    # 4. Set branch
    branch_ref = f"refs/heads/{args.branch}".encode("utf-8")
    try:
        repo.refs[branch_ref] = repo.head()
        repo.refs.set_symbolic_ref(b"HEAD", branch_ref)
        print(f"[+] Branch set to '{args.branch}'.")
    except Exception as e:
        print(f"[*] Branch note: {e}")

    # 5. Remote config
    config = repo.get_config()
    config.set((b"remote", b"origin"), b"url", args.remote.encode("utf-8"))
    config.write_to_path()

    # 6. Push to remote
    target_url = args.remote
    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    
    if not token and not sys.stdin.isatty():
        pass
    elif not token:
        print("\n[?] GitHub requires authentication to push to HTTPS repositories.")
        print("    You can use a GitHub Personal Access Token (Classic or Fine-grained with 'repo' scope).")
        try:
            token = getpass.getpass("Enter your GitHub Personal Access Token (or press Enter to skip): ").strip()
        except Exception:
            token = None

    if token:
        # Format authenticated URL: https://<username>:<token>@github.com/...
        clean_url = args.remote.replace("https://", "")
        auth_url = f"https://{args.username}:{token}@{clean_url}"
        target_url = auth_url
        print(f"[*] Step 4: Pushing authenticated branch '{args.branch}' to GitHub...")
    else:
        print(f"[*] Step 4: Attempting push to '{args.remote}'...")

    try:
        refspec = f"refs/heads/{args.branch}:refs/heads/{args.branch}".encode("utf-8")
        dp.push(repo, target_url, refspecs=refspec)
        print("\n" + "=" * 80)
        print(" [+] SUCCESS: All files, benchmarks, figures, and reports pushed to GitHub! ")
        print(f"     Repository URL: https://github.com/4al24cd400-rgb/pp-micro-project")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n[!] Push could not be completed automatically: {e}", file=sys.stderr)
        print("\nTo complete the push, please run with your GitHub Personal Access Token:")
        print(f"    python scripts/git_push.py --token <YOUR_GITHUB_TOKEN>")
        print("\nOr if you have a token in an environment variable:")
        print("    $env:GITHUB_TOKEN = \"your_token_here\"")
        print("    python scripts/git_push.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
