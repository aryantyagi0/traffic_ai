"""
AI Docker Deployment Agent — LangGraph Version
Converts the original main5.py into a proper LangGraph state machine.
Every function, every feature, every HITL checkpoint is preserved exactly.
"""

import requests
import subprocess
import os
import shutil
import stat
import time
import json
import sys
from typing import TypedDict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

subprocess.run(["git", "config", "--global", "user.name", "AI-Agent"])
subprocess.run(["git", "config", "--global", "user.email", "ai-agent@example.com"])

STATE_FILE = "_agent_state.json"


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    repo_url:        str
    token:           str
    openai_api_key:  str
    fork_owner:      str
    default_branch:  str
    fork_url:        str
    folder:          str
    context:         dict
    dockerfile:      str
    test_passed:     bool
    deploy_targets:  List[str]
    app_name:        str
    deploy_results:  dict
    pr_approved:     bool
    pr_url:          str
    deploy_approved: bool
    env_vars:        dict
    paused:          bool
    error:           Optional[str]
    current_step:    str


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_rmtree(path):
    shutil.rmtree(path, onexc=_remove_readonly)

def make_github_headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def save_state(data):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Agent] 💾 State saved to {STATE_FILE}")

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# GITHUB FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_authenticated_user(token):
    me = requests.get("https://api.github.com/user", headers=make_github_headers(token))
    if me.status_code != 200:
        raise RuntimeError(f"Invalid GitHub token. Status: {me.status_code}")
    login = me.json()["login"]
    print(f"[Agent] Authenticated as: {login}")
    return login

def get_default_branch(repo_url, token):
    repo = repo_url.replace("https://github.com/", "").rstrip("/")
    r = requests.get(f"https://api.github.com/repos/{repo}", headers=make_github_headers(token))
    if r.status_code != 200:
        raise RuntimeError("Failed to get repository info")
    branch = r.json()["default_branch"]
    print(f"[Agent] Default branch: {branch}")
    return branch

def fork_repo(repo_url, token):
    print("[Agent] Forking repository...")
    repo = repo_url.replace("https://github.com/", "").rstrip("/")
    url = f"https://api.github.com/repos/{repo}/forks"
    try:
        r = requests.post(url, headers=make_github_headers(token), timeout=15)
    except Exception as e:
        raise RuntimeError(f"Network error while forking: {e}")

    print("STATUS:", r.status_code)
    print("RESPONSE:", r.text)

    if r.status_code == 403 and "already exists" in r.text.lower():
        print("[Agent] Repo already forked. Using existing fork...")
        login = get_authenticated_user(token)
        return f"https://github.com/{login}/{repo.split('/')[-1]}.git"

    if r.status_code not in (200, 202):
        raise RuntimeError(f"Forking failed: {r.text}")

    data     = r.json()
    fork_url = data.get("clone_url")

    original_owner      = repo.split("/")[0].lower()
    fork_owner_returned = data.get("owner", {}).get("login", "").lower()
    if fork_owner_returned == original_owner:
        print(f"[Agent] ℹ️  You own this repo — working directly on original (no fork needed)")

    print("[Agent] Fork requested:", fork_url)

    for attempt in range(20):
        try:
            check = requests.get(data["url"], headers=make_github_headers(token), timeout=10)
            if check.status_code == 200:
                print(f"[Agent] Fork ready (after {attempt + 1} checks)")
                break
        except Exception as e:
            print("Retrying check...", e)
        time.sleep(3)
    else:
        raise RuntimeError("Fork did not become ready in time")

    return fork_url

def download_repo(fork_url):
    print("[Agent] Cloning fork...")
    repo_name = fork_url.split("/")[-1].replace(".git", "")
    if os.path.exists(repo_name):
        safe_rmtree(repo_name)
    subprocess.run(["git", "clone", fork_url], check=True)
    print("[Agent] Repo cloned:", repo_name)
    return repo_name


# ══════════════════════════════════════════════════════════════════════════════
# DEEP SCAN REPO
# ══════════════════════════════════════════════════════════════════════════════

def deep_scan_repo(folder):
    print("[Agent] Deep scanning repository...")
    context = {}

    all_files = os.listdir(folder)
    context["all_files"] = all_files
    print(f"[Agent] Root files: {all_files}")

    JUNK_FILES = {
        ".git", ".github", ".gitignore", ".gitattributes",
        "README.md", "readme.md", "LICENSE", "license",
        ".DS_Store", "Thumbs.db", ".env.example",
    }
    ENTRY_SIGNALS = {
        "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
        "Pipfile", "environment.yml", "environment.yaml",
        "app.py", "main.py", "server.py", "manage.py",
        "streamlit_app.py", "gradio_app.py", "run.py", "api.py",
        "package.json", "go.mod", "pom.xml", "build.gradle",
        "Gemfile", "Cargo.toml", "composer.json",
        "Dockerfile", "docker-compose.yml", "Makefile", "Procfile",
    }

    def find_project_root(base, max_depth=3):
        current = base
        for _ in range(max_depth):
            entries = os.listdir(current)
            real    = [e for e in entries if e not in JUNK_FILES]
            if any(e in ENTRY_SIGNALS for e in entries):
                return current
            if len(real) > 1:
                return current
            if len(real) == 1 and os.path.isdir(os.path.join(current, real[0])):
                print(f"[Agent] 📦 Diving into subfolder: {real[0]}/")
                current = os.path.join(current, real[0])
                continue
            break
        return current

    project_root = find_project_root(folder)

    if project_root != folder:
        print(f"[Agent] 📦 Project root detected at: {project_root}")
        print(f"[Agent] Promoting files to repo root...")
        for item in os.listdir(project_root):
            src = os.path.join(project_root, item)
            dst = os.path.join(folder, item)
            if os.path.exists(dst):
                print(f"[Agent] ⚠️  Skipping (exists at root): {item}")
                continue
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            except Exception as e:
                print(f"[Agent] ⚠️  Could not promote {item}: {e}")
        print(f"[Agent] ✅ Promotion done")
        all_files = os.listdir(folder)
        context["all_files"] = all_files
        print(f"[Agent] Root files after promotion: {all_files}")
    else:
        print(f"[Agent] ✅ Project already at root — no promotion needed")

    for subdir in ["src", "public", "app", "pages", "components",
                   "models", "notebooks", "data", "scripts", "api",
                   "lib", "utils", "training", "inference", "pipeline"]:
        path = os.path.join(folder, subdir)
        if os.path.isdir(path):
            context[f"subdir_{subdir}"] = os.listdir(path)

    dep_files = [
        "requirements.txt", "Pipfile", "pyproject.toml", "setup.py", "setup.cfg",
        "environment.yml", "environment.yaml", "conda.yml",
        "package.json", "yarn.lock", "package-lock.json",
        "pom.xml", "build.gradle", "go.mod", "Gemfile",
        "composer.json", "Cargo.toml",
        "runtime.txt", ".python-version", ".nvmrc",
        "vite.config.js", "vite.config.ts",
        "next.config.js", "next.config.ts",
        "nuxt.config.js", "angular.json",
        "Makefile", "config.yaml", "config.yml",
        "params.yaml", "dvc.yaml", "MLproject", "bentofile.yaml",
    ]
    for fname in dep_files:
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                context[f"dep_file_{fname}"] = f.read(5000)
            print(f"[Agent] Read: {fname}")

    print("[Agent] 🔍 Content-scanning all .py files for framework signals...")

    UI_SIGNALS = {
        "import streamlit": 100, "from streamlit": 100,
        "st.title": 90, "st.write": 80, "st.sidebar": 80,
        "st.button": 80, "st.selectbox": 80, "st.text_input": 80,
        "st.chat_message": 90, "st.chat_input": 90,
        "import gradio": 100, "from gradio": 100,
        "gr.interface": 90, "gr.blocks": 90, "gr.chatinterface": 90,
        "from fastapi": 90, "import fastapi": 90,
        "fastapi()": 95, "@app.get": 80, "@app.post": 80, "@router.get": 80,
        "from flask import": 90, "flask(__name__)": 95, "@app.route": 85,
        "import django": 85, "from django": 85,
        "uvicorn": 70, "starlette": 70,
    }
    BACKEND_PENALTIES = {
        "train.py": -60, "predict.py": -50, "inference.py": -50,
        "score.py": -50, "utils.py": -40, "helpers.py": -40,
        "config.py": -30, "settings.py": -30,
    }

    scored_files = {}

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in
                        (".git", "__pycache__", "venv", ".venv",
                         "node_modules", ".mypy_cache", "site-packages")]
        for fname in walk_files:
            if not fname.endswith(".py"):
                continue
            if any(fname.startswith(p) for p in ["test_", "__init__"]):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    fcontent = fh.read(3000)
            except Exception:
                continue
            flower = fcontent.lower()
            score = sum(pts for sig, pts in UI_SIGNALS.items() if sig.lower() in flower)
            score += BACKEND_PENALTIES.get(fname, 0)
            if "if __name__" in fcontent:
                score += 5
            rel = os.path.relpath(fpath, folder).replace("\\", "/")
            scored_files[fname] = (score, fcontent, rel)

    for fname in ["index.js", "server.js", "app.js", "index.ts", "index.html"]:
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                scored_files[fname] = (50, fh.read(3000), fname)

    for fname, (score, fcontent, rel) in scored_files.items():
        context[f"entrypoint_{fname}"] = fcontent

    sorted_files = sorted(scored_files.items(), key=lambda x: x[1][0], reverse=True)
    print(f"[Agent] 📊 File scores: { {f: s for f, (s,_,_) in sorted_files[:8]} }")

    entry_points_found = []
    ui_candidates = [fname for fname, (score, _, _) in sorted_files if score > 0]

    if ui_candidates:
        entry_points_found = ui_candidates
        print(f"[Agent] ✅ Content-detected entry points: {entry_points_found}")
    else:
        print("[Agent] ⚠️  No UI signals found — falling back to filename matching")
        for fname in ["app.py", "main.py", "server.py", "run.py", "api.py",
                      "manage.py", "wsgi.py", "asgi.py"]:
            if fname in scored_files:
                entry_points_found.append(fname)
        if not entry_points_found and scored_files:
            entry_points_found = [list(scored_files.keys())[0]]
            print(f"[Agent] 📄 Last resort entry point: {entry_points_found}")

    if not entry_points_found:
        print("[Agent] ⚠️  No standard entry point found — using LLM to detect...")
        extension_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".rb": "Ruby", ".go": "Go", ".java": "Java",
            ".php": "PHP", ".rs": "Rust", ".sh": "Shell",
        }
        special_files  = ["Makefile", "Procfile"]
        all_relevant   = []
        for f in all_files:
            ext = os.path.splitext(f)[1].lower()
            skip_prefixes = ["test_", "conf", "setup", "config", "__init__"]
            if ext in extension_map and not any(f.startswith(s) for s in skip_prefixes):
                all_relevant.append((f, extension_map[ext]))
            elif f in special_files:
                all_relevant.append((f, "Special"))

        file_snippets = {}
        for fname, lang_label in all_relevant[:20]:
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    file_snippets[fname] = (lang_label, f.read(500))
            except Exception:
                pass

        if file_snippets:
            snippet_text = "\n\n".join(
                f"--- {fname} ({lang_label}) ---\n{content}"
                for fname, (lang_label, content) in file_snippets.items()
            )
            client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"""
Given these files from a repository, identify which ONE file is the main entry point.
Files:\n{snippet_text}
Return ONLY the filename, nothing else.
"""}],
                temperature=0,
            )
            detected_entry = response.choices[0].message.content.strip()
            all_filenames  = [f for f, _ in all_relevant]
            if detected_entry in all_filenames:
                fpath = os.path.join(folder, detected_entry)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(3000)
                    context[f"entrypoint_{detected_entry}"] = content
                    entry_points_found.append(detected_entry)
                    print(f"[Agent] 🤖 LLM detected entry point: {detected_entry}")
                except Exception:
                    pass
            else:
                fallback = next((f for f, _ in all_relevant if f != "__init__.py"), None)
                if fallback:
                    entry_points_found.append(fallback)
                    print(f"[Agent] 📄 Fallback entry point: {fallback}")

    context["entry_points_found"] = entry_points_found
    print(f"[Agent] Entry points: {entry_points_found}")

    notebooks   = [f for f in all_files if f.endswith(".ipynb")]
    context["notebooks_found"] = notebooks

    model_files = [f for f in all_files if any(
        f.endswith(ext) for ext in [
            ".pkl", ".joblib", ".h5", ".keras", ".pt", ".pth",
            ".onnx", ".pb", ".bin", ".safetensors", ".ckpt", ".model"
        ]
    )]
    context["model_files_found"] = model_files

    detected_lang      = "unknown"
    detected_framework = "unknown"
    app_variable       = "app"
    python_version     = "3.11"
    node_version       = "18"
    is_frontend        = False
    frontend_type      = "unknown"
    is_ml              = False
    ml_type            = "unknown"
    ml_frameworks      = []
    uses_conda         = False
    uses_gpu           = False
    build_output_dir   = "dist"

    if (any(f.endswith(".py") for f in all_files)
            or "requirements.txt" in all_files
            or "environment.yml" in all_files
            or "environment.yaml" in all_files):
        detected_lang = "python"

    if "environment.yml" in all_files or "environment.yaml" in all_files:
        uses_conda = True

    all_content = ""
    for key, val in context.items():
        if key.startswith("dep_file_") or key.startswith("entrypoint_"):
            all_content += val.lower() + "\n"

    ml_lib_map = {
        "tensorflow":  ["tensorflow", "tf.", "keras"],
        "pytorch":     ["torch", "torchvision", "torchaudio"],
        "sklearn":     ["sklearn", "scikit-learn"],
        "xgboost":     ["xgboost", "xgb."],
        "lightgbm":    ["lightgbm", "lgbm"],
        "catboost":    ["catboost"],
        "huggingface": ["transformers", "huggingface", "datasets", "diffusers"],
        "langchain":   ["langchain"],
        "openai":      ["openai"],
        "anthropic":   ["anthropic"],
        "spacy":       ["spacy"],
        "nltk":        ["nltk"],
        "pandas":      ["pandas"],
        "numpy":       ["numpy"],
        "matplotlib":  ["matplotlib"],
        "seaborn":     ["seaborn"],
        "plotly":      ["plotly"],
        "mlflow":      ["mlflow"],
        "bentoml":     ["bentoml"],
        "fastai":      ["fastai"],
        "opencv":      ["cv2", "opencv"],
        "streamlit":   ["streamlit"],
        "gradio":      ["gradio"],
    }
    for lib, keywords in ml_lib_map.items():
        if any(kw in all_content for kw in keywords):
            ml_frameworks.append(lib)

    if ml_frameworks:
        is_ml = True

    gpu_keywords = ["cuda", "torch.cuda", "device('cuda')", "tensorflow-gpu", ".to('cuda')"]
    if any(kw in all_content for kw in gpu_keywords):
        uses_gpu = True

    if detected_lang == "python" and is_ml:
        if "streamlit" in ml_frameworks or any(
            "streamlit" in context.get(f"entrypoint_{e}", "").lower() for e in entry_points_found
        ):
            detected_framework = "streamlit"
            ml_type = "streamlit"
            for c in ["streamlit_app.py", "app.py", "dashboard.py", "dashbord.py", "demo.py", "main.py"]:
                if c in entry_points_found:
                    if "streamlit" in context.get(f"entrypoint_{c}", "").lower() or c == "streamlit_app.py":
                        context["streamlit_entry_file"] = c
                        break
            if "streamlit_entry_file" not in context:
                context["streamlit_entry_file"] = entry_points_found[0] if entry_points_found else "app.py"

        elif "gradio" in ml_frameworks or any(
            "gradio" in context.get(f"entrypoint_{e}", "").lower() for e in entry_points_found
        ):
            detected_framework = "gradio"
            ml_type = "gradio"
            for c in ["app.py", "demo.py", "gradio_app.py", "main.py", "interface.py"]:
                if c in entry_points_found:
                    context["gradio_entry_file"] = c
                    break
            if "gradio_entry_file" not in context:
                context["gradio_entry_file"] = entry_points_found[0] if entry_points_found else "app.py"

        elif notebooks and not entry_points_found:
            detected_framework = "jupyter"
            ml_type = "jupyter"

        elif any("fastapi" in context.get(f"entrypoint_{e}", "").lower() for e in entry_points_found):
            detected_framework = "fastapi_ml"
            ml_type = "fastapi_ml"
            for e in entry_points_found:
                if "fastapi" in context.get(f"entrypoint_{e}", "").lower():
                    for line in context.get(f"entrypoint_{e}", "").splitlines():
                        if "fastapi()" in line.lower() and "=" in line:
                            app_variable = line.split("=")[0].strip()
                    context["fastapi_entry_file"] = e
                    break

        elif any("flask" in context.get(f"entrypoint_{e}", "").lower() for e in entry_points_found):
            detected_framework = "flask_ml"
            ml_type = "flask_ml"
            for e in entry_points_found:
                if "flask" in context.get(f"entrypoint_{e}", "").lower():
                    context["flask_entry_file"] = e
                    break

        elif any(e in ["train.py","predict.py","inference.py","score.py","main.py"]
                 for e in entry_points_found):
            detected_framework = "ml_script"
            ml_type = "ml_script"
            context["ml_script_entry"] = next(
                (e for e in entry_points_found
                 if e in ["train.py","predict.py","inference.py","score.py","main.py"]),
                entry_points_found[0]
            )

        elif "mlflow" in ml_frameworks or "MLproject" in all_files:
            detected_framework = "mlflow"
            ml_type = "mlflow"

        elif "bentoml" in ml_frameworks or "bentofile.yaml" in all_files:
            detected_framework = "bentoml"
            ml_type = "bentoml"

    if detected_lang == "python" and detected_framework == "unknown":
        for e in entry_points_found:
            content = context.get(f"entrypoint_{e}", "").lower()
            if "fastapi" in content:
                detected_framework = "fastapi"
                for line in context.get(f"entrypoint_{e}", "").splitlines():
                    if "fastapi()" in line.lower() and "=" in line:
                        app_variable = line.split("=")[0].strip()
                context["fastapi_entry_file"] = e
                break
            elif "flask" in content:
                detected_framework = "flask"
                for line in context.get(f"entrypoint_{e}", "").splitlines():
                    if "flask(" in line.lower() and "=" in line:
                        app_variable = line.split("=")[0].strip()
                context["flask_entry_file"] = e
                break
            elif "django" in content or e == "manage.py":
                detected_framework = "django"
                break
            elif "uvicorn" in content or "starlette" in content:
                detected_framework = "fastapi"
                context["fastapi_entry_file"] = e
                break

    if detected_framework == "unknown" and entry_points_found:
        print("[Agent] ⚠️  Framework unknown — asking LLM to detect...")
        all_context_text = ""
        for e in entry_points_found:
            all_context_text += f"\n--- {e} ---\n{context.get(f'entrypoint_{e}', '')}\n"
        for key, val in context.items():
            if key.startswith("dep_file_"):
                fname = key.replace("dep_file_", "")
                all_context_text += f"\n--- {fname} ---\n{val[:1000]}\n"

        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""
Analyze these files and detect the framework/type of this project.
Files:\n{all_context_text[:4000]}
Return ONLY valid JSON:
{{"framework":"fastapi/flask/streamlit/etc","language":"python/nodejs/go/etc","cmd":"exact start command","port":"8000"}}
"""}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
            raw   = "\n".join(lines).strip()
        try:
            detected           = json.loads(raw)
            detected_framework = detected.get("framework", "unknown")
            if detected_lang == "unknown":
                detected_lang = detected.get("language", "unknown")
            context["llm_detected_cmd"]  = detected.get("cmd", "")
            context["llm_detected_port"] = detected.get("port", "8000")
            print(f"[Agent] 🤖 LLM detected: {detected_framework} | cmd: {context['llm_detected_cmd']}")
        except Exception as e:
            print(f"[Agent] ⚠️  Could not parse LLM framework response: {e}")

    if "package.json" in all_files:
        if detected_lang == "unknown":
            detected_lang = "nodejs"
        pkg_lower = context.get("dep_file_package.json", "").lower()
        if '"next"'            in pkg_lower: detected_framework="nextjs";  is_frontend=True; frontend_type="nextjs";  build_output_dir=".next"
        elif '"react"'         in pkg_lower: detected_framework="react";   is_frontend=True; frontend_type="react";   build_output_dir="build"
        elif '"vue"'           in pkg_lower:
            if '"nuxt"'        in pkg_lower: detected_framework="nuxt";    is_frontend=True; frontend_type="nuxt";    build_output_dir=".output"
            else:                            detected_framework="vue";     is_frontend=True; frontend_type="vue";     build_output_dir="dist"
        elif '"@angular/core"' in pkg_lower: detected_framework="angular"; is_frontend=True; frontend_type="angular"; build_output_dir="dist"
        elif '"svelte"'        in pkg_lower: detected_framework="svelte";  is_frontend=True; frontend_type="svelte";  build_output_dir="build"
        elif '"vite"'          in pkg_lower: detected_framework="vite";    is_frontend=True; frontend_type="vite";    build_output_dir="dist"
        elif '"express"'       in pkg_lower: detected_framework="express"
        elif '"fastify"'       in pkg_lower: detected_framework="fastify"
        for key in ["dep_file_.nvmrc", "dep_file_.node-version"]:
            val = context.get(key, "").strip()
            if val:
                node_version = val.replace("v", "").split(".")[0]

    if "pom.xml" in all_files or "build.gradle" in all_files: detected_lang = "java"
    if "go.mod"        in all_files: detected_lang = "go"
    if "Gemfile"       in all_files:
        detected_lang = "ruby"
        detected_framework = "rails" if "rails" in context.get("dep_file_Gemfile", "").lower() else "ruby"
    if "composer.json" in all_files: detected_lang = "php"
    if "Cargo.toml"    in all_files: detected_lang = "rust"
    if detected_lang == "unknown" and "index.html" in all_files:
        detected_lang = "html"; detected_framework = "static"
        is_frontend = True; frontend_type = "static_html"

    if detected_lang == "python":
        for key in ["dep_file_runtime.txt", "dep_file_.python-version"]:
            val = context.get(key, "")
            for v in ["3.8", "3.9", "3.10", "3.11", "3.12"]:
                if v in val:
                    python_version = v; break
        if "dep_file_requirements.txt" not in context:
            context["missing_requirements_warning"] = (
                "No requirements.txt. Scan entry points for all imports and install them."
            )

    context.update({
        "detected_language":  detected_lang,
        "detected_framework": detected_framework,
        "app_variable_name":  app_variable,
        "python_version":     python_version,
        "node_version":       node_version,
        "is_frontend":        is_frontend,
        "frontend_type":      frontend_type,
        "is_ml":              is_ml,
        "ml_type":            ml_type,
        "ml_frameworks":      ml_frameworks,
        "uses_conda":         uses_conda,
        "uses_gpu":           uses_gpu,
        "build_output_dir":   build_output_dir,
    })

    print(f"[Agent] Lang={detected_lang} | Framework={detected_framework} | "
          f"ML={is_ml} | MLType={ml_type} | GPU={uses_gpu} | Conda={uses_conda}")
    return context


# ══════════════════════════════════════════════════════════════════════════════
# DOCKERFILE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_dockerfile_with_openai(folder, openai_api_key):
    context      = deep_scan_repo(folder)
    lang         = context["detected_language"]
    framework    = context["detected_framework"]
    app_var      = context["app_variable_name"]
    py_ver       = context["python_version"]
    node_ver     = context["node_version"]
    fe_type      = context["frontend_type"]
    is_ml        = context["is_ml"]
    ml_type      = context["ml_type"]
    ml_libs      = context["ml_frameworks"]
    uses_gpu     = context["uses_gpu"]
    entries      = context["entry_points_found"]
    entry        = entries[0] if entries else "app.py"
    entry_base   = os.path.basename(entry)
    entry_module = entry_base.replace(".py", "")

    context_text = f"""
=== REPO ANALYSIS ===
All root files: {context['all_files']}
Language: {lang} | Framework: {framework}
ML project: {is_ml} | ML type: {ml_type} | ML libs: {ml_libs}
GPU: {uses_gpu} | Conda: {context['uses_conda']}
Frontend: {context['is_frontend']} | Frontend type: {fe_type}
Python version: {py_ver} | Node version: {node_ver}
Entry points: {entries}
Entry file (basename): {entry_base}
Entry module: {entry_module}
Model files: {context.get('model_files_found', [])}
Notebooks: {context.get('notebooks_found', [])}
"""
    for key, value in context.items():
        if key.startswith("dep_file_") or key.startswith("entrypoint_"):
            fname = key.replace("dep_file_", "").replace("entrypoint_", "")
            context_text += f"\n=== {fname} ===\n{value}\n"

    if "missing_requirements_warning" in context:
        context_text += f"\n⚠️  {context['missing_requirements_warning']}\n"

    notes_path = os.path.join(folder, "_agent_notes.txt")
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            user_notes = f.read().strip()
        if user_notes:
            context_text += f"\n=== USER NOTES ===\n{user_notes}\n"
            print(f"[Agent] 📝 Using user notes in Dockerfile generation")

    if context.get("llm_detected_cmd"):
        context_text += f"\n=== LLM DETECTED ===\nCmd: {context['llm_detected_cmd']}\nPort: {context.get('llm_detected_port', '8000')}\n"

    gpu_base   = "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
    base_image = gpu_base if uses_gpu else f"python:{py_ver}-slim"

    specific_instructions = ""

    if ml_type == "streamlit":
        e = context.get("streamlit_entry_file", entry_base)
        specific_instructions = f"""
PROJECT: Streamlit App
FROM {base_image}
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN pip install --no-cache-dir streamlit
COPY . .
EXPOSE 8501
CMD ["sh", "-c", "streamlit run {e} --server.port=${{PORT:-8501}} --server.address=0.0.0.0 --server.headless=true"]
ML libs: {ml_libs}
"""
    elif ml_type == "gradio":
        e = context.get("gradio_entry_file", entry_base)
        specific_instructions = f"""
PROJECT: Gradio App
FROM {base_image}
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN pip install --no-cache-dir gradio
COPY . .
EXPOSE 7860
CMD ["sh", "-c", "python {e}"]
ML libs: {ml_libs}
"""
    elif ml_type == "jupyter":
        specific_instructions = f"""
PROJECT: Jupyter Notebook Server
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt* ./
RUN pip install --no-cache-dir jupyter notebook jupyterlab
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=${{PORT:-8888}} --no-browser --allow-root --NotebookApp.token=''"]
"""
    elif ml_type in ("fastapi_ml", "flask_ml"):
        e     = context.get("fastapi_entry_file") or context.get("flask_entry_file", entry_base)
        mod   = os.path.basename(e).replace(".py", "")
        is_fa = ml_type == "fastapi_ml"
        cmd   = (f"uvicorn {mod}:{app_var} --host 0.0.0.0 --port ${{PORT:-8000}}"
                 if is_fa else "flask run --host=0.0.0.0 --port=${PORT:-5000}")
        specific_instructions = f"""
PROJECT: {'FastAPI' if is_fa else 'Flask'} ML API
FROM {base_image}
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "{cmd}"]
ML libs: {ml_libs}
"""
    elif ml_type == "ml_script":
        e = context.get("ml_script_entry", entry_base)
        specific_instructions = f"""
PROJECT: Plain ML Script
FROM {base_image}
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
CMD ["python", "{e}"]
ML libs: {ml_libs}
"""
    elif framework == "fastapi":
        e   = context.get("fastapi_entry_file", entry_base)
        mod = os.path.basename(e).replace(".py", "")
        specific_instructions = f"""
PROJECT: FastAPI
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
CMD ["sh", "-c", "uvicorn {mod}:{app_var} --host 0.0.0.0 --port ${{PORT:-8000}}"]
"""
    elif framework == "flask":
        e = context.get("flask_entry_file", entry_base)
        specific_instructions = f"""
PROJECT: Flask
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP={e}
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=${{PORT:-5000}}"]
"""
    elif framework == "django":
        specific_instructions = f"""
PROJECT: Django
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:${{PORT:-8000}}"]
"""
    elif fe_type in ("react", "vue", "angular", "vite", "svelte"):
        specific_instructions = f"""
PROJECT: {fe_type.title()} SPA
Multi-stage: node:{node_ver}-alpine builder + nginx:alpine
RUN npm ci && npm run build
COPY build output to /usr/share/nginx/html
CMD ["nginx", "-g", "daemon off;"]
Use try_files $uri /index.html for SPA routing
"""
    elif fe_type == "nextjs":
        specific_instructions = f"""
PROJECT: Next.js
Multi-stage: node:{node_ver}-alpine builder + runner
RUN npm ci && npm run build
CMD ["sh", "-c", "npm start -- --port ${{PORT:-3000}}"]
"""
    elif fe_type == "static_html":
        specific_instructions = """
PROJECT: Static HTML
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
"""
    elif lang == "java":
        specific_instructions = """
PROJECT: Java
Multi-stage: maven:3.9-eclipse-temurin-17 + eclipse-temurin:17-jre-slim
RUN mvn package -DskipTests
CMD ["java", "-jar", "target/app.jar"]
"""
    elif lang == "go":
        specific_instructions = """
PROJECT: Go
Multi-stage: golang:1.21-alpine + alpine:3.18
RUN go build -o main .
CMD ["./main"]
"""
    elif lang == "ruby":
        specific_instructions = f"""
PROJECT: Ruby {'Rails' if framework == 'rails' else ''}
FROM ruby:3.2-slim
RUN bundle install
CMD rails server or ruby {entry_base} with $PORT
"""
    else:
        cmd  = context.get("llm_detected_cmd", f"python {entry_base}")
        port = context.get("llm_detected_port", "8000")
        specific_instructions = f"""
PROJECT: Custom ({framework or 'unknown'}) — Language: {lang}
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
EXPOSE {port}
CMD ["sh", "-c", "{cmd}"]
NOTE: Use ${{PORT:-{port}}} pattern. Entry file is {entry_base}.
"""
        print(f"[Agent] 🤖 Using cmd: {cmd} on port {port}")

    prompt = f"""You are a Docker expert. Generate a production-ready Dockerfile.

{context_text}

INSTRUCTIONS:
{specific_instructions}

CRITICAL RULES:
1. NEVER hardcode ports — always use ${{PORT:-DEFAULT}}
2. Python: always set ENV PYTHONUNBUFFERED=1 and PYTHONDONTWRITEBYTECODE=1
3. Install ALL dependencies BEFORE COPY . . (layer caching)
4. The entry file basename is: {entry_base} — CMD must reference this exact filename
5. WORKDIR must be /app — all files are copied there via COPY . .
6. Streamlit: --server.address=0.0.0.0 --server.headless=true --server.port=${{PORT:-8501}}
7. Gradio: ENV GRADIO_SERVER_NAME=0.0.0.0
8. XGBoost/LightGBM: apt-get install -y libgomp1
9. OpenCV: apt-get install -y libgl1-mesa-glx libglib2.0-0
10. HuggingFace: use python:3.11-slim not alpine
11. No requirements.txt: scan imports from entry file and RUN pip install each one
12. Output ONLY the raw Dockerfile — no markdown, no backticks, no explanation
"""

    client   = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Output ONLY raw Dockerfile. No markdown. No backticks. Entry file is {entry_base}. Always use ${{PORT:-DEFAULT}}."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
    )

    dockerfile_content = response.choices[0].message.content.strip()

    if dockerfile_content.startswith("```"):
        lines = [l for l in dockerfile_content.splitlines() if not l.strip().startswith("```")]
        dockerfile_content = "\n".join(lines).strip()

    for port in ["8000", "8080", "5000", "3000", "8501", "7860", "8888"]:
        if f"--port {port}" in dockerfile_content and "${PORT" not in dockerfile_content:
            dockerfile_content = dockerfile_content.replace(
                f"--port {port}", f"--port ${{PORT:-{port}}}")

    if lang == "python" and "PYTHONUNBUFFERED" not in dockerfile_content:
        dockerfile_content = dockerfile_content.replace(
            "WORKDIR /app",
            "WORKDIR /app\n\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1")

    if ml_type == "streamlit":
        if "--server.headless" not in dockerfile_content:
            dockerfile_content = dockerfile_content.replace(
                "streamlit run", "streamlit run --server.headless=true")
        if "--server.address" not in dockerfile_content:
            dockerfile_content = dockerfile_content.replace(
                "streamlit run", "streamlit run --server.address=0.0.0.0")

    path = os.path.join(folder, "Dockerfile")
    with open(path, "w", encoding="utf-8") as f:
        f.write(dockerfile_content)

    print(f"\n[Agent] ── Dockerfile ({lang}/{framework or ml_type}) ──")
    print(dockerfile_content)
    print("[Agent] ──────────────────────────────────────────────────\n")
    return dockerfile_content, context


# ══════════════════════════════════════════════════════════════════════════════
# DOCKER TESTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# def get_test_port(ml_type, framework):
#     port_map = {
#         "streamlit": "8501", "gradio": "7860", "jupyter": "8888",
#         "fastapi_ml": "8000", "flask_ml": "5000", "ml_script": None,
#         "fastapi": "8000", "flask": "5000", "django": "8000",
#         "nextjs": "3000", "nuxt": "3000", "react": "80", "vue": "80",
#         "angular": "80", "vite": "80", "svelte": "80",
#         "static_html": "80", "express": "3000", "fastify": "3000",
#     }
#     key = ml_type if ml_type and ml_type != "unknown" else framework
#     return port_map.get(key, "8000")
def get_test_port(ml_type, framework, folder=None):
    # ── First: try to read port from the actual Dockerfile ──────────────────
    if folder:
        dockerfile_path = os.path.join(folder, "Dockerfile")
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.upper().startswith("EXPOSE"):
                        parts = line.split()
                        if len(parts) >= 2:
                            port = parts[1].strip()
                            # Handle ${PORT:-8501} pattern
                            if "${PORT" in port:
                                import re
                                match = re.search(r'\$\{PORT:-(\d+)\}', port)
                                if match:
                                    return match.group(1)
                            elif port.isdigit():
                                return port

    # ── Fallback: use the type-based map ────────────────────────────────────
    port_map = {
        "streamlit": "8501", "gradio": "7860", "jupyter": "8888",
        "fastapi_ml": "8000", "flask_ml": "5000", "ml_script": None,
        "fastapi": "8000", "flask": "5000", "django": "8000",
        "nextjs": "3000", "nuxt": "3000", "react": "80", "vue": "80",
        "angular": "80", "vite": "80", "svelte": "80",
        "static_html": "80", "static": "80", "none": "80",
        "express": "3000", "fastify": "3000",
    }
    key = ml_type if ml_type and ml_type != "unknown" else framework
    if key in ("unknown", "none", "", None):
        key = "static_html"
    return port_map.get(key, "8000")

def detect_port_from_dockerfile(folder, fallback="8000"):
    """Read EXPOSE port from Dockerfile dynamically — works for any project."""
    import re
    dockerfile_path = os.path.join(folder, "Dockerfile")
    if not os.path.exists(dockerfile_path):
        return fallback
    with open(dockerfile_path, "r") as f:
        content = f.read()
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("EXPOSE"):
            parts = line.split()
            if len(parts) >= 2:
                port = parts[1].strip()
                # Handle ${PORT:-8501} pattern
                match = re.search(r'\$\{PORT:-(\d+)\}', port)
                if match:
                    return match.group(1)
                elif port.isdigit():
                    return port
    return fallback

def get_startup_wait(ml_type, framework):
    wait_map = {
        "streamlit": 15, "gradio": 15, "jupyter": 15,
        "fastapi_ml": 10, "flask_ml": 10, "huggingface": 30,
        "fastapi": 8, "flask": 8, "django": 8,
        "nextjs": 15, "react": 5, "vue": 5, "angular": 5,
    }
    key = ml_type if ml_type and ml_type != "unknown" else framework
    return wait_map.get(key, 10)

def get_container_logs(container_name):
    result = subprocess.run(
        ["docker", "logs", "--tail", "100", container_name],
        capture_output=True, text=True
    )
    return result.stdout + result.stderr

def cleanup_test_container(container_name, image_tag):
    print(f"[Test] Cleaning up test container and image...")
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    subprocess.run(["docker", "rmi", "-f", image_tag],     capture_output=True)
    print(f"[Test] Cleanup done")

def fix_dockerfile_with_llm(dockerfile_path, error_output, error_type, context, openai_api_key):
    with open(dockerfile_path, "r") as f:
        current_dockerfile = f.read()

    error_descriptions = {
        "build":        "The Docker image failed to BUILD with this error",
        "runtime":      "The Docker container failed to START with this error",
        "runtime_exit": "The Docker container STARTED but then EXITED immediately with this error",
        "no_response":  "The Docker container is running but the app is NOT RESPONDING to HTTP requests",
    }
    error_desc = error_descriptions.get(error_type, "There was an error")

    prompt = f"""You are a Docker expert. {error_desc}:

ERROR OUTPUT:
{error_output[-3000:]}

CURRENT DOCKERFILE:
{current_dockerfile}

PROJECT INFO:
- Language: {context.get("detected_language","unknown")}
- Framework: {context.get("detected_framework","unknown")}
- ML type: {context.get("ml_type","unknown")}
- ML libraries: {context.get("ml_frameworks",[])}

TASK: Fix the Dockerfile to resolve this error.

COMMON FIXES BY ERROR TYPE:
- ModuleNotFoundError: add the missing pip install to requirements or RUN pip install
- libgomp not found: add RUN apt-get install -y libgomp1
- libGL not found (opencv): add RUN apt-get install -y libgl1-mesa-glx libglib2.0-0
- Port not responding: check CMD uses correct port and 0.0.0.0 binding
- Streamlit not responding: ensure --server.address=0.0.0.0 --server.headless=true --server.port=${{PORT:-8501}}
- Gradio not responding: ensure ENV GRADIO_SERVER_NAME=0.0.0.0
- Container exits immediately: check CMD is correct and app file exists
- Permission denied: add RUN chmod +x or fix file paths
- File not found in COPY: check file actually exists in repo

Output ONLY the fixed raw Dockerfile. No markdown, no backticks, no explanation.
"""
    client   = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Output ONLY the fixed raw Dockerfile. No markdown. No backticks."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
    )

    fixed_content = response.choices[0].message.content.strip()
    if fixed_content.startswith("```"):
        lines = [l for l in fixed_content.splitlines() if not l.strip().startswith("```")]
        fixed_content = "\n".join(lines).strip()

    if not fixed_content.startswith("FROM"):
        print("[Test] ⚠️  LLM response doesn't look like a Dockerfile, skipping fix")
        return False

    with open(dockerfile_path, "w") as f:
        f.write(fixed_content)

    print(f"[Test] ✅ Dockerfile updated by GPT-4o fix")
    print(f"[Test] Fixed Dockerfile:\n{fixed_content}\n")
    return True

def test_docker_image(folder, app_name, context, openai_api_key, max_retries=3):
    ml_type        = context.get("ml_type", "unknown")
    framework      = context.get("detected_framework", "unknown")
    lang           = context.get("detected_language", "unknown")
    test_port      = detect_port_from_dockerfile(folder)
    image_tag      = f"{app_name}-test:latest"
    container_name = f"{app_name}-test-container"

    # ── Dynamically determine if this project has a web server ────────────
    NO_SERVER_TYPES = {
        "ml_script", "jupyter_script", "train", "batch", "etl", "cron", "worker"
    }
    WEB_SERVER_TYPES = {
        "streamlit", "gradio", "fastapi", "fastapi_ml", "flask", "flask_ml",
        "django", "nextjs", "react", "vue", "angular", "svelte", "vite",
        "express", "fastify", "nuxt", "static_html", "static", "jupyter"
    }

    # Check 1: known no-server ml_type
    is_no_server = ml_type in NO_SERVER_TYPES

    # Check 2: no EXPOSE in Dockerfile and no known web framework
    if not is_no_server:
        has_expose   = test_port is not None and test_port != "8000"  # "8000" is just our fallback default
        has_web_type = ml_type in WEB_SERVER_TYPES or framework in WEB_SERVER_TYPES
        # If Dockerfile has no EXPOSE line AND no recognized web framework → treat as script
        if not has_expose and not has_web_type:
            is_no_server = True

    # Check 3: read CMD from Dockerfile — if it's just "python script.py" with no server flags → script
    if not is_no_server:
        dockerfile_path_check = os.path.join(folder, "Dockerfile")
        if os.path.exists(dockerfile_path_check):
            with open(dockerfile_path_check, "r") as f:
                dockerfile_content = f.read().lower()
            server_keywords = [
                "uvicorn", "gunicorn", "flask run", "streamlit run",
                "gradio", "nginx", "node ", "npm start", "serve",
                "django", "waitress", "hypercorn", "daphne",
                "--host", "--port", "0.0.0.0"
            ]
            has_server_cmd = any(kw in dockerfile_content for kw in server_keywords)
            if not has_server_cmd:
                is_no_server = True

    if is_no_server:
        print(f"\n[Test] ══════════════════════════════════════════════════")
        print(f"[Test] Starting Docker image test for: {app_name}")
        print(f"[Test] Project type: {lang}/{framework or ml_type}")
        print(f"[Test] ℹ️  No web server detected — build-only test")
        print(f"[Test] ══════════════════════════════════════════════════\n")
        build_result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=folder, capture_output=True, text=True,
        )
        if build_result.returncode == 0:
            print("[Test] ✅✅✅ BUILD PASSED (no-server project) ✅✅✅")
            cleanup_test_container(container_name, image_tag)
            return True
        else:
            print(f"[Test] ❌ Build failed:\n{build_result.stderr[-2000:]}")
            return False
    # ── END dynamic no-server check ────────────────────────────────────────

    print(f"\n[Test] ══════════════════════════════════════════════════")
    print(f"[Test] Starting Docker image test for: {app_name}")
    print(f"[Test] Project type: {lang}/{framework or ml_type}")
    print(f"[Test] Test port: {test_port}")
    print(f"[Test] ══════════════════════════════════════════════════\n")

    dockerfile_path = os.path.join(folder, "Dockerfile")

    for attempt in range(1, max_retries + 1):
        print(f"\n[Test] ── Attempt {attempt}/{max_retries} ──────────────────────")
        print(f"[Test] Building image: {image_tag}")
        build_result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=folder, capture_output=True, text=True,
        )

        if build_result.returncode != 0:
            print(f"[Test] ❌ Build FAILED on attempt {attempt}")
            print(f"[Test] Build error:\n{build_result.stderr[-3000:]}")
            if attempt < max_retries:
                print(f"[Test] 🔧 Asking GPT-4o to fix the Dockerfile...")
                fixed = fix_dockerfile_with_llm(
                    dockerfile_path, error_output=build_result.stderr,
                    error_type="build", context=context, openai_api_key=openai_api_key,
                )
                if fixed:
                    print(f"[Test] ✅ Dockerfile updated, retrying build...")
                    continue
                else:
                    print(f"[Test] ❌ Could not auto-fix Dockerfile")
                    break
            else:
                print(f"[Test] ❌ All {max_retries} build attempts failed")
                return False

        print(f"[Test] ✅ Image built successfully: {image_tag}")
        test_port = detect_port_from_dockerfile(folder)
        print(f"[Test] Using port: {test_port}")

        if test_port is None:
            print(f"[Test] ℹ️  No port detected — skipping runtime test")
            print(f"[Test] ✅ Build passed — skipping runtime test")
            cleanup_test_container(container_name, image_tag)
            return True

        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        print(f"[Test] Starting container on port {test_port}...")
        run_result = subprocess.run(
            ["docker", "run", "-d", "--name", container_name,
             "-p", f"{test_port}:{test_port}", "-e", f"PORT={test_port}", image_tag],
            capture_output=True, text=True,
        )

        if run_result.returncode != 0:
            print(f"[Test] ❌ Container failed to start")
            logs = get_container_logs(container_name)
            print(f"[Test] Container logs:\n{logs}")
            if attempt < max_retries:
                fix_dockerfile_with_llm(dockerfile_path, error_output=logs or run_result.stderr,
                                        error_type="runtime", context=context, openai_api_key=openai_api_key)
                cleanup_test_container(container_name, image_tag)
                continue
            else:
                cleanup_test_container(container_name, image_tag)
                return False

        startup_wait = get_startup_wait(ml_type, framework)
        print(f"[Test] Waiting {startup_wait}s for app to start...")
        time.sleep(startup_wait)

        if container_name not in subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True, text=True
        ).stdout:
            print(f"[Test] ❌ Container exited unexpectedly")
            logs = get_container_logs(container_name)
            print(f"[Test] Container logs:\n{logs}")
            if attempt < max_retries:
                fix_dockerfile_with_llm(dockerfile_path, error_output=logs,
                                        error_type="runtime_exit", context=context, openai_api_key=openai_api_key)
                cleanup_test_container(container_name, image_tag)
                continue
            else:
                cleanup_test_container(container_name, image_tag)
                return False

        print(f"[Test] Checking if app responds on http://localhost:{test_port} ...")
        health_ok = False
        for check_attempt in range(5):
            try:
                import urllib.request
                req = urllib.request.urlopen(f"http://localhost:{test_port}", timeout=10)
                print(f"[Test] ✅ HTTP {req.getcode()} — app is responding!")
                health_ok = True
                break
            except Exception as e:
                print(f"[Test] HTTP check {check_attempt+1}/5 failed: {e}")
                time.sleep(5)

        if health_ok:
            print(f"\n[Test] ✅✅✅ DOCKER TEST PASSED ✅✅✅")
            cleanup_test_container(container_name, image_tag)
            return True
        else:
            logs = get_container_logs(container_name)
            print(f"[Test] ❌ App not responding. Logs:\n{logs}")
            if attempt < max_retries:
                fix_dockerfile_with_llm(dockerfile_path, error_output=logs,
                                        error_type="no_response", context=context, openai_api_key=openai_api_key)
                cleanup_test_container(container_name, image_tag)
                continue
            else:
                cleanup_test_container(container_name, image_tag)
                return False

    return False


# def test_docker_image(folder, app_name, context, openai_api_key, max_retries=3):
#     ml_type        = context.get("ml_type", "unknown")
#     framework      = context.get("detected_framework", "unknown")
#     lang           = context.get("detected_language", "unknown")
#     test_port      = detect_port_from_dockerfile(folder)
#     image_tag      = f"{app_name}-test:latest"
#     container_name = f"{app_name}-test-container"

#     print(f"\n[Test] ══════════════════════════════════════════════════")
#     print(f"[Test] Starting Docker image test for: {app_name}")
#     print(f"[Test] Project type: {lang}/{framework or ml_type}")
#     print(f"[Test] Test port: {test_port}")
#     print(f"[Test] ══════════════════════════════════════════════════\n")

#     dockerfile_path = os.path.join(folder, "Dockerfile")

#     for attempt in range(1, max_retries + 1):
#         print(f"\n[Test] ── Attempt {attempt}/{max_retries} ──────────────────────")
#         print(f"[Test] Building image: {image_tag}")
#         build_result = subprocess.run(
#             ["docker", "build", "-t", image_tag, "."],
#             cwd=folder, capture_output=True, text=True,
#         )

#         if build_result.returncode != 0:
#             print(f"[Test] ❌ Build FAILED on attempt {attempt}")
#             print(f"[Test] Build error:\n{build_result.stderr[-3000:]}")
#             if attempt < max_retries:
#                 print(f"[Test] 🔧 Asking GPT-4o to fix the Dockerfile...")
#                 fixed = fix_dockerfile_with_llm(
#                     dockerfile_path, error_output=build_result.stderr,
#                     error_type="build", context=context, openai_api_key=openai_api_key,
#                 )
#                 if fixed:
#                     print(f"[Test] ✅ Dockerfile updated, retrying build...")
#                     continue
#                 else:
#                     print(f"[Test] ❌ Could not auto-fix Dockerfile")
#                     break
#             else:
#                 print(f"[Test] ❌ All {max_retries} build attempts failed")
#                 return False

#         print(f"[Test] ✅ Image built successfully: {image_tag}")
#         # Re-read port after each build in case LLM fixed the Dockerfile
#         test_port = detect_port_from_dockerfile(folder)
#         print(f"[Test] Using port: {test_port}")

#         if test_port is None:
#             print(f"[Test] ℹ️  ML script project — no web server to test")
#             print(f"[Test] ✅ Build passed — skipping runtime test")
#             cleanup_test_container(container_name, image_tag)
#             return True

#         subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
#         print(f"[Test] Starting container on port {test_port}...")
#         run_result = subprocess.run(
#             ["docker", "run", "-d", "--name", container_name,
#              "-p", f"{test_port}:{test_port}", "-e", f"PORT={test_port}", image_tag],
#             capture_output=True, text=True,
#         )

#         if run_result.returncode != 0:
#             print(f"[Test] ❌ Container failed to start")
#             logs = get_container_logs(container_name)
#             print(f"[Test] Container logs:\n{logs}")
#             if attempt < max_retries:
#                 fix_dockerfile_with_llm(dockerfile_path, error_output=logs or run_result.stderr,
#                                         error_type="runtime", context=context, openai_api_key=openai_api_key)
#                 cleanup_test_container(container_name, image_tag)
#                 continue
#             else:
#                 cleanup_test_container(container_name, image_tag)
#                 return False

#         startup_wait = get_startup_wait(ml_type, framework)
#         print(f"[Test] Waiting {startup_wait}s for app to start...")
#         time.sleep(startup_wait)

#         if container_name not in subprocess.run(
#             ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
#             capture_output=True, text=True
#         ).stdout:
#             print(f"[Test] ❌ Container exited unexpectedly")
#             logs = get_container_logs(container_name)
#             print(f"[Test] Container logs:\n{logs}")
#             if attempt < max_retries:
#                 fix_dockerfile_with_llm(dockerfile_path, error_output=logs,
#                                         error_type="runtime_exit", context=context, openai_api_key=openai_api_key)
#                 cleanup_test_container(container_name, image_tag)
#                 continue
#             else:
#                 cleanup_test_container(container_name, image_tag)
#                 return False

#         print(f"[Test] Checking if app responds on http://localhost:{test_port} ...")
#         health_ok = False
#         for check_attempt in range(5):
#             try:
#                 import urllib.request
#                 req = urllib.request.urlopen(f"http://localhost:{test_port}", timeout=10)
#                 print(f"[Test] ✅ HTTP {req.getcode()} — app is responding!")
#                 health_ok = True
#                 break
#             except Exception as e:
#                 print(f"[Test] HTTP check {check_attempt+1}/5 failed: {e}")
#                 time.sleep(5)

#         if health_ok:
#             print(f"\n[Test] ✅✅✅ DOCKER TEST PASSED ✅✅✅")
#             cleanup_test_container(container_name, image_tag)
#             return True
#         else:
#             logs = get_container_logs(container_name)
#             print(f"[Test] ❌ App not responding. Logs:\n{logs}")
#             if attempt < max_retries:
#                 fix_dockerfile_with_llm(dockerfile_path, error_output=logs,
#                                         error_type="no_response", context=context, openai_api_key=openai_api_key)
#                 cleanup_test_container(container_name, image_tag)
#                 continue
#             else:
#                 cleanup_test_container(container_name, image_tag)
#                 return False

#     return False
def run_project_locally(folder, context, openai_api_key=None, max_retries=1):
    lang      = context.get("detected_language", "unknown")
    framework = context.get("detected_framework", "unknown")
    ml_type   = context.get("ml_type", "unknown")
    entries   = context.get("entry_points_found", [])
    entry     = entries[0] if entries else ""

    # ── Install dependencies ──────────────────────────────────────────
    req_path = os.path.join(folder, "requirements.txt")
    if os.path.exists(req_path):
        print(f"[LocalTest] 📦 Installing requirements...")
        result = subprocess.run(
            ["pip", "install", "-r", "requirements.txt", "--quiet"],
            cwd=folder, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[LocalTest] ⚠️  Some requirements failed to install")

    pkg_path = os.path.join(folder, "package.json")
    if os.path.exists(pkg_path):
        node_check = subprocess.run(["node", "--version"], capture_output=True)
        if node_check.returncode == 0:
            print(f"[LocalTest] 📦 Running npm install...")
            subprocess.run(["npm", "install", "--silent"],
                          cwd=folder, capture_output=True)
        else:
            print(f"[LocalTest] ⚠️  Node.js not found — skipping npm install")

    # ── Determine command and success signals ─────────────────────────
    cmd             = None
    success_signals = []
    is_server       = False

    if lang in ("python", "unknown"):

        if ml_type == "streamlit" or framework == "streamlit":
            # Check if streamlit is installed locally
            check = subprocess.run(
                ["python", "-m", "streamlit", "--version"],
                capture_output=True
            )
            if check.returncode != 0:
                print(f"[LocalTest] ℹ️  Streamlit not installed locally — skipping")
                return True
            e               = context.get("streamlit_entry_file", entry)
            cmd             = ["python", "-m", "streamlit", "run", e,
                               "--server.headless=true", "--server.port=8501",
                               "--server.address=0.0.0.0"]
            success_signals = ["you can now view", "network url", "local url"]
            is_server       = True

        elif ml_type == "gradio" or framework == "gradio":
            check = subprocess.run(
                ["python", "-c", "import gradio"],
                capture_output=True
            )
            if check.returncode != 0:
                print(f"[LocalTest] ℹ️  Gradio not installed locally — skipping")
                return True
            e               = context.get("gradio_entry_file", entry)
            cmd             = ["python", e]
            success_signals = ["running on", "local url", "gradio"]
            is_server       = True

        elif framework == "fastapi" or ml_type == "fastapi_ml":
            check = subprocess.run(
                ["python", "-c", "import uvicorn"],
                capture_output=True
            )
            if check.returncode != 0:
                print(f"[LocalTest] ℹ️  Uvicorn not installed locally — skipping")
                return True
            e       = context.get("fastapi_entry_file", entry)
            mod     = os.path.basename(e).replace(".py", "")
            app_var = context.get("app_variable_name", "app")
            cmd     = ["python", "-m", "uvicorn", f"{mod}:{app_var}",
                       "--host", "0.0.0.0", "--port", "8000"]
            success_signals = ["application startup complete", "uvicorn running"]
            is_server       = True

        elif framework == "flask" or ml_type == "flask_ml":
            check = subprocess.run(
                ["python", "-c", "import flask"],
                capture_output=True
            )
            if check.returncode != 0:
                print(f"[LocalTest] ℹ️  Flask not installed locally — skipping")
                return True
            e               = context.get("flask_entry_file", entry)
            cmd             = ["python", e]
            success_signals = ["running on", "serving flask", "debugger"]
            is_server       = True

        elif framework == "django":
            check = subprocess.run(
                ["python", "-c", "import django"],
                capture_output=True
            )
            if check.returncode != 0:
                print(f"[LocalTest] ℹ️  Django not installed locally — skipping")
                return True
            cmd       = ["python", "manage.py", "check"]
            is_server = False

        elif ml_type == "jupyter":
            # Just validate notebook JSON
            notebooks = context.get("notebooks_found", [])
            if notebooks:
                nb_path = os.path.join(folder, notebooks[0])
                try:
                    with open(nb_path) as f:
                        json.load(f)
                    print(f"[LocalTest] ✅ Notebook {notebooks[0]} is valid JSON")
                    return True
                except Exception as e:
                    print(f"[LocalTest] ❌ Notebook invalid: {e}")
                    return False
            return True

        elif ml_type == "ml_script":
            e   = context.get("ml_script_entry", entry)
            cmd = ["python", "-m", "py_compile", e]
            is_server = False

        elif entry and entry.endswith(".py"):
            cmd       = ["python", "-m", "py_compile", entry]
            is_server = False

        else:
            print(f"[LocalTest] ℹ️  No Python entry point found — skipping")
            return True

    elif lang == "nodejs":
        node_check = subprocess.run(["node", "--version"], capture_output=True)
        if node_check.returncode != 0:
            print(f"[LocalTest] ⚠️  Node.js not found locally — skipping")
            return True

        if framework in ("react", "vue", "vite", "svelte", "angular"):
            cmd             = ["npm", "run", "build"]
            success_signals = ["built in", "compiled successfully", "dist/", "chunks"]
            is_server       = False

        elif framework == "nextjs":
            cmd             = ["npm", "run", "build"]
            success_signals = ["compiled", "ready", "build"]
            is_server       = False

        elif framework in ("express", "fastify"):
            cmd             = ["node", entry or "index.js"]
            success_signals = ["listening", "started", "running", "server"]
            is_server       = True

        else:
            # Try npm start or npm run dev
            if os.path.exists(pkg_path):
                with open(pkg_path) as f:
                    pkg = json.load(f)
                scripts = pkg.get("scripts", {})
                if "start" in scripts:
                    cmd             = ["npm", "start"]
                    success_signals = ["listening", "started", "running"]
                    is_server       = True
                elif "dev" in scripts:
                    cmd             = ["npm", "run", "dev"]
                    success_signals = ["ready", "listening", "started"]
                    is_server       = True
                elif "build" in scripts:
                    cmd             = ["npm", "run", "build"]
                    success_signals = ["built", "compiled"]
                    is_server       = False
                else:
                    print(f"[LocalTest] ℹ️  No npm script found — skipping")
                    return True
            else:
                return True

    elif lang == "go":
        go_check = subprocess.run(["go", "version"], capture_output=True)
        if go_check.returncode != 0:
            print(f"[LocalTest] ⚠️  Go not found locally — skipping")
            return True
        # Download deps first
        subprocess.run(["go", "mod", "download"], cwd=folder, capture_output=True)
        cmd       = ["go", "build", "./..."]
        is_server = False

    elif lang == "java":
        mvn_check = subprocess.run(["mvn", "--version"], capture_output=True)
        if mvn_check.returncode != 0:
            print(f"[LocalTest] ⚠️  Maven not found locally — skipping")
            return True
        cmd       = ["mvn", "compile", "-q"]
        is_server = False

    elif lang == "rust":
        cargo_check = subprocess.run(["cargo", "--version"], capture_output=True)
        if cargo_check.returncode != 0:
            print(f"[LocalTest] ⚠️  Cargo not found locally — skipping")
            return True
        cmd       = ["cargo", "build"]
        is_server = False

    elif lang == "ruby":
        ruby_check = subprocess.run(["ruby", "--version"], capture_output=True)
        if ruby_check.returncode != 0:
            print(f"[LocalTest] ⚠️  Ruby not found locally — skipping")
            return True
        if framework == "rails":
            subprocess.run(["bundle", "install", "--quiet"],
                          cwd=folder, capture_output=True)
            cmd       = ["bundle", "exec", "rails", "runner", "puts 'ok'"]
            is_server = False
        elif entry:
            cmd       = ["ruby", "-c", entry]
            is_server = False
        else:
            print(f"[LocalTest] ℹ️  No Ruby entry point — skipping")
            return True

    elif lang == "php":
        php_check = subprocess.run(["php", "--version"], capture_output=True)
        if php_check.returncode != 0:
            print(f"[LocalTest] ⚠️  PHP not found locally — skipping")
            return True
        e         = entry or "index.php"
        cmd       = ["php", "-l", e]
        is_server = False

    elif lang in ("html", "static"):
        index = os.path.join(folder, "index.html")
        if os.path.exists(index):
            print(f"[LocalTest] ✅ Static site — index.html found")
        else:
            print(f"[LocalTest] ⚠️  Static site — no index.html found")
        return True

    else:
        print(f"[LocalTest] ℹ️  Language '{lang}' — skipping local test")
        return True

    if not cmd:
        print(f"[LocalTest] ℹ️  No run command — skipping")
        return True

    print(f"\n[LocalTest] 🧪 Running: {' '.join(cmd)}")

    # ── Non-server: just run and check return code ────────────────────
    if not is_server:
        result = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[LocalTest] ✅ Project OK locally")
            return True
        else:
            print(f"[LocalTest] ❌ Failed:\n{result.stderr[-500:]}")
            return False

    # ── Server: start process, watch output for success signal ────────
    proc       = subprocess.Popen(
        cmd, cwd=folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    startup_ok = False
    start_time = time.time()
    wait_time  = 30

    while time.time() - start_time < wait_time:
        # Check stderr first (most frameworks log to stderr)
        line = proc.stderr.readline()
        if not line:
            line = proc.stdout.readline()
        if line:
            print(f"[LocalTest]   {line.rstrip()}")
            if any(sig in line.lower() for sig in success_signals):
                startup_ok = True
                break
            # Detect fatal errors early
            if any(kw in line.lower() for kw in [
                "modulenotfounderror", "importerror", "no module named",
                "syntaxerror", "error:", "traceback"
            ]):
                print(f"[LocalTest] ❌ Fatal error detected")
                break
        if proc.poll() is not None:
            break
        time.sleep(0.3)

    proc.kill()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass

    if startup_ok:
        print(f"[LocalTest] ✅ Project runs locally")
        return True
    else:
        print(f"[LocalTest] ⚠️  Could not confirm startup — proceeding anyway")
        return False

# ══════════════════════════════════════════════════════════════════════════════
# GIT OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def push_branch(folder, fork_url, token):
    auth_url = fork_url.replace("https://", f"https://{token}@")
    subprocess.run(["git", "push", auth_url, "ai-docker-setup", "--force"],
                   cwd=folder, check=True)
    print("[Agent] Branch pushed")

def create_pull_request(repo_url, token, fork_owner, default_branch):
    repo      = repo_url.replace("https://github.com/", "").rstrip("/")
    headers   = make_github_headers(token)
    check_url = f"https://api.github.com/repos/{repo}/pulls"

    # head_ref must be defined before any requests
    repo_owner = repo.split("/")[0]
    head_ref   = "ai-docker-setup" if fork_owner == repo_owner else f"{fork_owner}:ai-docker-setup"

    # Check open PRs
    existing = requests.get(check_url, headers=headers,
                            params={"head": head_ref, "state": "open"})
    if existing.status_code == 200 and existing.json():
        url = existing.json()[0]["html_url"]
        print("[Agent] PR already exists (open):", url)
        return url

    # Check closed/merged PRs
    closed = requests.get(check_url, headers=headers,
                          params={"head": head_ref, "state": "closed"})
    if closed.status_code == 200 and closed.json():
        pr = closed.json()[0]
        if pr.get("merged_at"):
            print("[Agent] ℹ️  Branch already merged — skipping PR creation")
            return pr["html_url"]
        else:
            reopen = requests.patch(
                f"https://api.github.com/repos/{repo}/pulls/{pr['number']}",
                headers=headers, json={"state": "open"}
            )
            if reopen.status_code == 200:
                url = reopen.json()["html_url"]
                print(f"[Agent] ♻️  Reopened existing PR: {url}")
                return url

    # Create new PR
    data = {
        "title": "AI Generated Docker Setup (via OpenAI)",
        "head":  head_ref,
        "base":  default_branch,
        "body":  "Auto-generated Dockerfile by AI agent using OpenAI GPT-4o.",
    }
    r        = requests.post(check_url, headers=headers, json=data)
    response = r.json()

    if r.status_code == 201:
        url = response["html_url"]
        print("[Agent] PR created:", url)
        return url
    elif r.status_code == 422:
        all_resp = requests.get(check_url, headers=headers,
                                params={"head": head_ref})
        if all_resp.status_code == 200 and all_resp.json():
            url = all_resp.json()[0]["html_url"]
            print(f"[Agent] Existing PR: {url}")
            return url
        raise RuntimeError(f"PR failed: {response.get('message')}")
    else:
        raise RuntimeError(f"PR failed: {r.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# DEPLOY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def parse_deploy_targets(user_input, openai_api_key):
    client   = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""
Extract deployment platforms from this input: "{user_input}"

SUPPORTED PLATFORMS (only these, nothing else):
- aws
- azure  
- render
- railway

RULES:
- Only return platforms explicitly mentioned
- "aws" means ONLY ["aws"], NOT azure/render/railway
- Return ONLY a JSON array, nothing else
- Examples:
  "aws" → ["aws"]
  "deploy to railway" → ["railway"]
  "render and aws" → ["render", "aws"]
  "azure" → ["azure"]

Return ONLY the JSON array:
"""}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
        raw   = "\n".join(lines).strip()
    try:
        targets = json.loads(raw)
        # Safety check — only allow known platforms
        valid   = {"aws", "azure", "render", "railway"}
        targets = [t for t in targets if t in valid]
        print(f"[Agent] Targets: {targets}")
        return targets
    except Exception:
        return []



# ══════════════════════════════════════════════════════════════════════════════
# PLATFORM DEPLOYERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_free_tier_instance(ec2_client):
    """Dynamically find the free-tier eligible instance type for this region."""
    try:
        resp = ec2_client.describe_instance_types(
            Filters=[{"Name": "free-tier-eligible", "Values": ["true"]}]
        )
        types = [i["InstanceType"] for i in resp.get("InstanceTypes", [])]
        print(f"[AWS] ℹ️  Free tier eligible types in this region: {types}")

        # Preference order
        for preferred in ["t2.micro", "t3.micro", "t4g.micro", "t2.small"]:
            if preferred in types:
                print(f"[AWS] ✅ Using free tier instance: {preferred}")
                return preferred

        # Fallback to first available
        if types:
            print(f"[AWS] ✅ Using free tier instance: {types[0]}")
            return types[0]

    except Exception as e:
        print(f"[AWS] ⚠️  Could not detect free tier instance: {e}")

    # Hard fallback
    print(f"[AWS] ⚠️  Falling back to t3.micro")
    return "t3.micro"

def deploy_to_aws(folder, creds):
    import boto3
    import base64

    app_name   = creds["app_name"]
    region     = creds["region"]
    access_key = creds["access_key"]
    secret_key = creds["secret_key"]
    env_vars   = creds.get("env_vars", {})
    port       = detect_port_from_dockerfile(folder, fallback="8080")

    ec2 = boto3.client(
        "ec2",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    ecr = boto3.client(
        "ecr",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    # ── Step 1: Create ECR repo ──────────────────────────────────────────────
    print(f"[AWS] 🔧 Setting up ECR repository: {app_name}...")
    try:
        repo_resp = ecr.create_repository(repositoryName=app_name)
        repo_uri  = repo_resp["repository"]["repositoryUri"]
        print(f"[AWS] ✅ ECR repo created: {repo_uri}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        repo_resp = ecr.describe_repositories(repositoryNames=[app_name])
        repo_uri  = repo_resp["repositories"][0]["repositoryUri"]
        print(f"[AWS] ℹ️  ECR repo exists: {repo_uri}")

    # ── Step 2: Docker login to ECR ──────────────────────────────────────────
    print(f"[AWS] 🔐 Logging Docker into ECR...")
    token_resp   = ecr.get_authorization_token()
    auth_data    = token_resp["authorizationData"][0]
    auth_token   = base64.b64decode(auth_data["authorizationToken"]).decode()
    ecr_user, ecr_pass = auth_token.split(":", 1)
    registry_url = auth_data["proxyEndpoint"]

    subprocess.run(
        ["docker", "login", "--username", ecr_user, "--password-stdin", registry_url],
        input=ecr_pass.encode(), capture_output=True, check=True
    )
    print(f"[AWS] ✅ Docker logged into ECR")

    # ── Step 3: Build & push image ───────────────────────────────────────────
    image_tag = f"{repo_uri}:latest"
    local_tag = f"{app_name}:latest"

    print(f"[AWS] 🔨 Building image...")
    subprocess.run(["docker", "build", "-t", local_tag, "."],
                   cwd=folder, check=True)
    subprocess.run(["docker", "tag", local_tag, image_tag], check=True)

    print(f"[AWS] 📤 Pushing to ECR...")
    subprocess.run(["docker", "push", image_tag], check=True)
    print(f"[AWS] ✅ Image pushed: {image_tag}")

    # ── Step 4: Create security group ────────────────────────────────────────
    print(f"[AWS] 🔧 Setting up security group...")
    sg_name = f"{app_name}-sg"
    try:
        sg_resp = ec2.create_security_group(
            GroupName=sg_name,
            Description=f"Security group for {app_name}",
        )
        sg_id = sg_resp["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {"IpProtocol": "tcp", "FromPort": int(port), "ToPort": int(port),
                 "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                 "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            ]
        )
        print(f"[AWS] ✅ Security group created: {sg_id}")
    except ec2.exceptions.ClientError as e:
        if "InvalidGroup.Duplicate" in str(e):
            sgs = ec2.describe_security_groups(GroupNames=[sg_name])
            sg_id = sgs["SecurityGroups"][0]["GroupId"]
            print(f"[AWS] ℹ️  Security group exists: {sg_id}")
        else:
            raise

    # ── Step 5: Build user_data script to run on EC2 at launch ───────────────
    env_exports = "\n".join(
        f'export {k}="{v}"' for k, v in env_vars.items()
    )
    account_id = boto3.client(
        "sts",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ).get_caller_identity()["Account"]

    user_data = f"""#!/bin/bash
yum update -y
yum install -y docker
service docker start
usermod -aG docker ec2-user

# Embed AWS credentials so ECR login works without IAM role
export AWS_ACCESS_KEY_ID={access_key}
export AWS_SECRET_ACCESS_KEY={secret_key}
export AWS_DEFAULT_REGION={region}

# Login to ECR using embedded credentials
aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com

# Set env vars
{env_exports}
export PORT={port}

# Pull and run the image
docker pull {image_tag}
docker run -d --restart always \\
  -p {port}:{port} \\
  -e PORT={port} \\
  {" ".join(f'-e {k}="{v}"' for k, v in env_vars.items())} \\
  --name {app_name} \\
  {image_tag}
"""

    # ── Step 6: Launch free tier EC2 instance ────────────────────────────────
    ami_resp = ec2.describe_images(
        Filters=[
            {"Name": "name",        "Values": ["amzn2-ami-hvm-*-x86_64-gp2"]},
            {"Name": "state",       "Values": ["available"]},
            {"Name": "owner-alias", "Values": ["amazon"]},
        ],
        Owners=["amazon"],
    )
    ami_id = sorted(
        ami_resp["Images"], key=lambda x: x["CreationDate"], reverse=True
    )[0]["ImageId"]
    print(f"[AWS] ℹ️  Using AMI: {ami_id}")

    free_tier_instance = _get_free_tier_instance(ec2)
    print(f"[AWS] 🚀 Launching {free_tier_instance} EC2 instance (free tier)...")

    instance_resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=free_tier_instance,
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[sg_id],
        UserData=user_data,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": app_name}],
        }],
    )

    instance_id = instance_resp["Instances"][0]["InstanceId"]
    print(f"[AWS] ✅ Instance launched: {instance_id}")
    print(f"[AWS] ⏳ Waiting for instance to get public IP (30s)...")
    time.sleep(30)

    # ── Step 7: Get public IP ─────────────────────────────────────────────────
    desc      = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")

    if public_ip:
        url = f"http://{public_ip}:{port}"
        print(f"[AWS] ✅ Instance running at: {url}")
        print(f"[AWS] ⏳ App may take 5-7 minutes to start (install Docker + pull image + run)")
        print(f"[AWS] ℹ️  Instance ID: {instance_id} — stop it from AWS console to avoid charges")
        return url
    else:
        url = f"http://check-aws-console-for-ip:{port}"
        print(f"[AWS] ⚠️  Could not get public IP — check AWS console for instance: {instance_id}")
        return url


def deploy_to_azure(folder, creds):
    from azure.identity import ClientSecretCredential
    from azure.mgmt.containerregistry import ContainerRegistryManagementClient
    from azure.mgmt.appcontainers import ContainerAppsAPIClient

    app_name  = creds["app_name"]
    reg_name  = f"{app_name}registry".replace("-", "")[:50]
    fork_url  = creds.get("fork_url", "")
    repo_path = fork_url.replace("https://github.com/", "").replace(".git", "").rstrip("/")

    cred = ClientSecretCredential(
        tenant_id=creds["tenant_id"],
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
    )

    acr      = ContainerRegistryManagementClient(cred, creds["subscription_id"])
    result   = acr.registries.begin_create(
        creds["resource_group"], reg_name,
        {"location": "eastus", "sku": {"name": "Basic"}, "admin_user_enabled": True}
    ).result()
    login_server = result.login_server
    acr_creds    = acr.registries.list_credentials(creds["resource_group"], reg_name)
    acr_user     = acr_creds.username
    acr_pass     = acr_creds.passwords[0].value
    image_tag    = f"{login_server}/{app_name}:latest"

    env_vars = creds.get("env_vars", {})
    env_list = [{"name": k, "value": v} for k, v in env_vars.items()]

    aca = ContainerAppsAPIClient(cred, creds["subscription_id"])
    res = aca.container_apps.begin_create_or_update(
        creds["resource_group"], app_name,
        {
            "location": "eastus",
            "properties": {
                "configuration": {
                    "ingress": {"external": True, "targetPort": 8080},
                    "registries": [{"server": login_server, "username": acr_user, "passwordSecretRef": "acr-pass"}],
                    "secrets": [{"name": "acr-pass", "value": acr_pass}],
                },
                "template": {
                    "containers": [{"name": app_name, "image": image_tag,
                                    "resources": {"cpu": 0.5, "memory": "1Gi"}, "env": env_list}],
                    "scale": {"minReplicas": 1, "maxReplicas": 3},
                },
            },
        }
    ).result()

    url = f"https://{res.properties.configuration.ingress.fqdn}"
    print(f"[Azure] ✅ {url}")
    return url


def deploy_to_render(fork_url, creds, folder=""):
    app_name = creds["app_name"]
    headers  = {"Authorization": f"Bearer {creds['api_key']}", "Content-Type": "application/json"}

    owner_id = requests.get(
        "https://api.render.com/v1/owners?limit=1", headers=headers
    ).json()[0]["owner"]["id"]

    services = requests.get(
        "https://api.render.com/v1/services?limit=50", headers=headers
    ).json()
    existing = next((s["service"] for s in services
                     if s["service"]["name"] == app_name), None)

    if existing:
        svc_id = existing["id"]
        print(f"[Render] ℹ️  Service exists — updating env vars and redeploying...")
        env_vars = creds.get("env_vars", {})
        if env_vars:
            env_pairs = [{"key": k, "value": v} for k, v in env_vars.items()]
            requests.put(f"https://api.render.com/v1/services/{svc_id}/env-vars",
                         headers=headers, json=env_pairs)
            print(f"[Render] ✅ Updated {len(env_pairs)} env vars")
        requests.post(f"https://api.render.com/v1/services/{svc_id}/deploys",
                      headers=headers, json={"clearCache": "do_not_clear"})
        url = f"https://{existing['serviceDetails']['url']}"
        print(f"[Render] ✅ Redeployed: {url}")
        return url

    env_vars_list = [{"key": "PORT", "value": "10000"}]
    for k, v in creds.get("env_vars", {}).items():
        env_vars_list.append({"key": k, "value": v})
    if env_vars_list:
        print(f"[Render] 📦 Using {len(env_vars_list)-1} env vars")

    resp = requests.post("https://api.render.com/v1/services", headers=headers, json={
        "type":    "web_service",
        "name":    app_name,
        "ownerId": owner_id,
        "repo":    fork_url.replace(".git", ""),
        "branch":  "ai-docker-setup",
        "serviceDetails": {
            "env":    "docker",
            "plan":   "free",
            "region": "oregon",
            "envVars": env_vars_list,
            "pullRequestPreviewsEnabled": False,
        },
    })
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Render failed: {resp.text}")

    svc  = resp.json()["service"]
    url  = f"https://{svc['serviceDetails']['url']}"
    print(f"[Render] ✅ Service created from GitHub branch ai-docker-setup")
    print(f"[Render] ✅ {url}")
    return url


# def deploy_to_railway(folder, creds):
#     app_name       = creds["app_name"]
#     dockerhub_user = creds["dockerhub_user"]
#     dockerhub_pass = creds["dockerhub_pass"]
#     token          = creds["token"]

#     subprocess.run(["docker", "login", "--username", dockerhub_user, "--password-stdin"],
#                    input=dockerhub_pass.encode(), check=True)
#     print("[Railway] ✅ Docker Hub login")

#     image_name = f"{dockerhub_user}/{app_name}:latest"
#     subprocess.run(["docker", "build", "-t", image_name, "."], cwd=folder, check=True)
#     subprocess.run(["docker", "push", image_name], check=True)
#     print(f"[Railway] ✅ Pushed: {image_name}")

#     headers     = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
#     graphql_url = "https://backboard.railway.app/graphql/v2"

#     def gql(query):
#         return requests.post(graphql_url, headers=headers, json={"query": query}).json()

#     ws = gql("query { me { workspaces { id name } } }")
#     if "errors" in ws:
#         raise RuntimeError(f"Workspace error: {ws['errors']}")
#     workspace_id = ws["data"]["me"]["workspaces"][0]["id"]

#     proj = gql("""mutation { projectCreate(input:{name:"%s",workspaceId:"%s"}){
#         id environments{edges{node{id name}}}}}""" % (app_name, workspace_id))
#     if "errors" in proj:
#         raise RuntimeError(f"Project error: {proj['errors']}")
#     project_id     = proj["data"]["projectCreate"]["id"]
#     environment_id = proj["data"]["projectCreate"]["environments"]["edges"][0]["node"]["id"]

#     svc = gql("""mutation { serviceCreate(input:{projectId:"%s",name:"%s",
#         source:{image:"%s"}}){id name}}""" % (project_id, app_name, image_name))
#     if "errors" in svc:
#         raise RuntimeError(f"Service error: {svc['errors']}")
#     service_id = svc["data"]["serviceCreate"]["id"]

#     gql("""mutation { variableUpsert(input:{projectId:"%s",environmentId:"%s",
#         serviceId:"%s",name:"PORT",value:"8000"})}""" % (project_id, environment_id, service_id))
#     print("[Railway] ✅ PORT=8000 set")

#     # Use env_vars collected before deployment
#     env_vars = creds.get("env_vars", {})
#     if env_vars:
#         print("[Railway] 📦 Pushing env vars to Railway...")
#         for key, value in env_vars.items():
#             resp = gql("""mutation { variableUpsert(input:{projectId:"%s",environmentId:"%s",
#                 serviceId:"%s",name:"%s",value:"%s"})}""" % (
#                 project_id, environment_id, service_id, key, value))
#             if "errors" in resp:
#                 print(f"[Railway] ⚠️  Failed to set {key}: {resp['errors']}")
#             else:
#                 print(f"[Railway] ✅ Set: {key}")
#         print("[Railway] ✅ All env vars pushed to Railway")
#     else:
#         print("[Railway] ℹ️  No env vars provided — skipping")

#     time.sleep(20)
#     domain_q    = """mutation { serviceDomainCreate(input:{serviceId:"%s",environmentId:"%s"}){domain}}""" % (service_id, environment_id)
#     domain_resp = gql(domain_q)
#     if "errors" in domain_resp or not domain_resp.get("data", {}).get("serviceDomainCreate"):
#         print("[Railway] Retrying domain creation in 15 seconds...")
#         time.sleep(15)
#         domain_resp = gql(domain_q)

#     try:
#         url = f"https://{domain_resp['data']['serviceDomainCreate']['domain']}"
#     except Exception:
#         url = f"https://railway.app/project/{project_id}"
#         print("[Railway] ⚠️  Get domain manually from Railway dashboard")

#     print(f"[Railway] ✅ {url}")
#     return url

def deploy_to_railway(folder, creds):
    app_name       = creds["app_name"]
    dockerhub_user = creds["dockerhub_user"]
    dockerhub_pass = creds["dockerhub_pass"]
    token          = creds["token"]

    subprocess.run(["docker", "login", "--username", dockerhub_user, "--password-stdin"],
                   input=dockerhub_pass.encode(), check=True)
    print("[Railway] ✅ Docker Hub login")

    image_name = f"{dockerhub_user}/{app_name}:latest"
    subprocess.run(["docker", "build", "-t", image_name, "."], cwd=folder, check=True)
    subprocess.run(["docker", "push", image_name], check=True)
    print(f"[Railway] ✅ Pushed: {image_name}")

    headers     = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    graphql_url = "https://backboard.railway.app/graphql/v2"

    def gql(query):
        return requests.post(graphql_url, headers=headers, json={"query": query}).json()

    ws = gql("query { me { workspaces { id name } } }")
    if "errors" in ws:
        raise RuntimeError(f"Workspace error: {ws['errors']}")
    workspace_id = ws["data"]["me"]["workspaces"][0]["id"]

    proj = gql("""mutation { projectCreate(input:{name:"%s",workspaceId:"%s"}){
        id environments{edges{node{id name}}}}}""" % (app_name, workspace_id))
    if "errors" in proj:
        raise RuntimeError(f"Project error: {proj['errors']}")
    project_id     = proj["data"]["projectCreate"]["id"]
    environment_id = proj["data"]["projectCreate"]["environments"]["edges"][0]["node"]["id"]

    svc = gql("""mutation { serviceCreate(input:{projectId:"%s",name:"%s",
        source:{image:"%s"}}){id name}}""" % (project_id, app_name, image_name))
    if "errors" in svc:
        raise RuntimeError(f"Service error: {svc['errors']}")
    service_id = svc["data"]["serviceCreate"]["id"]

    # Dynamically detect port from Dockerfile
    railway_port = detect_port_from_dockerfile(folder, fallback="8000")
    gql("""mutation { variableUpsert(input:{projectId:"%s",environmentId:"%s",
        serviceId:"%s",name:"PORT",value:"%s"})}""" % (
        project_id, environment_id, service_id, railway_port))
    print(f"[Railway] ✅ PORT={railway_port} set")

    # Use env_vars collected before deployment
    env_vars = creds.get("env_vars", {})
    if env_vars:
        print("[Railway] 📦 Pushing env vars to Railway...")
        for key, value in env_vars.items():
            resp = gql("""mutation { variableUpsert(input:{projectId:"%s",environmentId:"%s",
                serviceId:"%s",name:"%s",value:"%s"})}""" % (
                project_id, environment_id, service_id, key, value))
            if "errors" in resp:
                print(f"[Railway] ⚠️  Failed to set {key}: {resp['errors']}")
            else:
                print(f"[Railway] ✅ Set: {key}")
        print("[Railway] ✅ All env vars pushed to Railway")
    else:
        print("[Railway] ℹ️  No env vars provided — skipping")

    time.sleep(20)
    domain_q    = """mutation { serviceDomainCreate(input:{serviceId:"%s",environmentId:"%s"}){domain}}""" % (service_id, environment_id)
    domain_resp = gql(domain_q)
    if "errors" in domain_resp or not domain_resp.get("data", {}).get("serviceDomainCreate"):
        print("[Railway] Retrying domain creation in 15 seconds...")
        time.sleep(15)
        domain_resp = gql(domain_q)

    try:
        url = f"https://{domain_resp['data']['serviceDomainCreate']['domain']}"
    except Exception:
        url = f"https://railway.app/project/{project_id}"
        print("[Railway] ⚠️  Get domain manually from Railway dashboard")

    print(f"[Railway] ✅ {url}")
    return url
def deploy_to_platforms(targets, folder, fork_url, creds):
    results = {}
    for platform in targets:
        print(f"\n{'='*50}\n[Agent] Deploying: {platform.upper()}\n{'='*50}")
        try:
            if platform == "aws":
                results["aws"]     = deploy_to_aws(folder, creds["aws"])
            elif platform == "azure":
                results["azure"]   = deploy_to_azure(folder, creds["azure"])
            elif platform == "render":
                results["render"]  = deploy_to_render(fork_url, creds["render"], folder=folder)
            elif platform == "railway":
                results["railway"] = deploy_to_railway(folder, creds["railway"])
        except Exception as e:
            print(f"[Agent] ❌ {platform}: {e}")
            results[platform] = f"FAILED: {e}"

    print(f"\n{'='*50}\n[Agent] 🚀 SUMMARY\n{'='*50}")
    for p, url in results.items():
        icon = "✅" if not str(url).startswith("FAILED") else "❌"
        print(f"  {icon} {p.upper():<10} -> {url}")
    print(f"{'='*50}\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CHECK MODE
# ══════════════════════════════════════════════════════════════════════════════

def check_mode(folder):
    print(f"\n[Agent] ── Check Mode ────────────────────────────────────")
    print(f"[Agent] Scanning: {os.path.abspath(folder)}\n")

    context = deep_scan_repo(folder)

    print(f"\n[Agent] ── Detection Results ─────────────────────────────")
    print(f"  Language:    {context['detected_language']}")
    print(f"  Framework:   {context['detected_framework']}")
    print(f"  ML type:     {context['ml_type']}")
    print(f"  ML libs:     {context['ml_frameworks']}")
    print(f"  Entry pts:   {context['entry_points_found']}")
    print(f"  GPU:         {context['uses_gpu']}")
    print(f"  Python ver:  {context['python_version']}")

    env_path = os.path.join(folder, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            keys = [l.split("=")[0].strip() for l in f
                    if l.strip() and not l.startswith("#") and "=" in l]
        print(f"  .env keys:   {keys} ✅")
    else:
        print(f"  .env:        ⚠️  NOT FOUND — add if app needs API keys")

    all_files = [f for f in os.listdir(folder) if f != ".git"]
    print(f"  Root files:  {all_files}")

    print(f"\n[Agent] ── Dockerfile preview ────────────────────────────")
    ml_type   = context.get("ml_type", "unknown")
    framework = context.get("detected_framework", "unknown")
    entries   = context.get("entry_points_found", [])
    entry     = entries[0] if entries else "unknown"
    py_ver    = context.get("python_version", "3.11")

    if ml_type == "streamlit":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   streamlit run {entry} --server.port=${{PORT:-8501}} --server.address=0.0.0.0")
    elif ml_type == "gradio":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   python {entry}")
    elif ml_type == "fastapi_ml" or framework == "fastapi":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   uvicorn {entry.replace('.py','')}:app --host 0.0.0.0 --port ${{PORT:-8000}}")
    elif ml_type == "flask_ml" or framework == "flask":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   flask run --host=0.0.0.0 --port=${{PORT:-5000}}")
    elif framework == "django":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   python manage.py runserver 0.0.0.0:${{PORT:-8000}}")
    elif framework in ("react", "vue", "angular", "svelte", "vite"):
        print(f"  Base:  node:18-alpine + nginx:alpine (multi-stage)")
        print(f"  CMD:   nginx -g 'daemon off;'")
    elif framework == "nextjs":
        print(f"  Base:  node:18-alpine (multi-stage)")
        print(f"  CMD:   npm start -- --port ${{PORT:-3000}}")
    elif context.get("detected_language") == "go":
        print(f"  Base:  golang:1.21-alpine + alpine:3.18 (multi-stage)")
        print(f"  CMD:   ./main")
    else:
        print(f"  Type:  {framework or ml_type} — GPT-4o will generate Dockerfile")

    print(f"\n[Agent] ── What to do ────────────────────────────────────")
    print(f"  If detection looks wrong:")
    print(f"    • Edit files in the cloned folder")
    print(f"    • Run script --check again to verify")
    print(f"  If detection looks correct:")
    print(f"    • Run script --resume to continue\n")


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def node_authenticate(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Authenticate ───────────────────────────")
    try:
        fork_owner = get_authenticated_user(state["token"])
        return {**state, "fork_owner": fork_owner, "current_step": "authenticate", "error": None}
    except Exception as e:
        return {**state, "error": str(e), "current_step": "authenticate"}


def node_get_default_branch(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Get Default Branch ─────────────────────")
    try:
        branch = get_default_branch(state["repo_url"], state["token"])
        return {**state, "default_branch": branch, "current_step": "get_branch", "error": None}
    except Exception as e:
        return {**state, "error": str(e), "current_step": "get_branch"}


def node_fork_repo(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Fork Repo ───────────────────────────────")
    try:
        fork_url = fork_repo(state["repo_url"], state["token"])
        return {**state, "fork_url": fork_url, "current_step": "fork_repo", "error": None}
    except Exception as e:
        return {**state, "error": str(e), "current_step": "fork_repo"}


def node_clone_repo(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Clone Repo ──────────────────────────────")
    try:
        folder = download_repo(state["fork_url"])
        return {**state, "folder": folder, "current_step": "clone_repo", "error": None}
    except Exception as e:
        return {**state, "error": str(e), "current_step": "clone_repo"}


def node_pause_for_user(state: AgentState) -> AgentState:
    """HITL pause — y/n loop, no --resume needed."""
    print("\n[Agent] ── Node: Pause For User ──────────────────────────")
    folder      = state["folder"]
    folder_path = os.path.abspath(folder)

    save_state({
        "folder":         state["folder"],
        "fork_url":       state["fork_url"],
        "token":          state["token"],
        "fork_owner":     state["fork_owner"],
        "default_branch": state["default_branch"],
        "repo_url":       state["repo_url"],
        "openai_api_key": state["openai_api_key"],
        "paused":         True,
    })

    print(f"\n{'='*55}")
    print(f"[Agent] ⏸️  PAUSED — Repo cloned and ready for your changes!")
    print(f"{'='*55}")
    print(f"[Agent] 📁 Location: {folder_path}")
    print(f"[Agent]")
    print(f"[Agent] Make any changes you want:")
    print(f"[Agent]   • Add .env with API keys")
    print(f"[Agent]   • Edit source files")
    print(f"[Agent]   • Add/fix requirements.txt")
    print(f"[Agent]   • Add missing data files")
    print(f"{'='*55}\n")

    try:
        subprocess.Popen(["code", folder_path])
        print(f"[Agent] ✅ VS Code opened at: {folder_path}")
    except FileNotFoundError:
        print(f"[Agent] ⚠️  VS Code not found — open manually: code {folder_path}")

    print()

    while True:
        answer = input("[Agent] Are you done making changes? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            print("[Agent] ▶️  Continuing deployment...\n")
            break
        elif answer in ("n", "no"):
            print("[Agent] ⏳ Take your time. Edit files, then type y when ready.")
        else:
            print("[Agent] Please type y or n.")

    return {**state, "current_step": "pause_for_user"}


def node_create_branch_and_dockerfile(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Create Branch & Dockerfile ─────────────")
    folder         = state["folder"]
    default_branch = state["default_branch"]
    openai_api_key = state["openai_api_key"]
    fork_url       = state["fork_url"]

    try:
        subprocess.run(["git", "remote", "set-url", "origin", fork_url], cwd=folder)
        subprocess.run(["git", "checkout", default_branch],              cwd=folder, check=True)
        subprocess.run(["git", "pull", "origin", default_branch],        cwd=folder, check=True)
        subprocess.run(["git", "checkout", "-B", "ai-docker-setup"],     cwd=folder, check=True)

        dockerfile_content, context = generate_dockerfile_with_openai(folder, openai_api_key)

        gitignore_path = os.path.join(folder, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                lines = f.readlines()
            filtered = [l for l in lines if l.strip().lower() not in
                        ("dockerfile", "/dockerfile", "dockerfile/")]
            if len(filtered) != len(lines):
                with open(gitignore_path, "w") as f:
                    f.writelines(filtered)
                subprocess.run(["git", "add", ".gitignore"], cwd=folder, check=True)

        subprocess.run(["git", "add", "--force", "Dockerfile"], cwd=folder, check=True)
        result = subprocess.run(
            ["git", "commit", "-m", "AI generated Dockerfile via OpenAI"],
            cwd=folder, capture_output=True, text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            combined = (result.stdout + result.stderr).lower()
            if "nothing to commit" in combined or "nothing added to commit" in combined:
                print("[Agent] ℹ️  Dockerfile already up to date — skipping commit")
            else:
                raise RuntimeError(f"Commit failed: {result.stderr.strip()}")
        else:
            print("[Agent] Committed Dockerfile")

        return {**state,
                "context":      context,
                "dockerfile":   dockerfile_content,
                "current_step": "create_branch_and_dockerfile",
                "error":        None}
    except Exception as e:
        return {**state, "error": str(e), "current_step": "create_branch_and_dockerfile"}


def node_test_docker(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Test Docker Image ──────────────────────")
    folder         = state["folder"]
    context        = state["context"]
    openai_api_key = state["openai_api_key"]
    app_name       = os.path.basename(folder).lower().replace("_", "-")

    test_passed = test_docker_image(
        folder=folder,
        app_name=app_name,
        context=context,
        openai_api_key=openai_api_key,
        max_retries=3,
    )
    return {**state, "test_passed": test_passed, "current_step": "test_docker"}


def node_hitl_pr_approval(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Create PR & Wait For Approval ──────────")
    return {**state, "pr_approved": True, "current_step": "hitl_pr_approval"}


def poll_pr_status(repo_url, token, fork_owner, poll_interval=30, timeout_minutes=30):
    repo      = repo_url.replace("https://github.com/", "").rstrip("/")
    headers   = make_github_headers(token)
    check_url = f"https://api.github.com/repos/{repo}/pulls"
    deadline  = time.time() + timeout_minutes * 60

    # Use same head_ref logic as create_pull_request
    repo_owner = repo.split("/")[0]
    head_param = "ai-docker-setup" if fork_owner == repo_owner else f"{fork_owner}:ai-docker-setup"

    print(f"[Agent] 👀 Polling GitHub PR status every {poll_interval}s (timeout: {timeout_minutes}min)...")

    while time.time() < deadline:
        try:
            r = requests.get(check_url, headers=headers,
                             params={"head": head_param, "state": "open"})
            open_prs = r.json() if r.status_code == 200 else []

            if open_prs:
                pr = open_prs[0]
                print(f"[Agent] ⏳ PR still open: {pr['html_url']}")
                time.sleep(poll_interval)
                continue

            r2 = requests.get(check_url, headers=headers,
                              params={"head": head_param, "state": "closed"})
            closed_prs = r2.json() if r2.status_code == 200 else []

            if closed_prs:
                pr = closed_prs[0]
                if pr.get("merged_at"):
                    print(f"[Agent] ✅ PR MERGED: {pr['html_url']}")
                    return "merged"
                else:
                    print(f"[Agent] ❌ PR CLOSED/REJECTED: {pr['html_url']}")
                    return "closed"

            print("[Agent] ⏳ PR not found yet — waiting...")
            time.sleep(poll_interval)

        except Exception as e:
            print(f"[Agent] ⚠️  Poll error: {e} — retrying...")
            time.sleep(poll_interval)

    return "timeout"


def node_push_and_create_pr(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Push Branch, Create PR & Wait ──────────")
    try:
        push_branch(state["folder"], state["fork_url"], state["token"])
        pr_url = create_pull_request(
            state["repo_url"], state["token"],
            state["fork_owner"], state["default_branch"]
        )
        print(f"\n[Agent] 🔗 PR created. Waiting for approval on GitHub...")
        print(f"[Agent] 👉 Open this URL to review and merge the PR:")
        print(f"[Agent]    {pr_url or 'Check GitHub for PR URL'}")
        print(f"[Agent] ⏳ Agent will continue automatically once PR is merged.\n")

        status = poll_pr_status(
            state["repo_url"], state["token"],
            state["fork_owner"],
            poll_interval=30,
            timeout_minutes=30,
        )

        if status == "merged":
            print("[Agent] ✅ PR merged — pulling latest main from GitHub...")
            subprocess.run(["git", "checkout", state["default_branch"]], cwd=state["folder"], check=True)
            subprocess.run(["git", "pull", "origin", state["default_branch"]], cwd=state["folder"], check=True)
            print("[Agent] ✅ Local folder updated to merged main — ready for deployment!")

            # ── Test locally after merge ──────────────────────────────────────
            print("\n[Agent] 🧪 Testing merged code locally before deployment...")
            context = state.get("context") or deep_scan_repo(state["folder"])
            local_ok = run_project_locally(
                folder=state["folder"],
                context=context,
                openai_api_key=state["openai_api_key"],
                max_retries=3,
            )
            if local_ok:
                print("[Agent] ✅ Local test passed — proceeding to deployment")
            else:
                print("[Agent] ⚠️  Local test did not pass — proceeding anyway")
            # ─────────────────────────────────────────────────────────────────

            return {**state, "pr_url": pr_url or "", "current_step": "push_and_create_pr", "error": None}
        elif status == "closed":
            msg = "PR was closed/rejected on GitHub. Fix your changes and run --resume."
            print(f"[Agent] ❌ {msg}")
            save_state({
                "folder": state["folder"], "fork_url": state["fork_url"],
                "token": state["token"], "fork_owner": state["fork_owner"],
                "default_branch": state["default_branch"], "repo_url": state["repo_url"],
                "openai_api_key": state["openai_api_key"], "paused": True,
            })
            return {**state, "error": msg, "current_step": "push_and_create_pr"}
        else:
            msg = "PR approval timed out after 30 minutes. Merge the PR on GitHub then run --resume."
            print(f"[Agent] ⏰ {msg}")
            save_state({
                "folder": state["folder"], "fork_url": state["fork_url"],
                "token": state["token"], "fork_owner": state["fork_owner"],
                "default_branch": state["default_branch"], "repo_url": state["repo_url"],
                "openai_api_key": state["openai_api_key"], "paused": True,
            })
            return {**state, "error": msg, "current_step": "push_and_create_pr"}

    except Exception as e:
        return {**state, "error": str(e), "current_step": "push_and_create_pr"}


def node_hitl_deploy_approval(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: HITL Deploy Approval ───────────────────")
    print("\n[Agent] ── Deployment Decision ──────────────────────────")
    deploy_approval = input("[Agent] Do you want to deploy the application? (yes/no): ").strip().lower()
    approved        = deploy_approval in ("yes", "y")
    if not approved:
        print("[Agent] 🛑 Deployment skipped by user. All done!")
    return {**state, "deploy_approved": approved, "current_step": "hitl_deploy_approval"}


def node_collect_deploy_info(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Collect Deploy Info ────────────────────")
    print("\n[Agent] ── Deployment ────────────────────────────────────")
    deploy_input = input("\nWhere to deploy? (e.g. 'deploy to railway'): ").strip()
    targets      = parse_deploy_targets(deploy_input, state["openai_api_key"])

    if not targets:
        print("[Agent] No valid targets. Skipping deployment.")
        return {**state, "deploy_targets": [], "current_step": "collect_deploy_info"}

    app_name = os.getenv("APP_NAME", "").strip() or input("\nApp name: ").strip()

    # Collect env vars — GitHub repo doesn't have .env so we ask here
    env_vars = {}
    folder   = state["folder"]
    env_file = os.path.join(folder, ".env")

    if os.path.exists(env_file):
        print(f"[Agent] 📦 Found local .env — loading env vars for deployment...")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
        print(f"[Agent] ✅ Loaded {len(env_vars)} vars: {list(env_vars.keys())}")
        add_more = input("[Agent] Add more env vars? (y/n): ").strip().lower()
        if add_more in ("y", "yes"):
            print("[Agent] Enter vars one by one. Press Enter with empty key to finish.")
            while True:
                key = input("  KEY: ").strip()
                if not key:
                    break
                val = input(f"  {key}=: ").strip()
                env_vars[key] = val
    else:
        print(f"[Agent] ℹ️  No local .env found.")
        needs_env = input("[Agent] Does your app need environment variables? (y/n): ").strip().lower()
        if needs_env in ("y", "yes"):
            print("[Agent] Enter your env vars one by one. Press Enter with empty key to finish.")
            while True:
                key = input("  KEY: ").strip()
                if not key:
                    break
                val = input(f"  {key}=: ").strip()
                env_vars[key] = val
            if env_vars:
                print(f"[Agent] ✅ Collected {len(env_vars)} env vars")

    return {**state,
            "deploy_targets": targets,
            "app_name":       app_name,
            "env_vars":       env_vars,
            "current_step":   "collect_deploy_info"}
def collect_credentials(targets, app_name):
    app_name = app_name.lower().replace(" ", "-")
    if len(app_name) < 4:
        app_name = f"{app_name}-app"
    print(f"[Agent] App name: {app_name}")
    creds = {}

    def get_value(env_key, label):
        val = os.getenv(env_key, "").strip()
        if val:
            print(f"  ✅ {env_key} loaded from .env")
            return val
        return input(f"  {label}: ").strip()

    if "aws" in targets:
        creds["aws"] = {
            "access_key": get_value("AWS_ACCESS_KEY_ID",     "AWS_ACCESS_KEY_ID"),
            "secret_key": get_value("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
            "region":     get_value("AWS_REGION",            "AWS_REGION (e.g. ap-south-1)"),
            "app_name":   app_name,
        }

    if "azure" in targets:
        creds["azure"] = {
            "client_id":       get_value("AZURE_CLIENT_ID",       "AZURE_CLIENT_ID"),
            "client_secret":   get_value("AZURE_CLIENT_SECRET",   "AZURE_CLIENT_SECRET"),
            "tenant_id":       get_value("AZURE_TENANT_ID",       "AZURE_TENANT_ID"),
            "subscription_id": get_value("AZURE_SUBSCRIPTION_ID", "AZURE_SUBSCRIPTION_ID"),
            "resource_group":  get_value("AZURE_RESOURCE_GROUP",  "AZURE_RESOURCE_GROUP"),
            "dockerhub_user":  get_value("DOCKERHUB_USERNAME",    "Docker Hub Username"),
            "dockerhub_pass":  get_value("DOCKERHUB_PASSWORD",    "Docker Hub Password"),
            "app_name":        app_name,
            "fork_url":        "",
        }

    if "render" in targets:
        creds["render"] = {
            "api_key":  get_value("RENDER_API_KEY", "RENDER_API_KEY"),
            "app_name": app_name,
            "fork_url": "",
        }

    if "railway" in targets:
        creds["railway"] = {
            "token":          get_value("RAILWAY_TOKEN",      "RAILWAY_TOKEN"),
            "dockerhub_user": get_value("DOCKERHUB_USERNAME", "Docker Hub Username"),
            "dockerhub_pass": get_value("DOCKERHUB_PASSWORD", "Docker Hub Password"),
            "app_name":       app_name,
        }

    return creds

def node_deploy(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Deploy ──────────────────────────────────")
    targets  = state.get("deploy_targets", [])
    app_name = state.get("app_name", "")
    folder   = state["folder"]
    fork_url = state["fork_url"]

    if not targets:
        print("[Agent] No deploy targets — skipping.")
        return {**state, "deploy_results": {}, "current_step": "deploy"}

    creds = collect_credentials(targets, app_name)
    for platform in creds:
        creds[platform]["fork_url"] = fork_url
        creds[platform]["folder"]   = folder
        creds[platform]["env_vars"] = state.get("env_vars", {})
    results = deploy_to_platforms(targets, folder, fork_url, creds)
    return {**state, "deploy_results": results, "current_step": "deploy"}


def node_done(state: AgentState) -> AgentState:
    print("\n[Agent] ══════════════════════════════════════════════════")
    print("[Agent] ✅ Pipeline complete!")
    results = state.get("deploy_results", {})
    if results:
        for p, url in results.items():
            icon = "✅" if not str(url).startswith("FAILED") else "❌"
            print(f"  {icon} {p.upper():<10} -> {url}")
    print("[Agent] ══════════════════════════════════════════════════\n")
    return {**state, "current_step": "done"}


def node_error(state: AgentState) -> AgentState:
    print(f"\n[Agent] ❌ Pipeline error at step '{state.get('current_step')}': {state.get('error')}")
    print("[Agent] Fix the issue and run 'python langgraph_agent.py --resume' again")
    return state


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def route_after_auth(state: AgentState) -> str:
    return "error" if state.get("error") else "get_default_branch"

def route_after_branch(state: AgentState) -> str:
    return "error" if state.get("error") else "fork_repo"

def route_after_fork(state: AgentState) -> str:
    return "error" if state.get("error") else "clone_repo"

def route_after_clone(state: AgentState) -> str:
    return "error" if state.get("error") else "pause_for_user"

def route_after_dockerfile(state: AgentState) -> str:
    return "error" if state.get("error") else "test_docker"

def route_after_test(state: AgentState) -> str:
    if state.get("test_passed"):
        return "hitl_pr_approval"
    else:
        print("\n[Agent] ❌ Docker test FAILED — skipping PR and deployment")
        print("[Agent] Fix the issues and run 'python langgraph_agent.py --resume' again")
        return "done"

def route_after_pr_approval(state: AgentState) -> str:
    return "push_and_create_pr"

def route_after_push_pr(state: AgentState) -> str:
    return "error" if state.get("error") else "hitl_deploy_approval"

def route_after_deploy_approval(state: AgentState) -> str:
    return "collect_deploy_info" if state.get("deploy_approved") else "done"

def route_after_collect_deploy(state: AgentState) -> str:
    return "deploy" if state.get("deploy_targets") else "done"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD THE LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("authenticate",                 node_authenticate)
    builder.add_node("get_default_branch",           node_get_default_branch)
    builder.add_node("fork_repo",                    node_fork_repo)
    builder.add_node("clone_repo",                   node_clone_repo)
    builder.add_node("pause_for_user",               node_pause_for_user)
    builder.add_node("create_branch_and_dockerfile", node_create_branch_and_dockerfile)
    builder.add_node("test_docker",                  node_test_docker)
    builder.add_node("hitl_pr_approval",             node_hitl_pr_approval)
    builder.add_node("push_and_create_pr",           node_push_and_create_pr)
    builder.add_node("hitl_deploy_approval",         node_hitl_deploy_approval)
    builder.add_node("collect_deploy_info",          node_collect_deploy_info)
    builder.add_node("deploy",                       node_deploy)
    builder.add_node("done",                         node_done)
    builder.add_node("error",                        node_error)

    builder.set_entry_point("authenticate")

    builder.add_conditional_edges("authenticate",       route_after_auth,            {"get_default_branch": "get_default_branch", "error": "error"})
    builder.add_conditional_edges("get_default_branch", route_after_branch,          {"fork_repo": "fork_repo",                   "error": "error"})
    builder.add_conditional_edges("fork_repo",          route_after_fork,            {"clone_repo": "clone_repo",                 "error": "error"})
    builder.add_conditional_edges("clone_repo",         route_after_clone,           {"pause_for_user": "pause_for_user",         "error": "error"})
    # ── CRITICAL: this edge makes pause_for_user flow into dockerfile generation ──
    builder.add_edge("pause_for_user", "create_branch_and_dockerfile")
    builder.add_conditional_edges("create_branch_and_dockerfile", route_after_dockerfile, {"test_docker": "test_docker", "error": "error"})
    builder.add_conditional_edges("test_docker",        route_after_test,            {"hitl_pr_approval": "hitl_pr_approval",     "done": "done"})
    builder.add_conditional_edges("hitl_pr_approval",   route_after_pr_approval,     {"push_and_create_pr": "push_and_create_pr"})
    builder.add_conditional_edges("push_and_create_pr", route_after_push_pr,         {"hitl_deploy_approval": "hitl_deploy_approval", "error": "error"})
    builder.add_conditional_edges("hitl_deploy_approval", route_after_deploy_approval, {"collect_deploy_info": "collect_deploy_info", "done": "done"})
    builder.add_conditional_edges("collect_deploy_info", route_after_collect_deploy,  {"deploy": "deploy", "done": "done"})

    builder.add_edge("deploy", "done")
    builder.add_edge("done",   END)
    builder.add_edge("error",  END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# RESUME GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_resume_graph():
    builder = StateGraph(AgentState)

    builder.add_node("create_branch_and_dockerfile", node_create_branch_and_dockerfile)
    builder.add_node("test_docker",                  node_test_docker)
    builder.add_node("hitl_pr_approval",             node_hitl_pr_approval)
    builder.add_node("push_and_create_pr",           node_push_and_create_pr)
    builder.add_node("hitl_deploy_approval",         node_hitl_deploy_approval)
    builder.add_node("collect_deploy_info",          node_collect_deploy_info)
    builder.add_node("deploy",                       node_deploy)
    builder.add_node("done",                         node_done)
    builder.add_node("error",                        node_error)

    builder.set_entry_point("create_branch_and_dockerfile")

    builder.add_conditional_edges("create_branch_and_dockerfile", route_after_dockerfile,      {"test_docker": "test_docker",             "error": "error"})
    builder.add_conditional_edges("test_docker",                  route_after_test,            {"hitl_pr_approval": "hitl_pr_approval",   "done": "done"})
    builder.add_conditional_edges("hitl_pr_approval",             route_after_pr_approval,     {"push_and_create_pr": "push_and_create_pr"})
    builder.add_conditional_edges("push_and_create_pr",           route_after_push_pr,         {"hitl_deploy_approval": "hitl_deploy_approval", "error": "error"})
    builder.add_conditional_edges("hitl_deploy_approval",         route_after_deploy_approval, {"collect_deploy_info": "collect_deploy_info", "done": "done"})
    builder.add_conditional_edges("collect_deploy_info",          route_after_collect_deploy,  {"deploy": "deploy", "done": "done"})

    builder.add_edge("deploy", "done")
    builder.add_edge("done",   END)
    builder.add_edge("error",  END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    def get_env_or_input(env_key, label):
        val = os.getenv(env_key, "").strip()
        if val:
            print(f"  ✅ {env_key} loaded from .env")
            return val
        return input(f"{label}: ").strip()

    if "--check" in sys.argv:
        state = load_state()
        if not state:
            print("[Agent] ❌ No paused session found. Run normally first.")
            sys.exit(1)
        check_mode(state["folder"])
        sys.exit(0)

    if "--resume" in sys.argv:
        saved = load_state()
        if not saved:
            print("[Agent] ❌ No paused session found. Run normally first.")
            sys.exit(1)

        print(f"\n[Agent] ▶️  Resuming from paused state...")
        print(f"[Agent] Folder: {saved['folder']}")
        os.remove(STATE_FILE)

        initial_state: AgentState = {
            "repo_url":        saved["repo_url"],
            "token":           saved["token"],
            "openai_api_key":  saved["openai_api_key"],
            "fork_owner":      saved["fork_owner"],
            "default_branch":  saved["default_branch"],
            "fork_url":        saved["fork_url"],
            "folder":          saved["folder"],
            "context":         {},
            "dockerfile":      "",
            "test_passed":     False,
            "deploy_targets":  [],
            "app_name":        "",
            "deploy_results":  {},
            "pr_approved":     False,
            "pr_url":          "",
            "deploy_approved": False,
            "env_vars":        {},
            "paused":          False,
            "error":           None,
            "current_step":    "resume",
        }

        graph = build_resume_graph()
        graph.invoke(initial_state)
        sys.exit(0)

    print("\n[Agent] ── Configuration ─────────────────────────────────")
    repo_url       = input("Enter GitHub repo URL: ").strip()
    token          = get_env_or_input("GITHUB_TOKEN",   "GitHub Token")
    openai_api_key = get_env_or_input("OPENAI_API_KEY", "OpenAI API Key")

    initial_state: AgentState = {
        "repo_url":        repo_url,
        "token":           token,
        "openai_api_key":  openai_api_key,
        "fork_owner":      "",
        "default_branch":  "",
        "fork_url":        "",
        "folder":          "",
        "context":         {},
        "dockerfile":      "",
        "test_passed":     False,
        "deploy_targets":  [],
        "app_name":        "",
        "deploy_results":  {},
        "pr_approved":     False,
        "pr_url":          "",
        "deploy_approved": False,
        "env_vars":        {},
        "paused":          False,
        "error":           None,
        "current_step":    "start",
    }

    graph = build_graph()
    graph.invoke(initial_state)
