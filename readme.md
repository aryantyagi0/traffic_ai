# 🚀 AI Docker Deployment Agent (LangGraph Version)

An intelligent end-to-end DevOps automation agent that:
- Clones any GitHub repository
- Analyzes the project structure
- Automatically generates a production-ready Dockerfile
- Tests the container locally
- Pushes changes via Pull Request
- Deploys to cloud platforms (AWS, Azure, Render, Railway)

Built using **LangGraph + OpenAI + Docker + GitHub APIs**

---

## 🧠 Key Features

### ✅ Automatic Repository Analysis
- Detects language (Python, Node, Go, Java, Ruby, etc.)
- Detects framework (FastAPI, Flask, Streamlit, React, Django, etc.)
- Identifies entry points intelligently using content scoring + LLM fallback

### ✅ AI Dockerfile Generation
- Uses OpenAI GPT-4o to generate optimized Dockerfiles
- Handles ML, backend, frontend, and hybrid apps
- Supports GPU, Conda, HuggingFace, Streamlit, Gradio, and more

### ✅ Self-Healing Docker Builds
- Detects build/runtime errors automatically
- Auto-fixes Dockerfile using GPT-4o (up to 3 retries)

### ✅ Local Docker Testing
- Builds image locally
- Runs container
- Performs HTTP health checks
- Skips runtime test for script-only projects

### ✅ GitHub Automation
- Forks repo
- Creates branch (`ai-docker-setup`)
- Pushes Dockerfile
- Creates Pull Request
- Polls for merge automatically

### ✅ Human-in-the-Loop (HITL)
- Pause after clone for manual edits
- PR approval step
- Deployment confirmation prompt

### ✅ Multi-Platform Deployment
| Platform | Deployment Method |
|----------|------------------|
| AWS | ECR + EC2 (Docker pull on launch) |
| Azure | ACR + Container Apps |
| Render | GitHub branch (Docker env) |
| Railway | Docker Hub + GraphQL API |

---
THE FLOW IS 
Phase 1 — Setup
Step 1: Start — Run python langgraph_agent.py. Enter GitHub repo URL, GitHub token, and OpenAI API key.
Step 2: Authenticate — Agent verifies your GitHub token and OpenAI key are valid.
Step 3: Get default branch — Fetches repo metadata via GitHub API to find the main/master branch.
Step 4: Fork repository — Forks the target repo to your GitHub account. If already forked, reuses the existing fork.
Step 5: Clone repository — git clone the fork to your local machine.

Phase 2 — Human-in-the-Loop (HITL)
Step 6: HITL pause — Agent pauses and opens VS Code. You can:

Add a .env file with API keys
Fix broken dependencies in requirements.txt
Edit source files
Add missing data files

Type y when done to continue.

Phase 3 — AI Dockerfile Generation
Step 7: Generate Dockerfile — GPT-4o deep-scans the repo (detects language, framework, entry point) and writes a production-ready Dockerfile.
Step 8: Test Docker locally — Agent runs docker build → docker run → HTTP health check on the app port.
Step 9: Test passed?

No → GPT-4o auto-fixes the Dockerfile and retries (up to 3 times)
Yes → proceed to PR


Phase 4 — GitHub PR
Step 10: Push branch + create PR — Pushes the ai-docker-setup branch with the Dockerfile to your fork and opens a Pull Request against the original repo.
Step 11: Poll PR status — Agent polls GitHub every 30 seconds waiting for you to merge the PR. You review and merge it on GitHub. Agent detects the merge and pulls the latest code automatically.

Phase 5 — Deployment
Step 12: Deploy the app?

No → skip to Done
Yes → continue

Step 13: Collect deploy info — Agent asks:

Where to deploy (aws / azure / render / railway)
App name
Any env vars needed

Step 14: Deploy to target platform
PlatformWhat happensAWSBuild image → push to ECR → launch EC2 → EC2 pulls and runs containerAzureBuild image → push to ACR → deploy to Container AppsRenderPoint Render at your GitHub branch → Render builds and deploysRailwayBuild image → push to Docker Hub → Railway pulls and runs
Step 15: Deployment complete — Agent prints the public URL.
Step 16: Done — Pipeline finishes. App is live.


## ⚙️ Installation

### 1️⃣ Clone the repo

### 2️⃣ Install dependencies


### 3️⃣ Setup environment variables

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key
GITHUB_TOKEN=your_github_token

# AWS
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1

# Railway
RAILWAY_TOKEN=your_railway_token

# Docker Hub (Railway + Azure)
DOCKERHUB_USERNAME=your_username
DOCKERHUB_PASSWORD=your_password

# Render
RENDER_API_KEY=your_render_key

# Azure
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
AZURE_TENANT_ID=
AZURE_SUBSCRIPTION_ID=
AZURE_RESOURCE_GROUP=
```

---

## ▶️ Usage

### 🔹 Run the agent
```bash
python langgraph_agent2.py
```

You will be prompted:
```
Enter GitHub repo URL: https://github.com/user/project
GitHub Token: (auto-loaded from .env)
OpenAI API Key: (auto-loaded from .env)
```

### 🔹 Resume a paused session
```bash
python langgraph_agent2.py --resume
```

### 🔹 Check detected project info
```bash
python langgraph_agent2.py --check
```

---

## 🔄 Workflow Steps
```
1. Fork & clone repo
2. PAUSE — make manual edits (optional)
3. Generate Dockerfile (GPT-4o)
4. Test Docker locally (build + run + health check)
5. Create PR on GitHub
6. Poll until PR is merged
7. Ask deployment target
8. Deploy to selected platform
```

---

## ☁️ AWS Deployment — How It Works

AWS deployment is fully automated with **no manual setup required**:

### What the agent does:
1. **Creates an ECR repository** to store your Docker image
2. **Builds and pushes** your Docker image to ECR from your local machine
3. **Creates a security group** with your app port and SSH (22) open
4. **Launches an EC2 instance** (t2.micro / t3.micro) with a startup script that:
   - Installs Docker
   - Authenticates to ECR using embedded credentials
   - Pulls your image from ECR
   - Runs the container with `docker run`
5. **Returns the public URL** after 30 seconds

### Prerequisites for AWS:
- AWS account (free tier works for first 12 months)
- IAM user with permissions: `AmazonEC2FullAccess`, `AmazonEC2ContainerRegistryFullAccess`
- Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` in `.env`

### AWS Free Tier limits:
| Service | Free Tier |
|---------|-----------|
| EC2 t2.micro | 750 hrs/month (12 months) |
| ECR | 500 MB/month (forever) |

> ⚠️ **Important:** Stop your EC2 instance from the AWS Console when not in use to avoid charges. The agent prints the Instance ID at the end — use it to find and stop the instance.

### App startup time:
After the agent prints the URL, wait **5–7 minutes** for the EC2 instance to:
- Install Docker (~2 min)
- Pull image from ECR (~1 min)
- Start the container (~30 sec)

### Note on college/corporate networks:
Raw EC2 IP addresses (e.g. `http://13.x.x.x:8501`) may be blocked by firewalls like FortiGate as "Not Rated". Test using **mobile data** or attach a domain name to your EC2 IP.

---

## ⏸️ Pause Mode (HITL)

Agent pauses after cloning — you can:
- Add `.env` with API keys
- Fix broken dependencies
- Modify source code
- Add missing data files

Then continue:
```
Are you done making changes? (y/n): y
```

---

## 🐳 Docker Testing Details

| Check | Description |
|-------|-------------|
| Build | `docker build` with auto-fix on failure |
| Runtime | `docker run` with port binding |
| Health | HTTP check on app port |
| Script projects | Build-only test (no HTTP check) |

---

## 🧩 Tech Stack

| Tool | Purpose |
|------|---------|
| LangGraph | Workflow orchestration |
| OpenAI GPT-4o | Dockerfile generation & auto-fix |
| Docker | Containerization & local testing |
| GitHub API | Fork, branch, PR automation |
| Boto3 | AWS EC2 + ECR deployment |
| Azure SDK | Azure Container Apps deployment |
| Render API | Render web service deployment |
| Railway GraphQL | Railway deployment |

---

## ⚠️ Requirements

- Docker installed and running locally
- Git configured (`git config --global user.name`)
- OpenAI API key (GPT-4o access)
- GitHub personal access token (repo + workflow scope)
- Platform credentials in `.env` for whichever platform you deploy to

---

## 💡 Example Session
```
Enter GitHub repo URL: https://github.com/user/streamlit-app
✅ GITHUB_TOKEN loaded from .env
✅ OPENAI_API_KEY loaded from .env

[Agent] Forking repository...
[Agent] Fork ready
[Agent] Repo cloned: streamlit-app
[Agent] ⏸️  PAUSED — make your changes, then press y

Are you done making changes? (y/n): y

[Agent] Generating Dockerfile...
[Test] ✅✅✅ DOCKER TEST PASSED
[Agent] PR created: https://github.com/user/streamlit-app/pull/1

Where to deploy? aws
App name: myapp

[AWS] ✅ Image pushed to ECR
[AWS] ✅ Instance launched: i-0abc123
[AWS] ✅ Instance running at: http://13.233.x.x:8501
```

---

## 🎯 Use Cases

- DevOps automation for any GitHub repo
- ML model deployment (Streamlit, Gradio, FastAPI)
- CI/CD pipeline bootstrapping
- Hackathon rapid deployment
- Startup MVP deployment
- Multi-repo batch processing

---

## 👨‍💻 Author

**Aryan Tyagi**
BTech CSE (AI/ML) | Web Developer & AI Engineer
