# ShopTalk — Phase 11: Deployment runbook

One Docker image runs the API (FastAPI, port 8000) and the UI (Gradio, port 7860). You test it on your laptop in
**fake mode** (no GPU needed), then run the real thing on **one EC2 GPU instance** for the demo and stop it right after.

| Step | Where | Time | Cost |
|---|---|---|---|
| 1. GPU quota + budget alert | AWS console | 10 min (+ approval wait, often 1–2 days) | free |
| 2. Prepare repo + artifacts | Colab: `ShopTalk_Phase11_Prepare.ipynb` | 15 min | free |
| 3. Local Docker test (fake mode) | your laptop | 30–60 min (first build) | free |
| 4. Deploy on EC2 | AWS + PowerShell | 45–60 min | g4dn.xlarge ≈ $0.53/h |
| 5. Record the demo | browser on your laptop | 30–60 min | (instance running) |
| 6. Stop / tear down | AWS console | 5 min | disk ≈ $0.27/day until terminated |

**Why EC2 g4dn.xlarge:** same Tesla T4 as Colab, so the Phase 9–10 latency numbers carry over; the
"Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)" already has the NVIDIA driver, Docker and the NVIDIA
Container Toolkit. SageMaker endpoints expect one model behind `/invocations`, not an API + UI pair, and cost more
per hour; a CPU instance would take tens of seconds per answer with Qwen-3B + Whisper.

---

## 1. Today: GPU quota and a budget alert

**GPU quota** (new accounts start at 0 for GPU instances):
1. AWS console → choose your region (e.g. **US East (N. Virginia) us-east-1**) top right.
2. **Service Quotas → AWS services → Amazon Elastic Compute Cloud (Amazon EC2)**.
3. Search **Running On-Demand G and VT instances** → **Request increase at account level** → **8** (it counts vCPUs;
   g4dn.xlarge uses 4). Reason: "Course project demo: one g4dn.xlarge for a few hours."
4. Wait for the approval email. Don't confuse it with "All G and VT **Spot** Instance Requests" — that is a different quota.

**Budget alert** (so a forgotten instance can't surprise you):
**Billing and Cost Management → Budgets → Create budget → Use a template → Monthly cost budget**, amount **$20**,
your email. You get an email at 85% and 100%.

---

## 2. Prepare the repo and the artifacts (Colab)

Run `ShopTalk_Phase11_Prepare.ipynb` top to bottom. It:
- writes `Dockerfile`, `.dockerignore`, `docker/start.sh`, `docker/smoke_test.py`, `deploy/ec2.sh` and this guide into the repo on Drive;
- writes **`requirements-docker.txt`**: the exact package versions from the Colab runtime the project was tested on;
- zips the two things the container needs but GitHub shouldn't hold — the LoRA embedding model and the Chroma index —
  into **`deploy_artifacts/shoptalk_artifacts.zip`** on Drive;
- pushes the current Drive code (Phases 7–10 included) to **GitHub** (needs a personal access token; see the notebook).

Then download `shoptalk_artifacts.zip` from Drive to your laptop (you upload it to S3 in step 4).

---

## 3. Local test on your laptop (fake mode)

Fake mode (`SHOPTALK_FAKE=1`) runs the same image, endpoints, UI, microphone and feedback buttons with stand-in models,
so it works without an NVIDIA GPU. It proves the container is correct; the real models are tested on EC2.

**Install Docker Desktop (Windows):**
1. Download **Docker Desktop for Windows** from docs.docker.com/desktop → install with **"Use WSL 2"** ticked → restart.
2. Start Docker Desktop and wait for "Engine running". In PowerShell: `docker run --rm hello-world` → "Hello from Docker!".
   If it asks for WSL: `wsl --install` in an administrator PowerShell, restart, try again.

**Build and run** (PowerShell, in the cloned repo folder: `git clone https://github.com/<you>/<repo>.git; cd <repo>`):
```powershell
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t shoptalk:cpu .
docker run --rm -d --name shoptalk -p 7860:7860 -p 8000:8000 -e SHOPTALK_FAKE=1 -e SHOPTALK_TTS=1 shoptalk:cpu
docker exec shoptalk python /app/docker/smoke_test.py
```
Expected: `ALL CHECKS PASSED` (answers are fake: "red sneakers", "Show me a red shirt…").
Then open **http://localhost:7860** in your browser: type a request, press 🎤 (localhost is allowed to use the
microphone), try 👍/👎 and "View product". Stop with `docker stop shoptalk`.

The CPU build is ~1.5 GB; the first build takes 5–15 min depending on your internet.

---

## 4. Deploy on EC2

### 4.1 Upload the artifacts to S3
1. **S3 → Create bucket** → name e.g. `shoptalk-artifacts-<yourname>` (same region), keep "Block all public access" ON.
2. Open the bucket → **Upload** → `shoptalk_artifacts.zip`.
3. Click the uploaded file → **Object actions → Share with a presigned URL** → **12 hours** → **Create** → copy the URL.
   (The URL lets the instance download this one file without AWS keys; it expires by itself.)

### 4.2 Launch the instance
**EC2 → Launch instance**:

| Setting | Value |
|---|---|
| Name | `shoptalk-demo` |
| AMI | search **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)** (x86_64) |
| Instance type | **g4dn.xlarge** |
| Key pair | **Create new key pair** → `shoptalk-key`, RSA, **.pem** → it downloads; keep it |
| Network | Allow SSH traffic from **My IP** (nothing else — the UI is reached through the SSH tunnel) |
| Storage | **100 GiB gp3** |
| Purchasing | On-Demand (Spot is ~50% cheaper but can be interrupted mid-demo) |

**Launch**, wait until "Running" with 2/2 status checks, and copy the **Public IPv4 address**.

### 4.3 Connect from Windows (PowerShell)
Fix the key's permissions once (Windows SSH refuses keys others can read):
```powershell
cd $HOME\Downloads
icacls.exe shoptalk-key.pem /reset
icacls.exe shoptalk-key.pem /grant:r "$($env:USERNAME):(R)"
icacls.exe shoptalk-key.pem /inheritance:r
```
Connect **with the tunnel** (your laptop's ports 7860/8000 now lead to the instance):
```powershell
ssh -i shoptalk-key.pem -L 7860:localhost:7860 -L 8000:localhost:8000 ubuntu@<PUBLIC-IP>
```
Type `yes` the first time. Keep this window open while you work and record.

### 4.4 Set up and start (on the instance)
```bash
git clone https://github.com/<you>/<repo>.git shoptalk && cd shoptalk
#   private repo: git clone https://<github-token>@github.com/<you>/<repo>.git shoptalk
bash deploy/ec2.sh setup "<PRESIGNED-URL>"     # keep the quotes; ~10 min (download + GPU build)
bash deploy/ec2.sh start
bash deploy/ec2.sh logs                        # wait for the models (first start ~5–10 min), Ctrl+C to leave
bash deploy/ec2.sh smoke                       # ALL CHECKS PASSED + a latency sample
```
Now open **http://localhost:7860** on your laptop (through the tunnel). The status line should show
🟢 API ready · Qwen · 🎤 whisper.

Need a link someone else can open? `bash deploy/ec2.sh tunnel` prints a public https URL (Ctrl+C closes it).

---

## 5. Record the demo (plan item 11.3)

**Before recording:** run `bash deploy/ec2.sh smoke` once (warms the models) and try your exact demo queries — pick
ones that work. Close other tabs; zoom the browser to 110–125%.

**Tools (Windows):** Xbox Game Bar (**Win + Alt + R** records the active window with your microphone) or OBS Studio
for the full screen; trim in Clipchamp. Aim for **3–5 minutes**.

**Shot list** (covers everything the plan asks for):
| # | Show | Example |
|---|---|---|
| 1 | Intro: what ShopTalk is, running on AWS (status line, 5 s) | — |
| 2 | Simple product query | "white sneakers" |
| 3 | Query with constraints (price/colour) | "a blue area rug under $100" |
| 4 | **Follow-up** in the same chat | "show me cheaper ones" or "in gray" |
| 5 | Another category | "gold hoop earrings for women" / "a gray sofa" |
| 6 | Something the catalog doesn't have → honest "no match" | test one first, e.g. "a used car" |
| 7 | **Spoken query** with 🎤 — point at the transcript in the box, then Send | "black leather shoes for men" |
| 8 | Optional: 🔊 read aloud | — |
| 9 | **Feedback**: 👍 on a good answer, 👎 with a comment on another, click **View product** | — |
| 10 | Proof it was logged (terminal) | `tail -n 3 ~/shoptalk/logs/feedback.jsonl` |

Avoid the known Phase 9 failure queries (sun hat, gemstone ring, diamond studs) unless you show one on purpose as a
known limitation with the fix from `failure_cases_final.md`.

---

## 6. Stop and tear down

**Right after recording:**
```bash
bash deploy/ec2.sh save-logs     # optional: keep the demo's request/feedback logs
bash deploy/ec2.sh stop
```
Download the logs if you want them (PowerShell, new window):
`scp -i shoptalk-key.pem ubuntu@<PUBLIC-IP>:~/shoptalk/shoptalk_logs_*.tgz .`

Then **EC2 → Instances → shoptalk-demo → Instance state → Stop.** GPU billing ends; the 100 GB disk stays
(≈ $8/month, ≈ $0.27/day) so you can start it again for a retake (the public IP changes; models are already cached).

**When the project is graded:**
1. **Terminate** the instance (Instance state → Terminate) — this deletes its disk too.
2. **EC2 → Volumes / Snapshots**: nothing left with your instance's name.
3. **S3**: delete `shoptalk_artifacts.zip` and the bucket.
4. **Billing → Bills** the next day: no running EC2 charges.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Launch error *VcpuLimitExceeded* / *You have requested more vCPU capacity…* | GPU quota not approved yet (step 1), or requested in another region |
| *InsufficientInstanceCapacity* | No T4s free in that zone: retry later, or pick another subnet/AZ in the launch "Network settings" |
| `ssh: … UNPROTECTED PRIVATE KEY FILE` | Run the three `icacls` commands in 4.3 |
| `ssh: connection timed out` | Security group SSH rule is for an old IP: EC2 → Security groups → edit inbound → My IP |
| `setup` download fails | Presigned URL expired (12 h): create a new one (4.1 step 3) |
| `/health` stays `loading` | First start downloads ~8 GB from Hugging Face: `bash deploy/ec2.sh logs` |
| `CUDA error: no kernel image` / torch can't see GPU | Use the Ubuntu 24.04 Base OSS DLAMI (driver supports CUDA 13); `nvidia-smi` must work on the host |
| UI loads but 🎤 says no microphone | Use `http://localhost:7860` through the SSH tunnel (or the https `tunnel` link) — not `http://<public-ip>` |
| Port 7860 already in use on the laptop | Stop the local container: `docker stop shoptalk` |
| Disk full | `docker system prune` (old images), or 100 GiB storage at launch |
