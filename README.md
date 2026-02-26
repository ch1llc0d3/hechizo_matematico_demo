# Hechizo Matemático Demo

Three demos in one repo:

1. **El Hechizo Matemático** — Cosine similarity “matchmaking” over latent vectors (3D visualization).
2. **Sociological Profiling Demo** — Image + EXIF “vibe” analysis with YOLO and a HUD overlay (educational only).
3. **Elo Ranking & Network Meritocracy** — Dating app as a network of nodes; Match = success, Dislike = latency; High-Elo = High-Priority with exponential SRE-style adjustments and a load balancer to avoid a single super-node.

---

## Prerequisites

- **Python 3.8, 3.9, 3.10, or 3.11** (3.9+ recommended).
- A terminal (macOS Terminal, iTerm, or VS Code / Cursor integrated terminal).

Check your Python version:

```bash
python3 --version
```

You should see something like `Python 3.9.x` or `Python 3.11.x`. If `python3` is missing, install Python from [python.org](https://www.python.org/downloads/) or with Homebrew: `brew install python@3.11`.

---

## Setup (step by step)

### 1. Go to the project folder

```bash
cd /Users/laptopmac/hechizo_matematico_demo
```

(Or wherever you cloned/unzipped the project.)

### 2. Create a virtual environment (recommended)

Using the same Python you’ll run the scripts with (e.g. `python3`):

```bash
python3 -m venv .venv
```

If you already have a `.venv` folder, you can skip this step or replace it with a fresh one by renaming the old `.venv` and running the command again.

### 3. Activate the virtual environment

- **macOS / Linux (bash/zsh):**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

When activated, your prompt usually starts with `(.venv)`.

### 4. Upgrade pip (avoids many install issues)

```bash
pip install --upgrade pip
```

### 5. Install dependencies

From the project root (with the venv activated):

```bash
pip install -r requirements.txt
```

This installs:

- `numpy`, `matplotlib`, `scikit-learn` — for the Hechizo Matemático demo.
- `rich` — for the Hechizo Matemático and Elo Ranking (live dashboard) demos.
- `opencv-python` — for video/image and the profiler HUD.
- `Pillow` — for image loading and EXIF in the profiler.
- `ultralytics` — for YOLO object detection in the profiler.

**First run of the profiler:** Ultralytics will download the YOLO weights (`yolov8n.pt`) automatically on first inference; allow network access.

### 6. Verify installation

```bash
python -c "
import numpy, matplotlib, sklearn, cv2, PIL, ultralytics
print('All dependencies OK')
"
```

If you see `All dependencies OK`, you’re set. If you get `ModuleNotFoundError`, the missing package is in the error message — fix it with `pip install <package>`.

---

## Running the demos

All commands below assume you’re in the project folder and the virtual environment is **activated** (`(.venv)` in the prompt). Use `python` or `python3` depending on how your system is set up (often they’re the same when the venv is active).

---

### Demo 1: El Hechizo Matemático (cosine similarity)

```bash
python hechizo_matematico.py
```

- **Console:** Cosine similarity, angle θ, “Probabilidad de Amarre”, and status messages.
- **Window:** 3D plot of two vectors and the angle between them (close angle → “MATCH INMINENTE”).

**If no window appears:** You may need a GUI (display). On macOS/Linux with a display it should work. For headless servers, you’d need a different matplotlib backend (e.g. `Agg`) or run only the math part without `plot_vectors_3d`.

---

### Demo 2: Sociological Profiling (image / webcam)

**Option A — Webcam (live):**

```bash
python sociological_profiler.py --webcam
```

- Opens the default camera (usually `0`).
- Shows “Scanning…” HUD, bounding boxes with “vibe” labels, and confidence bars.
- Press **`q`** in the OpenCV window to quit.

**Option B — Single image (with EXIF):**

```bash
python sociological_profiler.py --image path/to/your/photo.jpg
```

- Replace `path/to/your/photo.jpg` with a real path (e.g. `~/Desktop/photo.jpg` or `./sample.jpg`).
- Runs detection and shows EXIF (GPS, camera model, timestamp) when present.
- Press any key in the OpenCV window to close.

**Use a different webcam (e.g. external):**

```bash
python sociological_profiler.py --webcam --camera 1
```

---

### Demo 3: Elo Ranking & Network Meritocracy

```bash
python elo_ranking_sim.py
```

- **Concept:** The dating app user base is a network of nodes. A **Match** is a successful transaction; a **Dislike** is a latency/timeout.
- **Console (Rich):** Live-updating dashboard with:
  - **Top 10 Profiles (Nodes)** by Elo.
  - **System Entropy** (SRE health).
  - **Real-time transaction log** (Match vs Dislike and score deltas).
- High-Elo nodes behave as High-Priority nodes; when they interact with Low-Priority nodes, score adjustment is exponential. A **Load Balancer** prevents the simulation from converging into a single super-node.

---

## Troubleshooting

### “No module named 'cv2'” or “No module named 'PIL'”

You’re not in the venv, or dependencies weren’t installed. Do:

```bash
source .venv/bin/activate   # or Windows equivalent
pip install -r requirements.txt
```

Then run again with `python sociological_profiler.py ...` (or `python hechizo_matematico.py`).

---

### “No module named 'ultralytics'”

Install it explicitly:

```bash
pip install ultralytics>=8.0.0
```

Then run the profiler again.

---

### Webcam: “Could not open webcam” or black window

- **macOS:** Grant camera permission to Terminal (or Cursor/VS Code) in **System Settings → Privacy & Security → Camera**.
- Try another device: `python sociological_profiler.py --webcam --camera 1`.
- On Linux, ensure the user is in the `video` group and that no other app is holding the camera.

---

### YOLO: “Downloading yolov8n.pt” or slow first run

Normal. Ultralytics downloads the model (~6 MB) on first use. Wait for the download to finish; later runs use the cached file.

---

### Hechizo Matemático: matplotlib window doesn’t show

- Close other windows or dialogs that might be stealing focus.
- If you’re on a remote/headless machine, the script uses an interactive backend; for servers you’d need to switch to a non-GUI backend (e.g. `Agg`) and save to file instead of `plt.show()`.

---

### “Permission denied” or “Command not found”

- Use the full path to your venv’s Python if needed:
  ```bash
  /Users/laptopmac/hechizo_matematico_demo/.venv/bin/python hechizo_matematico.py
  ```
- On Windows:
  ```cmd
  .venv\Scripts\python.exe hechizo_matematico.py
  ```

---

### Image path with spaces

Put the path in quotes:

```bash
python sociological_profiler.py --image "/path/with spaces/photo.jpg"
```

---

## Project layout

```
hechizo_matematico_demo/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── hechizo_matematico.py     # Demo 1: Cosine similarity + 3D plot
├── elo_ranking_sim.py        # Demo 3: Elo ranking as SRE infrastructure
├── sociological_profiler.py # Demo 2: Vibe/EXIF profiling (YOLO + OpenCV)
└── .venv/                    # Virtual environment (create with python3 -m venv .venv)
```

---

## Quick reference

| Task | Command |
|------|--------|
| Create venv | `python3 -m venv .venv` |
| Activate venv (macOS/Linux) | `source .venv/bin/activate` |
| Install deps | `pip install -r requirements.txt` |
| Run Hechizo Matemático | `python hechizo_matematico.py` |
| Run Elo Ranking sim | `python elo_ranking_sim.py` |
| Run profiler (webcam) | `python sociological_profiler.py --webcam` |
| Run profiler (image) | `python sociological_profiler.py --image photo.jpg` |
| Quit webcam view | Press `q` in the OpenCV window |

If you’re still stuck, share the **exact command** you run and the **full error message** (or a screenshot) so we can narrow it down.
