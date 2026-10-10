# Digital Wizard’s Toolkit 🐙💻🔮

Welcome to your tiny **wizard lab for data, vibes, and networks**.  
This repo is a **Digital Wizard’s Toolkit**: three friendly spells (plus a bonus fourth!) you can run from the terminal without needing to be a grown‑up engineer.

- 🧪 **Spell 1 – The Love Compass** (`hechizo_matematico.py`)  
  Two invisible arrows in space, one for each person. The spell measures **how close their directions are** and draws them in 3D. Smaller angle = more “hmm… interesting…” 💘

- 🎭 **Spell 2 – The Digital Eye** (`sociological_profiler.py`)  
  A camera spell that looks at a **photo or webcam feed**, spots things (YOLO model), and reads the **hidden metadata** (EXIF) like when/where the picture was taken. Think of it as a gossiping crystal ball for images 🔮.

- 📈 **Spell 3 – The Popularity Game** (`elo_ranking_sim.py`)  
  A pretend dating app full of tiny nodes (people). Every interaction changes their **Elo score** like in online games. High‑Elo profiles behave like **VIP servers** in an SRE system, with a **load balancer** making sure no one becomes a scary all‑powerful super‑node 🛡️.

- 🚀 **Bonus Spell – The HA Love Cluster** (`dating_sre_simulation.py`)  
  A high‑availability twist on the Popularity Game! A dating app simulated as a **distributed system** with network effects, traffic‑aware load balancing, tier promotions, and system entropy. Watch nodes race to the **Desirability Peak** 🏔️.

All four spells are written in **Python**, and you run them from the terminal like a command‑line spellbook 🚀.

### ✨ Try it on the web ✨

Don't want to install anything yet? There is now a **live web version** at:

**<https://ch1llc0d3.github.io/hechizo_matematico_demo/>**

It hosts an interactive **browser version of The Love Compass** — no Python, no terminal, just open the link and start exploring cosine similarity in your browser 🌐💘.

---

## The Spells (Demos) 🔮

### 1. The Love Compass – `hechizo_matematico.py`

Imagine each person is a **direction in 3D space**.  
This spell:

- Turns simple profile data into **3D vectors**.
- Calculates how similar they are using **cosine similarity**.
- Shows:
  - A **number** (how close they are),
  - An **angle** (big angle = “meh”, tiny angle = “👀”),
  - A cute label like “MATCH INMINENTE”.
- Pops up a **3D plot window** where you can **see the two arrows and the angle between them**.

You can think of it as a **mathematical crush detector**.

---

### 2. The Digital Eye – `sociological_profiler.py`

This spell is your **AI camera friend**:

- With `--webcam`, it:
  - Opens your camera,
  - Draws boxes around things it sees,
  - Shows labels and confidence bars like a sci‑fi HUD 💻.
- With `--image some_photo.jpg`, it:
  - Looks at a single image,
  - Reads **EXIF** (when, where, which camera),
  - Overlays detections and info on the picture.

Think of it as a **smart crystal ball** that can say:  
“I see a person, a laptop, and this photo was taken at 2022‑08‑01, somewhere near here…” 🔮

> Note: This is **educational only**, not a real profiling tool.

---

### 3. The Popularity Game – `elo_ranking_sim.py`

Now we zoom out and pretend we’re SREs for a dating app:

- Every user is a **node** with an **Elo score**.
- Matches and dislikes are like **network events**:
  - **Match** = successful request,
  - **Dislike** = timeout/latency.
- High‑Elo nodes behave like **high‑priority servers**:
  - When VIPs talk to newbies, updates are **stronger (exponential K)**.
- A **load balancer** chooses which nodes talk to which,
  so the system doesn’t collapse into one all‑powerful super‑node.

In the terminal you get a live **Rich dashboard**:

- Top 10 nodes by Elo,
- “System entropy” (how spread out the scores are),
- A scrolling log of every interaction.

It’s like watching a **tiny internet of people** trying to be popular, with SRE logic behind the scenes 🛡️.

---

### 4. The HA Love Cluster (Bonus!) – `dating_sre_simulation.py`

This is the **evolved form** of the Popularity Game 🦋 — same core idea (users as nodes, Elo scoring, Rich live dashboard) but with extra **SRE wizardry** baked in:

- **Network Effect multiplier:** When a high‑Elo node interacts with a low‑Elo node, both Elo deltas are multiplied by `exp(gap/400)` (clamped to 3.0). Big desirability gaps create **bigger ranking shifts** — drama included 🌋.
- **Traffic‑aware load balancer:** When picking a match candidate, nodes that already got lots of traffic are **down‑weighted** (penalty `1/(1 + 0.3*traffic)`), so super‑nodes and influencers don't monopolize every match ⚖️.
- **Tier system:** Tier 1 = Elo ≥ 1800 (“Desirability Peak”), Tier 2 = ≥ 1600, Tier 3 = everyone else. A live **Promotion Log** shows nodes crossing into Tier 1 in real time 🏆.
- **System entropy:** Shannon entropy of the interaction distribution — a measure of **match diversity**. Higher = healthier cluster 📊.

**How is it different from the Popularity Game?**
Spell 3 is the gentle introduction. This bonus spell cranks the SRE metaphors up to eleven: network‑effect multipliers, traffic‑penalised load balancing, tier thresholds, and promotion detection — all visible on a richer dashboard with **Top 10 table, Metrics panel, and Promotion Log**.

Runs **30 nodes / 200 steps** with a live Rich dashboard.

---

## The Recipe: Setting Up Your Magic Kitchen 🧙‍♀️🍳

We’ll set things up **once**, then you can run any spell easily.

### Step 1 – Go to the wizard’s lair (project folder)

Open your terminal and go to the project folder:

```bash
cd ~/hechizo_matematico_demo
```

If your folder lives somewhere else, change the path accordingly.

---

### Step 2 – Create a private room for your code (virtual environment)

A **virtual environment** is like a **private room** where your spells keep their own toys (Python packages) without fighting with the rest of your computer.

Run:

```bash
python3 -m venv .venv
```

This creates a hidden folder called `.venv` that holds your spell ingredients.

If `.venv` already exists and you want to start fresh, you can delete or rename it first, then run the command again.

---

### Step 3 – Open the magic room (activate the venv)

You must **activate** the private room so every `python` and `pip` command uses it.

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

If it worked, your terminal line will start with something like:

```text
(.venv) your-computer-name %
```

That `(.venv)` means **“you are inside the magic room now”** 🪄.

---

### Step 4 – Give your tools a quick upgrade

We upgrade `pip` so installs are smoother:

```bash
pip install --upgrade pip
```

---

### Step 5 – Pour in all the ingredients (install dependencies)

From inside the project folder, with the venv **activated**:

```bash
pip install -r requirements.txt
```

This installs:

- `numpy`, `matplotlib`, `scikit-learn` → math & plotting for **The Love Compass**.
- `rich` → pretty terminal dashboards for **The Popularity Game** and **The HA Love Cluster**.
- `opencv-python` (`cv2`) → camera & image windows for **The Digital Eye**.
- `Pillow` (`PIL`) → reading images and EXIF.
- `ultralytics` → YOLO model for object detection.

💡 On the **first run** of the Digital Eye, `ultralytics` will download a file called `yolov8n.pt`. That’s just the trained model. It might take a moment; that’s normal.

---

### Step 6 – Quick “Did we break anything?” test

Run this small check:

```bash
python -c "
import numpy, matplotlib, sklearn, cv2, PIL, ultralytics
print('All dependencies OK')
"
```

If you see:

```text
All dependencies OK
```

you’re ready to cast spells 🚀.  
If you see `ModuleNotFoundError`, it will tell you which package is missing. You can usually fix it with:

```bash
pip install name_of_the_missing_package
```

---

## How to Cast Each Spell 🔮

All of these commands assume:

- You are **inside** the project folder (`cd ~/hechizo_matematico_demo`), and  
- Your virtual environment is **activated** (`(.venv)` visible in the terminal).

If your system uses `python3` instead of `python`, just swap the word.

---

### Spell 1: The Love Compass 💘 (`hechizo_matematico.py`)

```bash
python hechizo_matematico.py
```

What happens:

- The terminal prints:
  - The cosine similarity,
  - The angle between the two profiles,
  - A “Probabilidad de Amarre” (love probability) percentage.
- A new window opens showing:
  - Two colorful arrows in 3D,
  - The angle between them,
  - Extra text when they are very close (MATCH!).

If **no window appears**:

- You’re probably on a machine without a display (like a remote server), or your OS is blocking windows.
- On normal macOS / Linux desktops, it should just appear.

---

### Spell 2: The Digital Eye 👁💻 (`sociological_profiler.py`)

#### Option A – Live webcam mode

```bash
python sociological_profiler.py --webcam
```

You should see:

- A camera window popping up,
- Boxes around things the model sees (people, objects, etc.),
- Labels + confidence bars like a sci‑fi HUD.

To **quit**, click on the camera window and press **`q`** on your keyboard.

#### Option B – Single image mode

```bash
python sociological_profiler.py --image path/to/your/photo.jpg
```

Replace `path/to/your/photo.jpg` with a real image path, like:

- `~/Desktop/photo.jpg`
- `./sample.jpg`

The spell will:

- Show the image with boxes and labels,
- Try to read and print EXIF info (time, camera, maybe GPS),
- Close when you press any key in the window.

#### Option C – Use a different webcam

If you have more than one camera (e.g., laptop cam + external USB cam):

```bash
python sociological_profiler.py --webcam --camera 1
```

Try `--camera 0`, `--camera 1`, etc. until the right one works.

---

### Spell 3: The Popularity Game 📊🛡️ (`elo_ranking_sim.py`)

```bash
python elo_ranking_sim.py
```

You’ll see a **Rich dashboard** in your terminal:

- Top 10 profiles by Elo score,
- A bar showing **system entropy** (how evenly scores are spread),
- A live log showing:
  - Who interacted with whom,
  - Whether it was a Match or Dislike,
  - How much each Elo changed.

Just let it run and watch how the system evolves.  
When it finishes, it prints “Simulation finished.” You can run it again as many times as you like 🚀.

---

### Spell 4 (Bonus): The HA Love Cluster 🚀🏔️ (`dating_sre_simulation.py`)

```bash
python dating_sre_simulation.py
```

You’ll see a **Rich dashboard** with even more SRE flair:

- **Top 10 High-Priority Nodes** table (Elo score, Tier, interaction count),
- **Metrics panel** showing system entropy (match diversity),
- **Promotion Log** announcing nodes that climb into Tier 1 (Desirability Peak).

Let the cluster run its 200 steps and watch the tiers shake out. Run it as many times as you like — every run is a new tiny internet 🚀.

---

## Turbo Mode: Super‑Quick Cheat Sheet 🚀

If you’re in a hurry and already kind of know what a terminal is, this section is for you.

### One‑time setup

```bash
cd ~/hechizo_matematico_demo
python3 -m venv .venv
source .venv/bin/activate           # On macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### Every time you come back

```bash
cd ~/hechizo_matematico_demo
source .venv/bin/activate           # Or Windows equivalent
```

### Run the spells

```bash
python hechizo_matematico.py                # Spell 1 – Love Compass
python elo_ranking_sim.py                   # Spell 3 – Popularity Game
python dating_sre_simulation.py             # Spell 4 – HA Love Cluster (bonus!)
python sociological_profiler.py --webcam    # Spell 2 – Digital Eye (webcam)
python sociological_profiler.py --image photo.jpg
```

To quit the webcam spell: **press `q`** in the camera window.

---

## Fixing the Booboos: What If It Breaks? 🛡️

Even the best wizards get error messages. Here’s a friendly guide.

### “No module named 'cv2'” or “No module named 'PIL'”

This usually means:

- You **forgot to activate** your virtual environment, or  
- You **didn’t install** the ingredients yet.

Try:

```bash
source .venv/bin/activate   # or the Windows equivalent
pip install -r requirements.txt
```

Then run your spell again:

```bash
python sociological_profiler.py --webcam
# or
python hechizo_matematico.py
```

---

### “No module named 'ultralytics'”

Sometimes `ultralytics` didn’t install correctly the first time.

Run this inside your activated venv:

```bash
pip install ultralytics>=8.0.0
```

Then try the Digital Eye again:

```bash
python sociological_profiler.py --webcam
```

---

### Webcam: “Could not open webcam” or just a black window

Possible reasons and fixes:

- **macOS:** The OS might be blocking camera access.  
  Go to **System Settings → Privacy & Security → Camera** and allow access for your terminal app (Terminal, iTerm, Cursor, VS Code, etc.).
- Another program might already be using the camera. Close video apps (Zoom, Meet, etc.) and try again.
- You might be on the wrong camera index. Try:

  ```bash
  python sociological_profiler.py --webcam --camera 1
  python sociological_profiler.py --webcam --camera 2
  ```

- On Linux, make sure your user can use the camera (usually by being in the `video` group).

---

### YOLO: “Downloading yolov8n.pt” or first run is very slow

Totally normal 🐙.

The first time you run the Digital Eye, `ultralytics` downloads the YOLO model file (`yolov8n.pt`, around a few MB).  
Just let it finish; later runs will reuse the cached file and start much faster.

---

### Hechizo Matemático: matplotlib window doesn’t show

If you run:

```bash
python hechizo_matematico.py
```

and **nothing pops up**:

- Make sure you’re on a computer with a **GUI/desktop** (not just SSH into a server).
- Check if the window maybe opened behind other windows.
- On servers/headless machines, this script uses an interactive backend that needs a display.  
  Advanced option (for power users): switch matplotlib to a non‑GUI backend (like `Agg`) and save the plot to an image file instead of using `plt.show()`.

---

### “Permission denied” or “Command not found”

These often mean:

- Your system doesn’t know which `python` to use, or  
- The venv Python isn’t being used.

You can bypass that by calling the Python inside your venv directly:

```bash
~/hechizo_matematico_demo/.venv/bin/python hechizo_matematico.py
```

On Windows:

```cmd
.venv\Scripts\python.exe hechizo_matematico.py
```

If that works, then your venv is fine; you just need to make sure it’s activated before running commands.

---

### Image path with spaces

If your image file lives in a path with spaces, **wrap it in quotes**:

```bash
python sociological_profiler.py --image "/path/with spaces/photo.jpg"
```

Without quotes, the terminal thinks the path is several different words and gets confused 😅.

---

## Map of the Lair (Project Layout) 🐙

```text
hechizo_matematico_demo/
├── README.md                  # This magical guide you’re reading
├── LICENSE                    # MIT license (ch1llc0d3, 2026)
├── requirements.txt           # List of spell ingredients (Python packages)
├── hechizo_matematico.py      # Spell 1: The Love Compass (3D cosine magic)
├── elo_ranking_sim.py         # Spell 3: The Popularity Game (Elo + SRE vibes)
├── dating_sre_simulation.py   # Spell 4: The HA Love Cluster (bonus! SRE dating sim)
├── sociological_profiler.py   # Spell 2: The Digital Eye (YOLO + EXIF)
└── .venv/                     # Your private magic room (virtual environment)
```

---

If something still feels scary, remember:  
**You’re just typing short magic sentences (commands) into a box.**  
Nothing here is meant to break your computer, only to light up your inner data wizard 💻🔮🚀.
