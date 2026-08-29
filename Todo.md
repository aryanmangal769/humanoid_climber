# Humanoid Climber plan

## Current status — August 29, 2026

- [x] Isolated Python 3.12 environment with `uv` and MjLab 1.6.0.
- [x] Stock G1 flat-walking checkpoint plays successfully on the Mac.
- [x] Low-friction task created with training range 0.1–1.0 and fixed evaluation friction 0.2.
- [x] Same stock policy evaluated at friction 0.2: it walks sideways slowly and attempts to recover balance, but falls under a fast forward command.
- [ ] Decide the next experiment before spending GPU credits.

Work in humanoid_climber. Treat it as your project (ice task, scripts, notes). Do not dump the whole mjlab tree into it.
The other Mac helps watch and edit. It does not replace Hugging Face for training: mjlab train still wants Linux + NVIDIA. A faster Mac only makes play less painful.

Machine split
Where
What
This Mac / better Mac
Clone repo, uv, demo, play one G1, edit friction
Hugging Face Jobs (credits)
Train / finetune only if you need a new .pt

Use the same git repo on both Macs. Checkpoints live on Hugging Face, not in git.

Step 0 — Isolate Python (once per Mac)
Install uv if needed. Never pip install into system Python.
git clone https://github.com/aryanmangal769/humanoid_climber.git
cd humanoid_climber
Add mjlab as a dependency (uv add / pyproject.toml), or a git submodule pinned to one commit. Prefer that over forking all of mjlab.
Pass: uvx --from mjlab --refresh demo opens a robot. If that works, the Mac is fine.

Step 1 — Try an existing walk (no credits)
Download robomotic/mjlab-policies g1_velocity/.../model_final.pt.
uv run play Mjlab-Velocity-Flat-Unitree-G1 \
 --checkpoint-file ./ckpt/model_final.pt \
 --num-envs 1 --viewer viser
Pass: G1 walks in the browser.
Fail: crash / obs-size error → skip to Step 2. Do not tweak ice yet.

Step 2 — Only if walk failed: train a basic walk on HF
Job on A100 (or A10G), CUDA 12.4 image, MUJOCO_GL=egl, --timeout many hours, upload model_*.pt to the Hub before the job dies.
Train stock Mjlab-Velocity-Flat-Unitree-G1 until it walks on flat, normal friction.
Pass: download that .pt to the Mac and play like Step 1.
This is the only step that spends credits. Stop if Step 1 already walked.

Step 3 — Low friction (ice)
In your repo, copy the G1 velocity task and change foot friction (train range about 0.1–1.0, eval about 0.2).
Finetune from the walking .pt, do not train ice from scratch.
Then play on the Mac with the ice task + new checkpoint.
Pass: G1 still walks (slower, more slip) instead of falling immediately.

Step 4 — Later (not now)
Get-up, slope, wind, rope. Each is a new task on top of a walker that already works on ice.

Order to protect your time
Demo on Mac
Load Hub walk policy
HF train only if (2) fails
Ice finetune
Harder Himalaya stuff
Do not fork mjlab and do not start ice until a flat walk plays on the Mac. The better Mac is for steps 0–1 and 3 playback; Hugging Face is only step 2 (and the ice finetune GPU run in step 3).

