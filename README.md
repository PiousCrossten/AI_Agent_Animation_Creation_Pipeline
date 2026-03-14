# AI Animation Pipeline

**AI Animation Pipeline** is an AI agent that turns a short topic description into a full educational video: scripted narration, animation (Manim CE), and final export — all from a single text prompt.

Example:  
`python agent.py "Bubble Sort"` → a ~1-minute portrait video (1080×1920) with animation and voiceover.

---

## What It Does

1. **Understands** your topic (LLM produces a structured “production plan”: domain, animation type, narration script, scene plan).
2. **Generates** Manim CE Python code from that plan (second LLM, with type-specific templates).
3. **Renders** the animation (Manim + Cairo).
4. **Adds** narration (Kokoro TTS) and optional subtitles (Whisper), then mixes with FFmpeg.
5. **Exports** a final video (trimmed, 9:16, H.264) to `outputs/`.

No LangChain or agent toolkits — just a linear pipeline: Hugging Face Transformers, Manim, Kokoro, Whisper, and FFmpeg.

For an easy-to-understand explanation of how this AI agent works and why it’s built this way, see **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## Why This Approach

| Choice | Reason |
|--------|--------|
| **Two specialized LLMs** | One model plans *what* to show (narration + scene layout); a code model writes *how* (Manim). Keeps each task clear and reduces “generic” code. |
| **Staged pipeline with VRAM clear** | Classifier (7B) and codegen (14B) are 4-bit quantized and loaded one at a time; VRAM is cleared between stages so a single consumer GPU can run the full pipeline. |
| **Animation-type templates** | The code model is steered by full Manim examples (graph, geometry, particle, stepwise, hybrid). Output stays on-layout (e.g. portrait zones) and avoids invalid APIs. |
| **Post-processing of generated code** | Colors, deprecated calls (e.g. `ShowCreation` → `Create`), and structural fixes are applied so the script runs under Manim CE v0.18 without manual edits. |
| **No agent loop** | Deterministic stages are easier to debug and reproduce than an agent that calls tools in a loop. |

---

## Architecture (High Level)

```
user prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1 — Understand (Qwen2.5-7B-Instruct, 4-bit)                │
│   → JSON: topic, domain, animation_type, narration_script,       │
│           scenes[].visual_plan, formulas, etc.                   │
└─────────────────────────────────────────────────────────────────┘
    │  clear VRAM
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2 — Codegen (Qwen2.5-Coder-14B-Instruct, 4-bit)            │
│   → Python script (Manim CE) from plan + type-specific template  │
│   → Retries on syntax/short duration; post-fixes for colors/API   │
└─────────────────────────────────────────────────────────────────┘
    │  clear VRAM
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3 — Render (Manim CE + Cairo)                              │
│   → manim render per scene → concatenate MP4s (FFmpeg)           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4 — Audio (Kokoro TTS + Whisper)                           │
│   → Narration WAV from narration_script → SRT (optional)         │
│   → FFmpeg: mix audio into video, pad/trim to max duration       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5 — Export (FFmpeg)                                        │
│   → Trim to MAX_VIDEO_DURATION, scale/pad to 1080×1920, encode   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
outputs/<Topic>_<timestamp>.mp4
```

---

## Requirements Overview

| Category | What |
|----------|------|
| **Python** | 3.10+ (3.11 recommended) |
| **GPU** | NVIDIA GPU with ~12 GB+ VRAM recommended (for 4-bit 7B + 14B). Can run CPU-only but slow. |
| **System** | FFmpeg, LaTeX (for Manim `MathTex`), Cairo/Pango (for Manim CE). |
| **Models** | Classifier and codegen are downloaded from Hugging Face on first run. Kokoro TTS model files must be downloaded once (see below). |

---

## Setup

### 1. System dependencies (Linux example)

Install FFmpeg, LaTeX, and Cairo/Pango so Manim can render:

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install -y \
  ffmpeg \
  libcairo2-dev libpango1.0-dev pkg-config \
  texlive texlive-latex-extra texlive-fonts-extra texlive-science \
  dvipng cm-super
```

On macOS: `brew install ffmpeg cairo pango texlive` (or equivalent).

### 2. Python environment

```bash
cd manim_agent_v4
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. PyTorch (GPU recommended)

For CUDA 12.x:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only, skip this and install from `requirements.txt` (PyTorch CPU will be installed).

### 4. Python packages

```bash
pip install -r requirements.txt
```

If you already installed PyTorch with CUDA above, `requirements.txt` will not override it.

### 5. Kokoro TTS model files

The pipeline expects Kokoro ONNX model and voices on disk. Download once:

```bash
# From project root (or set paths in config.py)
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Set their locations in `config.py`:

```python
KOKORO_MODEL  = "/path/to/kokoro-v1.0.onnx"
KOKORO_VOICES = "/path/to/voices-v1.0.bin"
```

### 6. Config (optional)

Edit `config.py` to change:

- **Models:** `CLASSIFIER_MODEL`, `CODEGEN_MODEL`, `WHISPER_MODEL`
- **Paths:** `KOKORO_MODEL`, `KOKORO_VOICES`, `OUTPUT_DIR`, `TEMP_DIR`
- **Video:** `VIDEO_WIDTH`, `VIDEO_HEIGHT`, `FPS`, `MAX_VIDEO_DURATION`, `QUALITY`
- **TTS:** `TTS_VOICE`, `TTS_LANGUAGE`, `TTS_SPEED`

---

## Run

```bash
python agent.py "Your topic here"
```

Example topics that work well:

- `Bubble Sort`
- `Pythagorean theorem`
- `Projectile motion`
- `Fourier series`
- `Binary search`

Output path is printed at the end, e.g. `outputs/Bubble_Sort_20260314_141311.mp4`.

### Verify installation (optional)

```bash
python3 -c "
packages = {'torch': 'torch', 'manim': 'manim', 'transformers': 'transformers',
            'accelerate': 'accelerate', 'bitsandbytes': 'bitsandbytes',
            'whisper': 'openai-whisper', 'kokoro_onnx': 'kokoro-onnx',
            'soundfile': 'soundfile', 'scipy': 'scipy', 'numpy': 'numpy', 'PIL': 'Pillow'}
for mod, pkg in packages.items():
    try:
        m = __import__(mod)
        print(f'  [OK]  {pkg}')
    except ImportError:
        print(f'  [FAIL] {pkg}')
"
# GPU (if you use CUDA):
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
manim --version
```

---

## Pipeline Stages (Reference)

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| 1 | `stages/understand` | User prompt string | Concept dict (topic, domain, animation_type, narration_script, scenes) |
| 2 | `stages/codegen` | Concept dict | `temp/generated_scene.py` (Manim script) |
| 3 | `stages/render` | Script path | `temp/raw_animation.mp4` (concatenated scenes) |
| 4 | `stages/audio` | Narration script + raw video | `temp/video_with_audio.mp4` |
| 5 | `stages/export` | Mixed video + topic name | `outputs/<Topic>_<timestamp>.mp4` |

Prompts live in `prompts/`: `classifier.txt` (Stage 1), `manim_codegen.txt` (Stage 2). Type-specific Manim examples are in `stages/codegen.py` (`EXAMPLES`).

---

## Troubleshooting

| Issue | What to check |
|------|----------------|
| **Out of VRAM** | Ensure only one large model is loaded at a time (pipeline clears VRAM between Stage 1 and 2). Reduce batch or use smaller Whisper model in `config.py`. |
| **Manim render fails** | Run `manim render` on `temp/generated_scene.py` manually and read the traceback. Often invalid color names or deprecated APIs — post-processing in `codegen.py` may need an extra fix. |
| **No narration / wrong voice** | Confirm `KOKORO_MODEL` and `KOKORO_VOICES` paths in `config.py` and that the files exist. Check `TTS_VOICE` (e.g. `af_bella`). |
| **FFmpeg errors** | Install `ffmpeg` and ensure it’s on `PATH`. For “concat” or filter errors, check that `temp/` contains the expected intermediate files. |
| **LaTeX errors in Manim** | Install a full `texlive` set (see system deps). Manim uses it for `MathTex`. |

---

## File Layout

```
manim_agent_v4/
├── agent.py              # Entry point; runs the 5-stage pipeline
├── config.py              # Models, paths, video/TTS settings
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── METHODOLOGY.md         # How the AI agent works (easy-to-understand)
├── prompts/
│   ├── classifier.txt     # Stage 1 system prompt (JSON plan)
│   └── manim_codegen.txt   # Stage 2 system prompt (Manim rules + zones)
├── stages/
│   ├── understand.py      # Stage 1: LLM → concept dict
│   ├── codegen.py         # Stage 2: concept + template → Manim script
│   ├── render.py          # Stage 3: Manim render + concat
│   ├── audio.py           # Stage 4: TTS, Whisper, mix
│   └── export.py          # Stage 5: trim, scale, encode
├── temp/                  # Created at run; intermediates (cleaned after run)
└── outputs/               # Final videos
```

---

## License / Credits

- **Manim** — [Manim Community](https://www.manim.community/) (v0.18).
- **Qwen** — [Qwen2.5](https://huggingface.co/Qwen) (Hugging Face).
- **Kokoro** — [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (TTS).
- **Whisper** — [OpenAI Whisper](https://github.com/openai/whisper).

Use of model weights and data is subject to their respective terms.
