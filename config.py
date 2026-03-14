import os

# ── Models ────────────────────────────────────────────────────────────────────
CLASSIFIER_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
CODEGEN_MODEL       = "Qwen/Qwen2.5-Coder-14B-Instruct"
WHISPER_MODEL       = "base"

# ── Generation ────────────────────────────────────────────────────────────────
CODEGEN_MAX_TOKENS  = 6000
CODEGEN_TEMPERATURE = 0.2
MAX_RETRIES         = 3

# ── Video ─────────────────────────────────────────────────────────────────────
VIDEO_WIDTH         = 1080
VIDEO_HEIGHT        = 1920
FPS                 = 30
QUALITY             = "h"
MAX_VIDEO_DURATION  = 75      # hard cap in seconds — longer renders get trimmed

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR   = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")

os.makedirs(TEMP_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TTS / Audio ───────────────────────────────────────────────────────────────
TTS_VOICE          = "af_bella"
TTS_LANGUAGE       = "en-us"
TTS_SPEED          = 1.0      # natural speed — no stretching
BURN_SUBTITLES     = False    # subtitles disabled — animation text carries the explanation

# ── Kokoro model paths ────────────────────────────────────────────────────────
KOKORO_MODEL       = "/workspace/kokoro-v1.0.onnx"
KOKORO_VOICES      = "/workspace/voices-v1.0.bin"
