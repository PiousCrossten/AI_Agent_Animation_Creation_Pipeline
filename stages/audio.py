import os
import gc
import subprocess
import torch
import whisper
import soundfile as sf
from kokoro_onnx import Kokoro
from config import TEMP_DIR, WHISPER_MODEL, KOKORO_MODEL, KOKORO_VOICES, TTS_VOICE, TTS_LANGUAGE

TTS_SPEED          = getattr(__import__('config'), 'TTS_SPEED', 1.0)
MAX_VIDEO_DURATION = getattr(__import__('config'), 'MAX_VIDEO_DURATION', 75)


def _get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def generate_narration(script: str) -> str:
    print("[Stage 4] Generating narration...")
    kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    samples, sr = kokoro.create(
        script, voice=TTS_VOICE, speed=TTS_SPEED, lang=TTS_LANGUAGE
    )
    path = os.path.join(TEMP_DIR, "narration.wav")
    sf.write(path, samples, sr)
    del kokoro
    gc.collect()
    dur = len(samples) / sr
    print(f"[Stage 4] Narration: {dur:.1f}s at natural speed")
    return path


def generate_subtitles(audio_path: str) -> str:
    """Transcribe audio to SRT — kept for optional use, not burned into video."""
    print(f"[Stage 4] Transcribing with Whisper ({WHISPER_MODEL})...")
    model  = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, word_timestamps=True)
    srt_path = os.path.join(TEMP_DIR, "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            text = seg["text"].strip()
            if not text:
                continue
            f.write(f"{i}\n{_fmt(seg['start'])} --> {_fmt(seg['end'])}\n{text}\n\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Stage 4] Subtitles saved (not burned in): {srt_path}")
    return srt_path


def _fmt(s: float) -> str:
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec, ms = int(s % 60), int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def mix_audio_with_video(video: str, audio: str, srt: str) -> str:
    """
    Mix narration into video.
    Rules:
      - Audio plays at NATURAL speed — absolutely no stretching
      - Video will be trimmed to MAX_VIDEO_DURATION by export.py
      - Working length = min(video_dur, MAX_VIDEO_DURATION)
      - If audio < working length: pad end with silence (narration finishes, animation continues)
      - If audio > working length: trim audio to working length
      - NO subtitles burned in
    """
    print("[Stage 4] Mixing audio into video (natural speed, no subtitles)...")

    video_dur    = _get_duration(video)
    audio_dur    = _get_duration(audio)
    working_dur  = min(video_dur, MAX_VIDEO_DURATION)

    print(f"[Stage 4] Video: {video_dur:.1f}s | Audio: {audio_dur:.1f}s | Working: {working_dur:.1f}s")

    out = os.path.join(TEMP_DIR, "video_with_audio.mp4")

    if audio_dur >= working_dur:
        # Audio covers full duration — trim audio to working length
        print(f"[Stage 4] Audio covers full duration, trimming audio to {working_dur:.1f}s")
        cmd = [
            "ffmpeg", "-y",
            "-i", video,
            "-i", audio,
            "-map", "0:v",
            "-map", "1:a",
            "-t", f"{working_dur:.3f}",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out,
        ]
    else:
        # Audio shorter than working length — pad remainder with silence
        silence_dur = working_dur - audio_dur
        print(f"[Stage 4] Audio {audio_dur:.1f}s < working {working_dur:.1f}s — padding {silence_dur:.1f}s silence")
        cmd = [
            "ffmpeg", "-y",
            "-i", video,
            "-i", audio,
            "-filter_complex",
            f"[1:a]apad=whole_dur={working_dur:.3f}[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-t", f"{working_dur:.3f}",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Stage 4] Mix error — simple fallback:\n{result.stderr[-500:]}")
        # Simplest possible fallback — just overlay audio, trim to shortest
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video,
            "-i", audio,
            "-map", "0:v",
            "-map", "1:a",
            "-t", f"{working_dur:.3f}",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out,
        ], check=True, capture_output=True)

    final_dur = _get_duration(out)
    print(f"[Stage 4] Mixed output: {final_dur:.1f}s — {out}")
    return out