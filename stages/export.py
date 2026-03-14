import os
import subprocess
from datetime import datetime
from config import OUTPUT_DIR, VIDEO_WIDTH, VIDEO_HEIGHT, FPS

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


def export_final(video_path: str, topic: str) -> str:
    print("[Stage 5] Final export...")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe     = "".join(c if c.isalnum() else "_" for c in topic)[:40]
    out_path = os.path.join(OUTPUT_DIR, f"{safe}_{ts}.mp4")

    # Measure actual duration and apply hard cap
    video_dur = _get_duration(video_path)
    trim_dur  = min(video_dur, MAX_VIDEO_DURATION)
    print(f"[Stage 5] Source: {video_dur:.1f}s → Output: {trim_dur:.1f}s (cap={MAX_VIDEO_DURATION}s)")

    # Scale + pad to exact 9:16 portrait
    vf = (
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
    )

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", f"{trim_dur:.3f}",      # hard trim to cap
        "-vf", vf,
        "-r", str(FPS),
        "-c:v", "libx264", "-crf", "18",
        "-preset", "slow",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Stage 5] Export error:\n{result.stderr}")
        raise RuntimeError("Final export failed.")

    final_dur = _get_duration(out_path)
    size      = os.path.getsize(out_path) / 1e6
    print(f"[Stage 5] Done → {out_path} ({final_dur:.1f}s, {size:.1f} MB)")
    return out_path