import os
import glob
import shutil
import subprocess
from config import TEMP_DIR, QUALITY

def get_scene_names(script_path: str) -> list:
    names = []
    with open(script_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("class ") and "Scene" in stripped and "(" in stripped:
                name = stripped.split("class ")[1].split("(")[0].strip()
                names.append(name)
    print(f"[Stage 3] Detected scenes: {names}")
    return names

def render_scene(script_path: str, scene_name: str) -> str:
    print(f"[Stage 3] Rendering: {scene_name}")
    out_dir = os.path.join(TEMP_DIR, "manim_output")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "manim", "render",
        "-q", QUALITY,               # h = 1080p, p = 4K, l = 480p
        "--media_dir", out_dir,
        "--disable_caching",
        "--renderer", "cairo",
        script_path,
        scene_name,
    ]

    print(f"[Stage 3] CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"[Stage 3] FAILED: {scene_name}")
        print("STDERR:", result.stderr[-1500:])
        print("STDOUT:", result.stdout[-500:])
        return None

    # Find the rendered mp4
    pattern = os.path.join(out_dir, "**", "*.mp4")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("[Stage 3] No mp4 found after render.")
        return None

    latest = max(files, key=os.path.getmtime)
    print(f"[Stage 3] Rendered: {latest}")
    return latest

def concatenate_scenes(video_paths: list, output_path: str) -> str:
    if len(video_paths) == 1:
        shutil.copy(video_paths[0], output_path)
        return output_path

    list_path = os.path.join(TEMP_DIR, "concat.txt")
    with open(list_path, "w") as f:
        for p in video_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy", output_path
    ], check=True, capture_output=True)

    print(f"[Stage 3] Concatenated {len(video_paths)} → {output_path}")
    return output_path

def render_all(script_path: str) -> str:
    scenes = get_scene_names(script_path)

    if not scenes:
        raise RuntimeError("[Stage 3] No scene classes found in script.")

    rendered = []
    for name in scenes:
        path = render_scene(script_path, name)
        if path:
            rendered.append(path)

    if not rendered:
        # Print script for debugging
        print("[Stage 3] All scenes failed. Script contents:")
        with open(script_path) as f:
            print(f.read())
        raise RuntimeError("[Stage 3] All scenes failed to render.")

    out = os.path.join(TEMP_DIR, "raw_animation.mp4")
    return concatenate_scenes(rendered, out)
