import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
import shutil
import traceback
import torch

from stages.understand import understand_prompt
from stages.codegen    import generate_manim_code
from stages.render     import render_all
from stages.audio      import generate_narration, generate_subtitles, mix_audio_with_video
from stages.export     import export_final
from config            import TEMP_DIR


def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    print(f"[VRAM] Free: {free/1e9:.2f} GB / {total/1e9:.2f} GB total")


def cleanup_temp():
    for item in os.listdir(TEMP_DIR):
        p = os.path.join(TEMP_DIR, item)
        try:
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except Exception as e:
            print(f"[Cleanup] Could not remove {p}: {e}")
    print("[Cleanup] Temp cleared.")


def run_pipeline(user_prompt: str) -> str:
    print(f"\n{'='*60}")
    print(f" INPUT: {user_prompt}")
    print(f"{'='*60}\n")

    # Stage 1
    print("[Pipeline] Stage 1: Understanding prompt...")
    concept = understand_prompt(user_prompt)
    clear_vram()   # flush classifier VRAM before loading codegen

    # Stage 2
    print("\n[Pipeline] Stage 2: Generating Manim code...")
    script = generate_manim_code(concept)
    clear_vram()

    # Stage 3
    print("\n[Pipeline] Stage 3: Rendering animation...")
    raw_video = render_all(script)
    clear_vram()

    # Stage 4
    print("\n[Pipeline] Stage 4: Narration + subtitles...")
    audio = generate_narration(concept["narration_script"])
    srt   = generate_subtitles(audio)
    mixed = mix_audio_with_video(raw_video, audio, srt)
    clear_vram()

    # Stage 5
    print("\n[Pipeline] Stage 5: Final export...")
    final = export_final(mixed, concept["topic"])

    cleanup_temp()

    print(f"\n{'='*60}")
    print(f" OUTPUT: {final}")
    print(f"{'='*60}\n")
    return final


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your topic here"')
        sys.exit(1)
    try:
        out = run_pipeline(" ".join(sys.argv[1:]))
        print(f"Video ready: {out}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)