import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import CLASSIFIER_MODEL, PROMPT_DIR

_model     = None
_tokenizer = None


def load_classifier():
    global _model, _tokenizer
    if _model is not None:
        return
    print(f"[Stage 1] Loading: {CLASSIFIER_MODEL}")
    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        CLASSIFIER_MODEL,
        quantization_config=q,
        device_map="auto",
    )
    _model.eval()
    print("[Stage 1] Classifier ready.")


def unload_classifier():
    global _model, _tokenizer
    del _model
    del _tokenizer
    _model     = None
    _tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"[Stage 1] Unloaded. VRAM free: {free/1e9:.2f}/{total/1e9:.2f} GB")


def understand_prompt(user_prompt: str) -> dict:
    load_classifier()

    with open(f"{PROMPT_DIR}/classifier.txt") as f:
        system = f.read()

    messages = [{
        "role": "user",
        "content": f"{system}\n\nAnalyze this topic and return JSON only: \"{user_prompt}\""
    }]

    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1500
    ).to(next(_model.parameters()).device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=700,
            temperature=0.2,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    reply = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"[Stage 1] Reply preview: {reply[:400]}")

    try:
        start  = reply.find("{")
        end    = reply.rfind("}") + 1
        result = json.loads(reply[start:end])

        # Normalize — support both old and new schema
        if "suggested_scenes" in result and "scenes" not in result:
            result["scenes"] = result["suggested_scenes"]

        # Ensure animation_type is set
        if "animation_type" not in result:
            result["animation_type"] = _infer_type(user_prompt, result.get("domain","general"))

        # Propagate animation_type into scenes
        for s in result.get("scenes", []):
            if "animation_type" not in s:
                s["animation_type"] = result["animation_type"]
            if "visual_plan" not in s:
                s["visual_plan"] = s.get("description", user_prompt)

        print(f"[Stage 1] Topic: {result.get('topic')}")
        print(f"[Stage 1] Type: {result.get('animation_type')}")
        print(f"[Stage 1] Visual plan: {result.get('scenes',[{}])[0].get('visual_plan','')[:120]}")

    except Exception as e:
        print(f"[Stage 1] JSON parse failed ({e}), using fallback")
        anim_type = _infer_type(user_prompt, "general")
        result = _fallback(user_prompt, anim_type)

    unload_classifier()
    return result


def _infer_type(prompt: str, domain: str) -> str:
    """Rule-based fallback animation type inference."""
    p = prompt.lower()
    if any(w in p for w in ["sort", "search", "algorithm", "step", "process", "reaction", "proof"]):
        return "stepwise"
    if any(w in p for w in ["circle", "triangle", "polygon", "theorem", "angle", "vector", "geometry"]):
        return "geometry"
    if any(w in p for w in ["motion", "projectile", "pendulum", "orbit", "wave propagation", "particle", "spring"]):
        return "particle"
    if any(w in p for w in ["fourier", "unit circle", "sin cos", "lissajous", "epicycle"]):
        return "hybrid"
    if any(w in p for w in ["graph", "function", "derivative", "integral", "series", "transform", "frequency"]):
        return "graph"
    if domain in ["physics"]:
        return "particle"
    if domain in ["cs"]:
        return "stepwise"
    if domain in ["geometry"]:
        return "geometry"
    return "graph"


def _fallback(user_prompt: str, anim_type: str) -> dict:
    return {
        "topic": user_prompt,
        "domain": "general",
        "animation_type": anim_type,
        "key_concepts": [user_prompt],
        "narration_script": (
            f"Let us explore {user_prompt}. "
            f"This concept is fundamental to understanding the underlying principles at work. "
            f"We will visualize the key ideas step by step, revealing the patterns and relationships "
            f"that make this topic both powerful and elegant in its application."
        ),
        "animation_style": "explanatory",
        "scenes": [{
            "scene_name": "ExplanationScene",
            "animation_type": anim_type,
            "visual_plan": f"Illustrate {user_prompt} visually with appropriate animations",
            "formulas": [],
            "has_axes": anim_type in ["graph", "hybrid"],
            "has_shapes": anim_type in ["geometry", "hybrid"],
            "has_formula": True,
            "duration_seconds": 55,
        }],
    }