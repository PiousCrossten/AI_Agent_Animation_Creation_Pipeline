import os
import re
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import PROMPT_DIR, TEMP_DIR, MAX_RETRIES, CODEGEN_MAX_TOKENS, CODEGEN_TEMPERATURE

CODEGEN_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"

_model     = None
_tokenizer = None

MAX_VIDEO_DURATION = getattr(__import__('config'), 'MAX_VIDEO_DURATION', 75)

MANIM_HEADER = """\
from manim import *
import numpy as np

config.pixel_width  = 1080
config.pixel_height = 1920
config.frame_rate   = 30
config.background_color = "#000000"

"""

# ── Valid Manim CE v0.18 colors ONLY ─────────────────────────────────────────
VALID_COLORS = [
    "BLUE_C", "BLUE_E", "BLUE_D", "BLUE_B", "BLUE_A", "DARK_BLUE",
    "TEAL_C", "TEAL_E", "TEAL_D", "TEAL_B", "TEAL_A",
    "GREEN_C", "GREEN_E", "GREEN_D", "GREEN_B", "GREEN_A", "DARK_GREEN",
    "YELLOW_C", "YELLOW_E", "YELLOW_D", "YELLOW_B", "YELLOW_A",
    "GOLD", "GOLD_A", "GOLD_B", "GOLD_C", "GOLD_D", "GOLD_E",
    "RED_C", "RED_E", "RED_D", "RED_B", "RED_A",
    "ORANGE", "WHITE", "GREY_B", "GREY_C", "GREY_D",
    "PURPLE", "PINK", "MAROON", "DARK_BROWN",
]

# Hallucinated → real color mapping (auto-fixed at post-processing)
COLOR_FIXES = {
    "PURPLE_C":  "PURPLE",   "PURPLE_E":  "PURPLE",
    "PINK_C":    "PINK",     "PINK_E":    "PINK",
    "MAROON_C":  "MAROON",   "MAROON_E":  "MAROON",
    "LIME_C":    "GREEN_C",  "LIME_E":    "GREEN_E",
    "NAVY_C":    "DARK_BLUE","NAVY_E":    "DARK_BLUE",
    "BROWN_C":   "DARK_BROWN","BROWN_E":  "DARK_BROWN",
    "GREY_C":    "GREY_B",   "GREY_E":    "GREY_B",
    "GRAY_C":    "GREY_B",   "GRAY_B":    "GREY_B",   "GRAY_E": "GREY_B",
    "CYAN_C":    "TEAL_C",   "CYAN_E":    "TEAL_E",
    "INDIGO_C":  "BLUE_D",   "VIOLET_C":  "PURPLE",
    "MAGENTA_C": "PINK",     "CRIMSON_C": "RED_C",
    "AMBER_C":   "GOLD",     "SILVER_C":  "GREY_B",
}

# ── Per-type concrete example templates ───────────────────────────────────────
EXAMPLES = {

"graph": """\
# GRAPH TYPE — adapt function and formula to your specific topic
class ExplanationScene(Scene):
    def construct(self):
        # Moving background via updater — never use blocking self.play(t.animate..., run_time=50)
        t = ValueTracker(0)
        bg_ax = Axes(x_range=[-4,4], y_range=[-2,2], x_length=10, y_length=5,
                     axis_config={"stroke_opacity":0.0})
        bg1 = always_redraw(lambda: bg_ax.plot(
            lambda x: np.sin(x + t.get_value()),
            color=BLUE_E, stroke_opacity=0.12, stroke_width=1.5))
        bg2 = always_redraw(lambda: bg_ax.plot(
            lambda x: 0.5*np.cos(2*x - t.get_value()*0.7),
            color=TEAL_C, stroke_opacity=0.08, stroke_width=1.5))
        self.add(bg_ax, bg1, bg2)
        def tick(dt): t.increment_value(dt * 0.4)
        self.add_updater(tick)

        # Zone A — formula at top (ONLY text/formulas here)
        formula = MathTex(r"f(x) = \sin(x)", font_size=46, color=WHITE)
        formula.to_edge(UP, buff=0.4)
        self.play(Write(formula), run_time=2)
        self.wait(1.5)

        # Zone B — axes and graph in middle
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 2, 1],
            x_length=5.5, y_length=4.5,
            axis_config={"color": GREY_B, "stroke_width": 2, "include_tip": True},
            x_axis_config={"numbers_to_include": np.arange(-2, 3, 1)},
            y_axis_config={"numbers_to_include": np.arange(-1, 2, 1)},
        )
        axes.move_to(DOWN * 0.3)
        ax_label_x = axes.get_x_axis_label(MathTex("x", font_size=28))
        ax_label_y = axes.get_y_axis_label(MathTex("y", font_size=28))
        self.play(Create(axes), Write(ax_label_x), Write(ax_label_y), run_time=2)
        self.wait(1.5)

        graph = axes.plot(lambda x: np.sin(x), color=BLUE_C, stroke_width=4)
        graph_label = axes.get_graph_label(
            graph, MathTex(r"\sin(x)", font_size=28, color=BLUE_C), x_val=2.0, direction=UP)
        self.play(Create(graph), Write(graph_label), run_time=2.5)
        self.wait(1.5)

        # Dot moving along curve
        dot = Dot(color=YELLOW_C, radius=0.12)
        dot.move_to(axes.c2p(-3, np.sin(-3)))
        self.play(FadeIn(dot), run_time=0.5)
        self.play(MoveAlongPath(dot, graph), run_time=3.5, rate_func=linear)
        self.play(FadeOut(dot), run_time=0.5)
        self.wait(1.5)

        # Parameter sweep with ValueTracker
        k = ValueTracker(1.0)
        sweep = always_redraw(lambda: axes.plot(
            lambda x: np.sin(k.get_value() * x), color=TEAL_C, stroke_width=4))
        k_disp = always_redraw(lambda: MathTex(
            f"k = {k.get_value():.1f}", font_size=30, color=GOLD
        ).to_edge(DOWN, buff=0.9))
        self.play(FadeOut(graph), FadeOut(graph_label), run_time=0.5)
        self.add(sweep, k_disp)
        self.play(k.animate.set_value(4.0), run_time=5, rate_func=linear)
        self.wait(1.5)
        self.play(k.animate.set_value(0.5), run_time=3, rate_func=linear)
        self.wait(1.5)

        # Zone C — insight at bottom (ONLY labels/text here)
        insight = Text("Key insight about this concept", font_size=28, color=YELLOW_C)
        insight.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(insight, shift=UP * 0.2), run_time=1)
        self.wait(2)

        self.remove_updater(tick)
        self.play(FadeOut(Group(*self.mobjects)), run_time=2)
""",

"geometry": """\
# GEOMETRY TYPE — adapt shapes and theorem to your specific topic
class ExplanationScene(Scene):
    def construct(self):
        # Moving background — rotating polygon
        bg_hex = RegularPolygon(n=6, color=BLUE_E, stroke_opacity=0.08, stroke_width=1)
        bg_hex.scale(4.5)
        bg_hex.add_updater(lambda m, dt: m.rotate(dt * 0.12))
        self.add(bg_hex)

        # Zone A — theorem at top
        theorem = MathTex(r"a^2 + b^2 = c^2", font_size=46, color=WHITE)
        theorem.to_edge(UP, buff=0.4)
        self.play(Write(theorem), run_time=2)
        self.wait(1.5)

        # Zone B — geometric construction in middle
        A = np.array([-1.5, -1.5, 0])
        B = np.array([ 1.5, -1.5, 0])
        C = np.array([-1.5,  1.5, 0])
        tri = Polygon(A, B, C, color=WHITE, stroke_width=3)
        tri.move_to(ORIGIN)
        self.play(Create(tri), run_time=2)
        self.wait(1.5)

        # Right angle marker
        ra = RightAngle(Line(A, B), Line(A, C), length=0.25, color=GREY_B)
        self.play(Create(ra), run_time=0.8)

        # Side labels
        a_lbl = MathTex("a", font_size=34, color=BLUE_C)
        a_lbl.next_to(Line(A, C).get_center(), LEFT, buff=0.2)
        b_lbl = MathTex("b", font_size=34, color=TEAL_C)
        b_lbl.next_to(Line(A, B).get_center(), DOWN, buff=0.2)
        c_lbl = MathTex("c", font_size=34, color=YELLOW_C)
        c_lbl.next_to(Line(B, C).get_center(), RIGHT, buff=0.2)
        self.play(Write(a_lbl), Write(b_lbl), Write(c_lbl), run_time=1.5)
        self.wait(1.5)

        # Squares on sides
        sq_a = Square(side_length=1.5, color=BLUE_C, fill_opacity=0.25, stroke_width=2)
        sq_a.next_to(tri, LEFT, buff=0)
        sq_b = Square(side_length=1.5, color=TEAL_C, fill_opacity=0.25, stroke_width=2)
        sq_b.next_to(tri, DOWN, buff=0)
        sq_c = Square(side_length=2.12, color=YELLOW_C, fill_opacity=0.25, stroke_width=2)
        sq_c.rotate(np.arctan2(3, 3))
        sq_c.move_to(Line(B, C).get_center() + RIGHT * 0.9 + UP * 0.1)
        self.play(Create(sq_a), run_time=1.5)
        self.play(Create(sq_b), run_time=1.5)
        self.play(Create(sq_c), run_time=2)
        self.wait(1.5)

        # Pulse theorem
        self.play(Indicate(theorem, scale_factor=1.2, color=YELLOW_C), run_time=1.5)
        self.wait(1.5)

        # Zone C — insight at bottom
        insight = Text("Sum of squares on legs\\nequals square on hypotenuse",
                       font_size=26, color=YELLOW_C, line_spacing=1.4)
        insight.to_edge(DOWN, buff=0.4)
        self.play(Write(insight), run_time=2)
        self.wait(2)

        self.play(FadeOut(Group(*self.mobjects)), run_time=2)
""",

"particle": """\
# PARTICLE TYPE — adapt trajectory and physics to your specific topic
class ExplanationScene(Scene):
    def construct(self):
        # Background — subtle grid
        grid = NumberPlane(
            x_range=[-7, 7, 1], y_range=[-4, 4, 1],
            background_line_style={"stroke_opacity": 0.06, "stroke_width": 1},
            axis_config={"stroke_opacity": 0.0},
        )
        self.add(grid)

        # Zone A — formula at top
        formula = MathTex(r"y = v_0 t - \frac{1}{2}g t^2", font_size=46, color=WHITE)
        formula.to_edge(UP, buff=0.4)
        self.play(Write(formula), run_time=2)
        self.wait(1.5)

        # Zone B — particle motion in middle
        v0x, v0y, g = 2.0, 3.0, 9.8
        t_flight = 2 * v0y / g
        path = ParametricFunction(
            lambda s: np.array([-2.5 + v0x*s, 0.5 + v0y*s - 0.5*g*s**2, 0]),
            t_range=[0, t_flight], color=BLUE_C, stroke_width=3
        )
        self.play(Create(path), run_time=2)
        self.wait(1)

        ball = Dot(color=TEAL_C, radius=0.16)
        ball.move_to(path.get_start())
        trace = TracedPath(ball.get_center,
                           stroke_color=TEAL_C, stroke_width=2, stroke_opacity=0.6)
        self.add(trace, ball)
        self.play(MoveAlongPath(ball, path), run_time=3.5, rate_func=linear)
        self.wait(1.5)

        # Gravity arrow
        land = ball.get_center()
        g_arrow = Arrow(land, land + DOWN*1.2, color=RED_C, buff=0, stroke_width=4)
        g_label = MathTex(r"g = 9.8\ \text{m/s}^2", font_size=28, color=RED_C)
        g_label.next_to(g_arrow, RIGHT, buff=0.15)
        self.play(GrowArrow(g_arrow), Write(g_label), run_time=1.5)
        self.wait(1.5)

        # Velocity components
        v_arrow_x = Arrow(path.get_start(), path.get_start()+RIGHT*1.2,
                          color=GREEN_C, buff=0, stroke_width=3)
        v_arrow_y = Arrow(path.get_start(), path.get_start()+UP*1.2,
                          color=YELLOW_C, buff=0, stroke_width=3)
        vx_lbl = MathTex("v_x", font_size=26, color=GREEN_C).next_to(v_arrow_x, DOWN, buff=0.1)
        vy_lbl = MathTex("v_y", font_size=26, color=YELLOW_C).next_to(v_arrow_y, RIGHT, buff=0.1)
        self.play(GrowArrow(v_arrow_x), GrowArrow(v_arrow_y),
                  Write(vx_lbl), Write(vy_lbl), run_time=2)
        self.wait(1.5)

        # Zone C — insight at bottom
        insight = Text("Horizontal velocity is constant\\nVertical velocity changes due to gravity",
                       font_size=25, color=YELLOW_C, line_spacing=1.4)
        insight.to_edge(DOWN, buff=0.4)
        self.play(Write(insight), run_time=2)
        self.wait(2)

        self.play(FadeOut(Group(*self.mobjects)), run_time=2)
""",

"stepwise": """\
# STEPWISE TYPE — adapt steps and data to your specific algorithm/process
class ExplanationScene(Scene):
    def construct(self):
        # Background — subtle grid
        grid = NumberPlane(
            x_range=[-7, 7, 1], y_range=[-4, 4, 1],
            background_line_style={"stroke_opacity": 0.05, "stroke_width": 1},
            axis_config={"stroke_opacity": 0.0},
        )
        self.add(grid)

        # Zone A — title at top
        title = Text("Bubble Sort", font_size=40, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(1.5)

        # Zone B — bar array in middle (MAX 6 bars, use only valid 6 colors)
        values    = [5, 3, 8, 1, 6, 2]
        bar_cols  = [BLUE_C, TEAL_C, GREEN_C, YELLOW_C, ORANGE, RED_C]
        bar_w     = 0.65
        spacing   = 0.85
        base_y    = -1.2       # bars grow upward from this y

        rects, lbls = [], []
        for i, val in enumerate(values):
            h = val * 0.35
            r = Rectangle(width=bar_w, height=h,
                          fill_color=bar_cols[i], fill_opacity=0.85,
                          stroke_color=WHITE, stroke_width=1.5)
            r.move_to(np.array([-2.5 + i*spacing, base_y + h/2, 0]))
            lbl = Text(str(val), font_size=24, color=WHITE)
            lbl.move_to(r.get_top() + UP*0.22)
            rects.append(r)
            lbls.append(lbl)

        self.play(*[Create(r) for r in rects], *[Write(l) for l in lbls], run_time=2)
        self.wait(1.5)

        # Swap helper
        def do_swap(i, j, step_num, note):
            lbl = Text(f"Pass {step_num}: {note}", font_size=26, color=YELLOW_C)
            lbl.to_edge(DOWN, buff=0.85)
            self.play(Write(lbl), run_time=0.8)
            self.play(Indicate(rects[i], color=RED_C, scale_factor=1.15),
                      Indicate(rects[j], color=RED_C, scale_factor=1.15), run_time=1)
            shift = (j - i) * spacing
            self.play(rects[i].animate.shift(RIGHT*shift), rects[j].animate.shift(LEFT*shift),
                      lbls[i].animate.shift(RIGHT*shift),  lbls[j].animate.shift(LEFT*shift),
                      run_time=1.2)
            rects[i], rects[j] = rects[j], rects[i]
            lbls[i],  lbls[j]  = lbls[j],  lbls[i]
            self.wait(1.2)
            self.play(FadeOut(lbl), run_time=0.4)

        def no_swap(i, j, step_num, note):
            lbl = Text(f"Pass {step_num}: {note}", font_size=26, color=GREEN_C)
            lbl.to_edge(DOWN, buff=0.85)
            self.play(Write(lbl), run_time=0.8)
            self.play(Indicate(rects[i], color=GREEN_C, scale_factor=1.1),
                      Indicate(rects[j], color=GREEN_C, scale_factor=1.1), run_time=1)
            self.wait(1)
            self.play(FadeOut(lbl), run_time=0.4)

        # Pass 1
        do_swap(0, 1, 1, "5 > 3 → Swap")
        no_swap(1, 2, 1, "5 < 8 → Keep")
        do_swap(2, 3, 1, "8 > 1 → Swap")

        # Sorted marker on last
        done_box = SurroundingRectangle(rects[-1], color=GREEN_C, stroke_width=2)
        done_lbl = Text("sorted", font_size=20, color=GREEN_C)
        done_lbl.next_to(done_box, DOWN, buff=0.1)
        self.play(Create(done_box), Write(done_lbl), run_time=1)
        self.wait(1.5)

        # Complexity in Zone A
        complexity = MathTex(r"\mathcal{O}(n^2)", font_size=40, color=TEAL_C)
        complexity.next_to(title, RIGHT, buff=0.8)
        self.play(Write(complexity), run_time=1.5)
        self.wait(1.5)

        # Zone C — insight at bottom
        insight = Text("Each pass bubbles the largest\\nunsorted element to its final place",
                       font_size=26, color=YELLOW_C, line_spacing=1.4)
        insight.to_edge(DOWN, buff=0.4)
        self.play(Write(insight), run_time=2)
        self.wait(2)

        self.play(FadeOut(Group(*self.mobjects)), run_time=2)
""",

"hybrid": """\
# HYBRID TYPE — geometry + graph split in Zone B vertically
class ExplanationScene(Scene):
    def construct(self):
        # Moving background
        t = ValueTracker(0)
        bg_ax = Axes(x_range=[-4, 4], y_range=[-2, 2], x_length=10, y_length=5,
                     axis_config={"stroke_opacity": 0.0})
        bg1 = always_redraw(lambda: bg_ax.plot(
            lambda x: 0.4*np.sin(2*x + t.get_value()),
            color=BLUE_E, stroke_opacity=0.10, stroke_width=1.5))
        self.add(bg_ax, bg1)
        def tick(dt): t.increment_value(dt * 0.3)
        self.add_updater(tick)

        # Zone A — formula at top
        formula = MathTex(
            r"\sin(\theta) = \frac{\text{opposite}}{\text{hypotenuse}}",
            font_size=38, color=WHITE
        )
        formula.to_edge(UP, buff=0.4)
        self.play(Write(formula), run_time=2)
        self.wait(1.5)

        # Zone B TOP — unit circle above center
        theta = ValueTracker(0)
        circle = Circle(radius=1.4, color=GREY_B, stroke_width=2)
        circle.move_to(UP * 1.3)
        self.play(Create(circle), run_time=1.5)

        dot_on_circle = always_redraw(lambda: Dot(
            circle.get_center() + 1.4*np.array([
                np.cos(theta.get_value()), np.sin(theta.get_value()), 0]),
            color=YELLOW_C, radius=0.12))
        radius_line = always_redraw(lambda: Line(
            circle.get_center(),
            circle.get_center() + 1.4*np.array([
                np.cos(theta.get_value()), np.sin(theta.get_value()), 0]),
            color=TEAL_C, stroke_width=3))
        sine_proj = always_redraw(lambda: Line(
            circle.get_center() + 1.4*np.array([
                np.cos(theta.get_value()), np.sin(theta.get_value()), 0]),
            circle.get_center() + np.array([
                1.4*np.cos(theta.get_value()), 0, 0]),
            color=RED_C, stroke_width=2.5))
        self.add(dot_on_circle, radius_line, sine_proj)
        self.wait(1.5)

        # Zone B BOTTOM — sine axes below center
        axes = Axes(
            x_range=[0, TAU, PI/2], y_range=[-1.5, 1.5, 0.5],
            x_length=5.0, y_length=2.2,
            axis_config={"color": GREY_B, "stroke_width": 1.5, "include_tip": False},
        )
        axes.move_to(DOWN * 2.3)
        ax_lbl = axes.get_x_axis_label(MathTex(r"\theta", font_size=26))
        self.play(Create(axes), Write(ax_lbl), run_time=1.5)

        sine_graph = always_redraw(lambda: axes.plot(
            lambda x: np.sin(x) if x <= theta.get_value() else np.nan,
            color=BLUE_C, stroke_width=3, use_vectorized=False))
        self.add(sine_graph)

        # Full rotation
        self.play(theta.animate.set_value(TAU*2), run_time=9, rate_func=linear)
        self.wait(1.5)

        # Zone C — insight at bottom
        insight = Text("The sine traces the vertical\\ncomponent of circular motion",
                       font_size=26, color=YELLOW_C, line_spacing=1.4)
        insight.to_edge(DOWN, buff=0.35)
        self.play(Write(insight), run_time=2)
        self.wait(2)

        self.remove_updater(tick)
        self.play(FadeOut(Group(*self.mobjects)), run_time=2)
"""
}


def load_codegen():
    global _model, _tokenizer
    if _model is not None:
        return
    print(f"[Stage 2] Loading: {CODEGEN_MODEL}")
    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(CODEGEN_MODEL, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        CODEGEN_MODEL,
        quantization_config=q,
        device_map="auto",
    )
    _model.eval()
    print("[Stage 2] Codegen ready.")


def unload_codegen():
    global _model, _tokenizer
    del _model, _tokenizer
    _model     = None
    _tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("[Stage 2] Codegen unloaded.")


def _generate(messages: list) -> str:
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, max_length=4096
    ).to(next(_model.parameters()).device)

    input_len = inputs["input_ids"].shape[1]
    print(f"[Stage 2] Input tokens: {input_len} | Output budget: {CODEGEN_MAX_TOKENS}")

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=CODEGEN_MAX_TOKENS,
            temperature=CODEGEN_TEMPERATURE,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )

    new_tokens = outputs[0][input_len:]
    result = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"[Stage 2] Output: {len(new_tokens)} tokens, {len(result)} chars")
    return result


def _extract_code(raw: str) -> str:
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if m:
        return _ensure_header(m.group(1).strip())
    m = re.search(r"```\s*(from manim.*?)```", raw, re.DOTALL)
    if m:
        return _ensure_header(m.group(1).strip())
    if "from manim import" in raw:
        return _ensure_header(raw[raw.find("from manim import"):].strip())
    return MANIM_HEADER + raw.strip()


def _ensure_header(code: str) -> str:
    if "config.pixel_width" not in code:
        code = MANIM_HEADER + code
    return _apply_fixes(code)


def _apply_fixes(code: str) -> str:
    """Auto-correct all known model hallucination patterns."""

    # 1. Fix invalid color names
    for wrong, right in COLOR_FIXES.items():
        code = re.sub(rf'\b{re.escape(wrong)}\b', right, code)

    # 2. VGroup(*self.mobjects) → Group(*self.mobjects)
    code = re.sub(r"VGroup\(\*self\.mobjects\)", "Group(*self.mobjects)", code)

    # 3. self.clear() is not a Scene method
    code = re.sub(
        r"\bself\.clear\(\)",
        "self.play(FadeOut(Group(*self.mobjects)), run_time=1)",
        code
    )

    # 4. Remove blocking background animation
    code = re.sub(
        r"self\.play\(\s*t\.animate\.set_value\([^)]+\)\s*,\s*run_time\s*=\s*\d+[^)]*\)",
        "# background runs via add_updater(tick)",
        code
    )

    # 5. ShowCreation deprecated → Create
    code = re.sub(r"\bShowCreation\b", "Create", code)

    # 6. DrawBorderThenFill on Text crashes
    code = re.sub(r"DrawBorderThenFill\((title|text|label|header)\b", r"Write(\1", code)

    # 7. Ensure numpy imported if used
    if "np." in code and "import numpy" not in code:
        code = code.replace("from manim import *", "from manim import *\nimport numpy as np")

    # 8. Trailing newline
    if not code.endswith("\n"):
        code += "\n"

    return code


def _check_syntax(path: str) -> tuple:
    try:
        with open(path) as f:
            source = f.read()
        compile(source, path, "exec")
        return True, ""
    except SyntaxError as e:
        lines = source.splitlines()
        bad   = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
        return False, f"SyntaxError line {e.lineno}: {e.msg} | Code: '{bad.strip()}'"


def _count_scenes(code: str) -> list:
    return re.findall(r"^class\s+(\w+)\s*\(\s*\w*Scene\w*\s*\):", code, re.MULTILINE)


def _estimate_duration(code: str) -> float:
    total = 0.0
    for m in re.finditer(r"self\.wait\(([\d.]+)\)", code):
        total += float(m.group(1))
    for m in re.finditer(r"run_time=([\d.]+)", code):
        total += float(m.group(1))
    return total


def _build_message(concept: dict, error: str = "", prev_code: str = "") -> list:
    with open(f"{PROMPT_DIR}/manim_codegen.txt") as f:
        system_prompt = f.read()

    topic        = concept.get("topic", "Unknown")
    domain       = concept.get("domain", "general")
    anim_type    = concept.get("animation_type", "graph")
    key_concepts = ", ".join(concept.get("key_concepts", [topic]))
    visual_plan  = ""
    formulas     = []

    scenes = concept.get("scenes", concept.get("suggested_scenes", []))
    if scenes:
        s           = scenes[0]
        visual_plan = s.get("visual_plan", s.get("description", ""))
        formulas    = s.get("formulas", [])
        anim_type   = s.get("animation_type", anim_type)

    example          = EXAMPLES.get(anim_type, EXAMPLES["graph"])
    valid_color_list = ", ".join(VALID_COLORS)
    formula_str      = f"Formulas for Zone A: {', '.join(formulas)}\n" if formulas else ""

    # Target duration that matches MAX_VIDEO_DURATION
    target_dur = MAX_VIDEO_DURATION

    user_msg = (
        f"{system_prompt}\n\n"
        f"TOPIC: {topic}\n"
        f"DOMAIN: {domain}\n"
        f"ANIMATION TYPE: {anim_type}\n"
        f"KEY CONCEPTS: {key_concepts}\n"
        f"VISUAL PLAN: {visual_plan}\n"
        f"{formula_str}"
        f"VALID COLORS (use ONLY these): {valid_color_list}\n\n"
        f"TARGET DURATION: {target_dur} seconds\n"
        f"  - Use self.wait(1.5) between animations — not more than 2.0\n"
        f"  - Total wait() + run_time should sum to {target_dur-10}–{target_dur} seconds\n\n"
        f"LAYOUT ZONES — HARD RULES, no exceptions:\n"
        f"  Zone A (y≥+3.0): .to_edge(UP, buff=0.4)   → ONLY MathTex formulas and Text titles\n"
        f"  Zone B (y -2.2 to +2.2): .move_to(ORIGIN) or .move_to(DOWN*0.3) → ONLY axes/shapes/particles\n"
        f"  Zone C (y≤-3.0): .to_edge(DOWN, buff=0.4) → ONLY insight text and labels\n"
        f"  Axes MUST be at .move_to(DOWN*0.3) — NEVER at .to_edge(UP) or near formula\n"
        f"  Formula MUST be at .to_edge(UP, buff=0.4) — NEVER near axes\n\n"
        f"TEMPLATE TO FOLLOW (adapt ALL content to {topic}):\n"
        f"{example}\n"
        f"{'─'*50}\n"
        f"Now write ExplanationScene for '{topic}':\n"
        f"  1. Animation type '{anim_type}' — follow the template structure exactly\n"
        f"  2. ALL visuals must be specific to {topic} — not generic sine/cosine\n"
        f"  3. Zone A: formula.to_edge(UP, buff=0.4) — formula ABOVE axes, never overlapping\n"
        f"  4. Zone B: axes.move_to(DOWN*0.3) — axes BELOW formula, never overlapping\n"
        f"  5. Zone C: ONLY .to_edge(DOWN) objects\n"
        f"  6. Use ONLY the valid colors listed above\n"
        f"  7. Every self.play() must have run_time= argument\n"
        f"  8. Return ONLY the Python script — no markdown, no explanation\n"
    )

    if error:
        return [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": prev_code[:3000]},
            {"role": "user",      "content": (
                f"The script has this error:\n{error}\n\n"
                f"Fix it and return the complete corrected Python script only."
            )},
        ]
    return [{"role": "user", "content": user_msg}]


_last_code = ""


def generate_manim_code(concept_info: dict) -> str:
    global _last_code
    load_codegen()

    script_path = os.path.join(TEMP_DIR, "generated_scene.py")
    last_error  = ""

    for attempt in range(MAX_RETRIES):
        print(f"[Stage 2] Attempt {attempt + 1}/{MAX_RETRIES}")

        messages   = _build_message(concept_info, last_error if attempt > 0 else "", _last_code)
        raw        = _generate(messages)
        code       = _extract_code(raw)
        _last_code = code

        print(f"[Stage 2] Code: {len(code)} chars")
        first = re.search(r"^class \w+", code, re.MULTILINE)
        print(f"[Stage 2] First class: {'FOUND at char ' + str(first.start()) if first else 'NOT FOUND'}")

        with open(script_path, "w") as f:
            f.write(code)

        ok, err = _check_syntax(script_path)
        if ok:
            scenes  = _count_scenes(code)
            est_dur = _estimate_duration(code)
            print(f"[Stage 2] Scenes: {scenes} | Est. duration: {est_dur:.0f}s")

            if len(scenes) >= 1 and est_dur >= 25:
                print(f"[Stage 2] Accepted → {script_path}")
                unload_codegen()
                return script_path
            elif len(scenes) < 1:
                last_error = (
                    "No ExplanationScene class found. "
                    "Write class ExplanationScene(Scene): with full construct(self): method."
                )
            else:
                last_error = (
                    f"Duration only {est_dur:.0f}s. Need 40-65 seconds. "
                    f"Add a ValueTracker parameter sweep. "
                    f"Add a Dot moving along a path with MoveAlongPath. "
                    f"Use self.wait(1.5) after each self.play()."
                )
            print(f"[Stage 2] Retry: {last_error}")
        else:
            print(f"[Stage 2] Syntax error: {err}")
            last_error = err

    print("[Stage 2] Max retries reached — using best attempt.")
    unload_codegen()
    return script_path