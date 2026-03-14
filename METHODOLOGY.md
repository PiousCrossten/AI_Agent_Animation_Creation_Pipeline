# Methodology: AI Animation Pipeline

A simple guide to how the **AI Animation Pipeline** works and why it’s built this way. Think of it as an **AI agent** that turns your idea into a short educational video.

---

## What Is This?

**AI Animation Pipeline** is an AI agent that takes one sentence from you (e.g. *“Explain bubble sort”*) and produces a full short video: script, animation, and voiceover.

It does **not** use a single “magic” model. Instead, it uses a **fixed sequence of steps** (a pipeline), where each step has a clear job. That makes the system easier to understand, run, and fix.

---

## The Big Picture

You give a **topic** → The agent **plans** what to show and say → **Generates** animation code → **Renders** the video → **Adds** voice and exports the final file.

So the “agent” is really: **understand → plan → generate code → render → add audio → export**. One pass, no loops, no extra tools.

---

## Step 1: Plan First, Then Build

The agent splits “thinking” from “doing” into two AI steps:

1. **Understand & plan**  
   One AI model reads your topic and outputs a **structured plan** (like a short script): what’s the idea, what type of animation (graph, geometry, step-by-step, etc.), what to say in the voiceover, and what should appear on screen.

2. **Generate code**  
   Another AI model (good at code) takes that plan and writes the **animation program** (Manim script) that matches the plan.

**Why do it in two steps?**  
Planning (“what to explain”) and coding (“how to draw it”) need different skills. Doing both in one step often gives messy or generic results. With two steps, the plan is clear and the code model only has to “fill in” the animation from the plan, which it does better.

---

## The Agent Is a Pipeline, Not a Chatbot

The agent does **not** chat with you or decide step-by-step what to do next. It always runs the **same pipeline**:

1. Understand the topic → make a plan  
2. Generate animation code from the plan  
3. Render the animation to video  
4. Generate voiceover and mix it with the video  
5. Export the final video file  

So the “agent” is this fixed workflow. Same input (your topic) → same sequence → you get a video. That makes behavior **predictable** and **easy to debug**: if something breaks, you know which step failed.

---

## Why Two AI Models, and Why One at a Time?

We use **two** different models:

- **First model** (e.g. 7B parameters): understands your topic and writes the plan (text + structure).
- **Second model** (e.g. 14B parameters): reads the plan and writes the animation code.

They run **one after the other**, not together. After the first model finishes, we **free its memory** (VRAM) and then load the second. So at any moment only one big model is in memory.

**Why?**  
So you can run the whole pipeline on **one normal GPU** (e.g. 12 GB). Two big models at once would need more VRAM than many people have. Doing “plan then code” and “one model at a time” keeps it feasible.

---

## Why We Use Ready-Made Animation “Templates”

The code-writing model doesn’t invent the animation from zero. We give it **templates**: full, working examples for each **animation type** (e.g. “graph,” “geometry,” “step-by-step,” “particle motion”). The model’s job is to **adapt** the right template to your topic (change labels, formulas, numbers) while keeping the structure.

**Why?**  
Animation code has many small rules (correct colors, correct API names, layout). If the model invents everything, it often breaks those rules. Templates keep layout and style consistent and reduce broken or ugly output.

---

## Cleaning Up the Generated Code

The code that the second model produces is **not** run as-is. We run a **clean-up step** that:

- Fixes wrong or made-up color names  
- Replaces old or invalid API calls with the right ones  
- Makes sure resolution, frame rate, and background are set correctly  

**Why?**  
Models sometimes make small mistakes (wrong names, old syntax). A few automatic fixes are simpler and more reliable than asking the model to be perfect in one go.

---

## After the AI: Everything Else Is Deterministic

Only **two** steps use AI: (1) plan from your topic, (2) code from the plan. The rest is **fixed programs**:

- **Rendering** the animation (Manim)  
- **Generating** voice (TTS) and optional subtitles (Whisper)  
- **Mixing** audio and video and **exporting** the file (FFmpeg)  

Same plan and code → same video. No randomness, no extra AI calls. That keeps the result **predictable** and the **cost and time** under control.

---

## Summary in Plain Words

| What we do | Why we do it |
|------------|----------------|
| **Plan first, then generate code** | Planning and coding need different skills; two steps give a clear plan and better code. |
| **Fixed pipeline (understand → plan → code → render → audio → export)** | Same behavior every time; easy to understand and debug. |
| **Two AI models, used one at a time** | Fits on one GPU; we free memory after the first model. |
| **Templates for each animation type** | Keeps layout and API usage correct; fewer broken videos. |
| **Auto-fix colors and API in generated code** | Handles small model mistakes without extra AI calls. |
| **Only 2 steps use AI; rest is standard tools** | Predictable output; lower cost and latency. |

The **AI Animation Pipeline** is an AI agent that uses a **simple, fixed workflow** to turn your topic into a short educational video. We favor **clarity, reproducibility, and running on one GPU** over maximum flexibility or a fully free-form “agent” that chooses its own steps.
