# Real Video Generation + Lip Sync — Setup Guide

By default the pipeline produces a multi-shot ffmpeg composition with
animated stills (very watchable, but no real motion or lip sync). To unlock
**real** text-to-video and **real** lip-synced talking heads, configure one
of three providers below. Setup is 2 minutes and the default tier is free.

> **TL;DR** — Get a free [fal.ai](https://fal.ai) key, add `FAL_KEY=...` to
> your `.env`, re-run `python main.py "your prompt"`. That's it.

---

## Why fal.ai (recommended)

| Why | Detail |
|-----|--------|
| **Free trial credits** | Sign-up gives you ~$1 of credits — enough for several full pipelines |
| **One key, many models** | Same `FAL_KEY` powers text-to-video AND lip sync |
| **Fast** | Stable Video Diffusion runs in ~10–20 s per scene |
| **No SDK install** | Plain HTTPS — no extra `pip install` needed |
| **Cancellable** | Bills per-second; pause anytime |

Approximate cost per full pipeline (4-scene short film, 4 character lines per scene):

| Tier | Models invoked | Cost |
|------|----------------|------|
| Default (no key) | ffmpeg only | **$0** |
| `FAL_KEY` set | 4× SVD video + 8× SadTalker lip-sync | **~$0.15–0.30** |

So a $1 trial gets you 3–6 full premium runs.

---

## Step-by-step setup

### 1. Create a fal.ai account

1. Visit **<https://fal.ai>**
2. Sign in with Google / GitHub (free)
3. You'll land on the dashboard with a free credit balance shown top-right

### 2. Generate an API key

1. Open **<https://fal.ai/dashboard/keys>**
2. Click **"Add new key"**
3. Name it (e.g. `agentic-video-project`)
4. Copy the key. It looks like:
   ```
   abcdef12-3456-7890-abcd-ef1234567890:1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d
   ```
   (a UUID, a colon, then a 64-char secret)

> ⚠️ The key is shown **once**. If you lose it, regenerate.

### 3. Drop the key into `.env`

Edit (or create) `.env` in the project root:

```bash
# from the repo root
cp .env.example .env
```

Then open `.env` in any editor and add **one** line:

```env
FAL_KEY=abcdef12-3456-7890-abcd-ef1234567890:1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d
```

That's it. Save the file.

### 4. Verify the key is detected

```bash
python main.py providers
```

You should see:

```
LLM           : mock (template fallback — set GEMINI_API_KEY for real LLM)
Image gen     : pollinations.ai (free)
Text-to-video : fal.ai ✓
Lip sync      : fal.ai SadTalker ✓
TTS           : gTTS (free, online)
Video tier    : PREMIUM (real motion + real lip sync)
```

If you see `Text-to-video : (none configured)`, the key wasn't loaded —
double-check the `.env` file is in the project root and has no quotes.

### 5. Run a full pipeline

```bash
python main.py "A young astronaut discovers a hidden ocean on Mars" --duration 30
```

You should see new log lines like:

```
INFO | video_agent  | scene scene_1: t2v clip generated (fal)
INFO | video_agent  | scene scene_1 composed (3 shots)
INFO | lip_sync     | fal.ai sadtalker -> scene_2_l1.mp4
```

The final MP4 will have **real motion** in the wide shots (rippling water,
moving hair, camera dolly) and **real mouth movement** synced to dialogue
on character close-ups.

---

## Models the project uses

### Text-to-video (in `mcp/tools/vision_tools/text_to_video_tool.py`)

| Endpoint | When used | Why |
|----------|-----------|-----|
| `fal-ai/stable-video-diffusion` | image→video (we have an establishing image already) | Most reliable; ~4 s clips at 1024×576 |
| `fal-ai/fast-svd/text-to-video` | pure text→video fallback | Faster but requires good prompts |

The agent uses **image-to-video** by default because it gives the model a
strong visual anchor and the result follows the still composition closely.

### Lip sync (in `mcp/tools/vision_tools/lip_sync_tool.py`)

| Endpoint | When used | Why |
|----------|-----------|-----|
| `fal-ai/sadtalker` | character close-ups during dialogue | Realistic facial motion + lip sync |
| `fal-ai/sync-lipsync` | alternative if SadTalker quota hits | Pure mouth-region animation |

We pass the **character portrait** + the **rendered TTS audio** for that
specific dialogue line, and get back a talking-head MP4 of exactly the
right duration.

---

## Alternative providers

### Replicate (paid; very accurate)

If you'd rather use Replicate (Wav2Lip is best-in-class for lip sync):

1. Sign up at **<https://replicate.com>**
2. Add a payment method (it has a $0 free tier but requires CC for video models)
3. Generate a token at **<https://replicate.com/account/api-tokens>**
4. Add to `.env`:
   ```env
   REPLICATE_API_TOKEN=r8_abc...
   ```

Cost: ~$0.04 per SVD clip, ~$0.02 per Wav2Lip render. Models used:
- `stability-ai/stable-video-diffusion`
- `lucataco/sadtalker` (lip sync)
- `devxpy/cog-wav2lip` (alternative lip sync)

### Hugging Face Inference API (free, lower quality)

The free tier supports text-to-video but quality is significantly worse and
**no lip-sync option**. Useful only for experimentation.

1. Sign up at **<https://huggingface.co>**
2. Generate a token at **<https://huggingface.co/settings/tokens>** (Read scope is enough)
3. Add to `.env`:
   ```env
   HF_TOKEN=hf_abc...
   ```

Model used: `damo-vilab/text-to-video-ms-1.7b` (text-to-video only).

### Provider precedence

The agent picks providers in this order:
1. `FAL_KEY` / `FAL_API_KEY` (preferred)
2. `REPLICATE_API_TOKEN`
3. `HF_TOKEN` / `HUGGINGFACE_API_KEY`
4. Local fallback (ffmpeg only)

You can mix: e.g. `FAL_KEY` for video + a different provider for lip sync —
the tools fail gracefully and try the next provider.

---

## Disabling premium video on a per-run basis

You may want a fast preview without burning credits. Two options:

**CLI**:
```bash
python main.py "your prompt" --no-real-video --no-lipsync
```

**Programmatic** (`backend/services/pipeline_service.py` or your own script):
```python
orchestrator.run_full(
    prompt="...",
    use_text_to_video=False,
    use_lip_sync=False,
)
```

Both flags default to **auto-detect** (use whichever providers are
configured). Setting `False` forces the local ffmpeg path.

---

## Performance & timing

A typical 30-second project on a normal laptop:

| Phase | Tier 1 (default) | Tier 2 (fal.ai) |
|-------|------------------|-----------------|
| Story (mock) | < 1 s | < 1 s |
| Audio (gTTS) | 10–20 s | 10–20 s |
| Video — image gen (4 scenes + 3 portraits) | 60–120 s | 60–120 s |
| Video — text-to-video (4 scenes) | 0 s (skipped) | 60–120 s |
| Video — lip sync (8 character lines) | 5 s (heuristic) | 240–480 s |
| Video — composition | 5–15 s | 5–15 s |
| **Total** | **~3 min** | **~10–15 min** |

So premium runs are ~5× slower but produce a dramatically better demo video.
For development iterate at Tier 1 then do a final premium render.

---

## Troubleshooting

### "no text-to-video provider configured"
Your `.env` isn't loading. Check:
- File is named exactly `.env` (not `.env.txt`) and lives in the project root
- The line has no surrounding quotes: `FAL_KEY=abc...` (✅), not `FAL_KEY="abc..."` (❌ may work depending on shell)
- `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('FAL_KEY' in os.environ)"` should print `True`

### `401 Unauthorized` from fal.ai
Key is malformed. Re-copy from <https://fal.ai/dashboard/keys> — make sure
you copied the full string with the colon in the middle.

### `402 Payment Required` from fal.ai
Trial credits exhausted. Top up at <https://fal.ai/dashboard/billing>.

### Lip sync runs but mouth doesn't move convincingly
Some character portraits don't have a clearly visible face (e.g. the
template "Narrator" is a silhouette). Either:
- Use a real LLM (`GEMINI_API_KEY`) so character `visual_description`s are
  more specific, OR
- Use the edit agent to regenerate the portrait: `"change character design"`

### `429 Rate limited`
Wait a minute and re-run. fal.ai has a per-minute concurrent-request limit
on free trials. The agent is sequential by default so this is rare.

### Hugging Face responds with HTML instead of a video
The model is cold-loading. The agent will surface this as a failed call and
fall back to local ffmpeg. Wait 30–60 s and try again.

---

## Cost-saving tips

1. **Iterate on Tier 1** while you're tweaking the script / prompts. Only
   enable `FAL_KEY` for the final render you'll demo.
2. **Lower scene count**. `--scenes 3 --duration 20` halves the cost.
3. **Skip lip sync for the narrator** — it's already done automatically (the
   narrator stays on the establishing shot, not a portrait).
4. **Re-use generated assets**. The state manager keeps every snapshot
   under `data/state_versions/`; revert with the edit agent's `revert N`
   command instead of re-running the pipeline.
5. **Edit, don't regenerate**. The edit agent's "apply vintage filter",
   "make scene 2 darker", etc. all do local-only operations — they cost $0.

---

## How it integrates with the edit agent

When you say things like:
- "change voice tone to whispered" — Phase 5 re-runs only the affected
  TTS lines (not video). $0 / fast.
- "regenerate scene 2" — Phase 5 regenerates the establishing image AND
  re-runs SVD on it (if `FAL_KEY` is set). One scene's worth of cost.
- "change character design" — Phase 5 regenerates ALL scenes featuring that
  character + re-renders the portrait. Bigger cost.

So the edit-and-undo loop stays cheap: most edits are local re-renders and
only "regenerate" intents touch the paid APIs.

---

## Putting it all together — example session

```bash
# 1. One-time setup
cp .env.example .env
echo "FAL_KEY=abcdef12-...:secret" >> .env

# 2. Verify
python main.py providers
# -> Video tier : PREMIUM (real motion + real lip sync)

# 3. Generate
python main.py "A lighthouse keeper befriends a stranded whale" --duration 30

# 4. Inspect
ls data/outputs/<project_id>/video/shots/
#   scene_1_est.mp4   <- real SVD clip with rippling water motion
#   scene_1_l1.mp4    <- narrator over establishing
#   scene_1_l2.mp4    <- Aria portrait, lip-synced via SadTalker
#   ...

# 5. Edit + iterate (free, no API cost)
python main.py edit <project_id>
edit> apply vintage filter
edit> make scene 2 darker
edit> revert 1
edit> quit

# 6. Done
open data/outputs/<project_id>/final_output_subtitled.mp4
```
