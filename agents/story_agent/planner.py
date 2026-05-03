"""Template-based story generator — used when the configured LLM is the mock provider.

Produces a deterministic but plausible 4-scene short film from any input prompt.
The template is intentionally generic so it works for any genre.
"""
from __future__ import annotations
import hashlib
import random
import re
from typing import List

from shared.schemas.story import (
    Character, CharacterRoster, DialogueLine, Scene, ScriptOutput, StoryOutput,
)


_GENRE_HINTS = {
    "scifi": ("sci-fi", ["wonder", "discovery", "humanity"]),
    "space": ("sci-fi", ["wonder", "isolation", "frontier"]),
    "astronaut": ("sci-fi", ["isolation", "discovery", "courage"]),
    "ocean": ("adventure", ["mystery", "exploration"]),
    "forest": ("fantasy", ["mystery", "nature"]),
    "dragon": ("fantasy", ["bravery", "destiny"]),
    "robot": ("sci-fi", ["identity", "humanity"]),
    "ghost": ("horror", ["fear", "memory"]),
    "love": ("romance", ["connection", "longing"]),
    "war": ("drama", ["sacrifice", "loss"]),
    "detective": ("mystery", ["truth", "justice"]),
}


def _detect_genre(prompt: str) -> tuple[str, List[str]]:
    p = prompt.lower()
    for k, (g, themes) in _GENRE_HINTS.items():
        if k in p:
            return g, themes
    return "drama", ["change", "discovery"]


def _seeded_rng(prompt: str) -> random.Random:
    seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def _title_from_prompt(prompt: str) -> str:
    words = [w for w in re.findall(r"\w+", prompt) if len(w) > 3][:5]
    if not words:
        return "An Untitled Short"
    return " ".join(w.capitalize() for w in words)


def template_script(project_id: str, prompt: str, target_duration_s: int = 45) -> ScriptOutput:
    genre, themes = _detect_genre(prompt)
    rng = _seeded_rng(prompt)
    title = _title_from_prompt(prompt)
    logline = (f"In a {genre} world, {prompt.strip().rstrip('.')}, "
               f"unfolding in four short acts.")

    characters = CharacterRoster(characters=[
        Character(
            id="char_narrator",
            name="Narrator",
            role="narrator",
            description="A calm, knowing observer who frames the journey.",
            visual_description="silhouette of an older figure, soft side-light, simple cloak",
            voice_style="warm, measured",
            voice_gender="neutral",
            voice_age="adult",
        ),
        Character(
            id="char_protagonist",
            name="Aria",
            role="protagonist",
            description="Curious and resilient, drawn forward by an inner question.",
            visual_description="young adventurer, weathered jacket, focused expression, "
                               "short brown hair, hazel eyes",
            voice_style="determined, hopeful",
            voice_gender="female",
            voice_age="young",
        ),
        Character(
            id="char_supporting",
            name="Kai",
            role="supporting",
            description="Pragmatic friend whose doubts give voice to the audience.",
            visual_description="lean figure, round glasses, dark coat, kind eyes",
            voice_style="thoughtful, slightly skeptical",
            voice_gender="male",
            voice_age="young",
        ),
    ])

    arc = [
        ("Opening",   "neutral",   "ambient",    "wide establishing shot, soft natural light",  "fade"),
        ("Inciting",  "curious",   "mysterious", "medium shot, cool palette, gentle motion",     "fade"),
        ("Climax",    "tense",     "epic",       "close-up, dramatic backlight, high contrast",  "cut"),
        ("Resolution","reflective","ethereal",   "wide aerial pull-back, golden-hour lighting",  "fade"),
    ]
    seg = max(4, target_duration_s // 4)
    scenes: List[Scene] = []
    for i, (label, tone, mood, camera, transition) in enumerate(arc):
        scene_id = f"scene_{i+1}"
        beat = _scene_beat(label, prompt, rng)
        # Each scene gets the narrator + the protagonist + (sometimes) Kai.
        lines: List[DialogueLine] = []
        lines.append(DialogueLine(
            line_id=f"{scene_id}_l1",
            character_id="char_narrator",
            text=beat["narration"],
            emotion="reflective",
            duration_ms=4500,
        ))
        lines.append(DialogueLine(
            line_id=f"{scene_id}_l2",
            character_id="char_protagonist",
            text=beat["aria"],
            emotion=tone,
            duration_ms=3500,
        ))
        if i in (1, 2):
            lines.append(DialogueLine(
                line_id=f"{scene_id}_l3",
                character_id="char_supporting",
                text=beat["kai"],
                emotion="concerned" if i == 1 else "urgent",
                duration_ms=3500,
            ))
        scenes.append(Scene(
            scene_id=scene_id,
            index=i,
            title=f"{label}: {beat['title']}",
            setting=beat["setting"],
            tone=tone,
            visual_prompt=(
                f"{beat['visual']}, {camera}, cinematic, highly detailed, dramatic lighting"
            ),
            camera=camera,
            duration_ms=seg * 1000,
            dialogue=lines,
            music_mood=mood,
            transition_in=transition,
        ))

    story = StoryOutput(
        project_id=project_id,
        title=title,
        logline=logline,
        synopsis=(f"An exploration inspired by '{prompt}'. "
                  "Across four acts, our protagonist Aria — joined by Kai — "
                  "confronts the question implied by the prompt and emerges changed."),
        genre=genre,
        themes=themes,
        arc="three-act+epilogue",
        target_duration_s=target_duration_s,
    )
    return ScriptOutput(story=story, characters=characters, scenes=scenes)


def _scene_beat(label: str, prompt: str, rng: random.Random) -> dict:
    p = prompt.strip().rstrip(".")
    if label == "Opening":
        return {
            "title": "A world before the spark",
            "setting": "the place where the story begins, calm and ordinary",
            "visual": f"{p}, opening shot, peaceful, establishing the world",
            "narration": f"Every story begins in stillness. Here is where ours starts: {p}.",
            "aria": "Something is about to change. I can feel it.",
            "kai": "Nothing has happened yet. Maybe it never will.",
        }
    if label == "Inciting":
        return {
            "title": "The first sign",
            "setting": "the moment the ordinary breaks",
            "visual": f"{p}, mysterious detail revealed, hint of the unknown",
            "narration": "And then — quietly, almost gently — the world shifted.",
            "aria": "Did you see that? It wasn't there a second ago.",
            "kai": "We should turn back. We don't know what this is.",
        }
    if label == "Climax":
        return {
            "title": "The choice",
            "setting": "at the edge of the unknown, no path back",
            "visual": f"{p}, climactic confrontation, dramatic lighting, peak tension",
            "narration": "There are moments that ask everything of us. This was one.",
            "aria": "If I don't try now, I never will. I have to know.",
            "kai": "Then I'm with you. Whatever this costs.",
        }
    return {
        "title": "What we became",
        "setting": "after, looking back at where we came from",
        "visual": f"{p}, peaceful aftermath, golden hour, hopeful resolution",
        "narration": "And so the story ends — not with an answer, but with a changed way of asking.",
        "aria": "I'll never see it the same way again.",
        "kai": "Neither will I.",
    }
