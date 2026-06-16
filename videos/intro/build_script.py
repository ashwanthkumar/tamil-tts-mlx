"""Build the intro video assets: generate TTS voice-over + example clips, lay out a timeline,
write src/script.json (consumed by the Remotion composition), and synthesize length-matched music.

Run from repo root:  uv run python videos/intro/build_script.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
PUB = Path(__file__).resolve().parent / "public"
SRC = Path(__file__).resolve().parent / "src"
FPS = 30
MODEL = "models/tamil_ns"

# (tamil spoken by the model, english subtitle). First block = intro narration; rest = examples.
SEGMENTS = [
    ("வணக்கம், இது ஒரு தமிழ் பேச்சு தொகுப்பு மாதிரி.", "Hello — this is a Tamil text-to-speech model."),
    ("நீங்கள் கேட்கும் இந்தக் குரல் முழுவதும் கணினியால் உருவாக்கப்பட்டது.", "Every voice you hear is fully machine-generated."),
    ("இது சிறிய மாதிரி, ஆனால் எந்த சாதனத்திலும் வேகமாக இயங்கும்.", "It's a small model — yet runs fast on any device."),
    ("தமிழ் ஒரு பழமையான, அழகான மொழி.", "Tamil is an ancient, beautiful language."),
    ("இன்று வானிலை மிகவும் அழகாக இருக்கிறது.", "The weather is lovely today."),
    ("எனக்கு புத்தகங்கள் படிக்க மிகவும் பிடிக்கும்.", "I really enjoy reading books."),
    ("காலை உணவு உடல் நலத்திற்கு மிகவும் முக்கியம்.", "Breakfast is very important for health."),
    ("நீங்கள் எந்த வாக்கியத்தையும் இங்கே எழுதலாம்.", "You can type any sentence here."),
    ("செயற்கை நுண்ணறிவு தமிழிலும் வளர்ந்து வருகிறது.", "AI is growing in Tamil too."),
    ("மழை பெய்தால் குடை எடுத்துச் செல்லுங்கள்.", "If it rains, take an umbrella."),
    ("இசை மனதிற்கு மகிழ்ச்சியைத் தருகிறது.", "Music brings joy to the mind."),
    ("கல்வி வாழ்க்கையை மாற்றும் சக்தி கொண்டது.", "Education has the power to change life."),
    ("வருகைக்கு நன்றி, மீண்டும் சந்திப்போம்.", "Thanks for watching — see you again!"),
]

INTRO = 80          # title frames before first clip
GAP = 16            # frames between clips
SHOWCASE_TITLE = 80  # frames for the "controllable prosody" title card
# closing thank-you voice-over, spoken by the model (pure Tamil so it's in-vocab)
ACK_TEXT = "இந்தத் தரவுத்தொகுப்பைப் பகிர்ந்த சென்னை ஐ.ஐ.டி. குழுவிற்கு எங்களின் மிக்க நன்றி."

# v0.2 prosody showcase: one sentence, re-spoken with different speed/pitch/energy knobs.
SHOWCASE_TEXT = "தமிழ் ஒரு அழகான மொழி."   # "Tamil is a beautiful language."
SHOWCASE = [
    {"label": "Natural", "sub": "speed 1.0 · pitch 1.0 · energy 1.0"},
    {"label": "Lower pitch", "sub": "pitch 0.85", "pitch": 0.85},
    {"label": "Higher pitch", "sub": "pitch 1.15", "pitch": 1.15},
    {"label": "Softer", "sub": "energy 0.7", "energy": 0.7},
    {"label": "Faster", "sub": "speed 1.25", "speed": 1.25},
]


def gen(text: str, out: Path, speed: float = 1.0, pitch: float = 1.0, energy: float = 1.0):
    subprocess.run(["uv", "run", "python", "-m", "tamiltts.mlx.onnx_infer_ns",
                    "-m", MODEL, "--text", text, "-o", str(out),
                    "--speed", str(speed), "--pitch", str(pitch), "--energy", str(energy)],
                   cwd=ROOT, check=True, capture_output=True)


def main():
    PUB.mkdir(parents=True, exist_ok=True)
    segs = []
    cursor = INTRO
    for i, (ta, en) in enumerate(SEGMENTS):
        f = PUB / f"seg{i:02d}.wav"
        gen(ta, f)
        dur = sf.info(f).frames / sf.info(f).samplerate
        df = round(dur * FPS)
        segs.append({"file": f.name, "tamil": ta, "en": en, "start": cursor, "dur": df})
        print(f"seg{i:02d}: {dur:.2f}s -> {df}f  start={cursor}  | {en}")
        cursor += df + GAP

    # v0.2 prosody showcase: title card, then the same line at different knob settings
    showcase_title_start = cursor + GAP
    cursor = showcase_title_start + SHOWCASE_TITLE
    shows = []
    for i, sc in enumerate(SHOWCASE):
        f = PUB / f"show{i:02d}.wav"
        gen(SHOWCASE_TEXT, f, speed=sc.get("speed", 1.0), pitch=sc.get("pitch", 1.0), energy=sc.get("energy", 1.0))
        dur = sf.info(f).frames / sf.info(f).samplerate
        df = round(dur * FPS)
        shows.append({"file": f.name, "label": sc["label"], "sub": sc["sub"], "start": cursor, "dur": df})
        print(f"show{i:02d}: {dur:.2f}s -> {df}f  start={cursor}  | {sc['label']} ({sc['sub']})")
        cursor += df + GAP

    # closing thank-you voice-over
    ack_file = PUB / "ack.wav"
    gen(ACK_TEXT, ack_file)
    ack_audio_dur = sf.info(ack_file).frames / sf.info(ack_file).samplerate
    ack_audio_frames = round(ack_audio_dur * FPS)
    print(f"ack: {ack_audio_dur:.2f}s -> {ack_audio_frames}f")
    ack_start = cursor + GAP
    ack_window = ack_audio_frames + 75          # lead-in + tail around the voice-over
    total = ack_start + ack_window
    script = {"fps": FPS, "width": 1280, "height": 720, "intro": INTRO,
              "ack_start": ack_start, "ack_audio": ack_file.name, "ack_lead": 10,
              "total": total, "segments": segs,
              "showcase_title_start": showcase_title_start, "showcase_title": SHOWCASE_TITLE,
              "showcase_text": SHOWCASE_TEXT, "showcase": shows}
    (SRC / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"total {total} frames = {total/FPS:.1f}s -> wrote src/script.json")

    # length-matched ambient music (C-major sine pad, soft, faded)
    dur = total / FPS
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=130.81:duration={dur}",
        "-f", "lavfi", "-i", f"sine=frequency=164.81:duration={dur}",
        "-f", "lavfi", "-i", f"sine=frequency=196.00:duration={dur}",
        "-f", "lavfi", "-i", f"sine=frequency=261.63:duration={dur}",
        "-filter_complex",
        f"[0][1][2][3]amix=inputs=4,tremolo=f=0.18:d=0.5,lowpass=f=1000,"
        f"afade=t=in:ss=0:d=3,afade=t=out:st={dur-3:.2f}:d=3,volume=0.6[a]",
        "-map", "[a]", "-ac", "1", "-ar", "22050", str(PUB / "music.wav"),
    ], check=True)
    print("wrote public/music.wav")


if __name__ == "__main__":
    main()
