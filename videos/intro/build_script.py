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
ACK = 180           # frames for the closing acknowledgment / thank-you slide (~6s)


def gen(text: str, out: Path):
    subprocess.run(["uv", "run", "python", "-m", "tamiltts.mlx.onnx_infer_ns",
                    "-m", MODEL, "--text", text, "-o", str(out)],
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
    ack_start = cursor
    total = ack_start + ACK
    script = {"fps": FPS, "width": 1280, "height": 720, "intro": INTRO,
              "ack_start": ack_start, "total": total, "segments": segs}
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
