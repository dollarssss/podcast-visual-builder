"""
Podcast Auto-Visual Builder (Streamlit, one-file)
Upload an English podcast audio → auto transcript (Whisper) → scene segmentation → B‑roll fetch (Pexels/Unsplash) →
animated montage (MoviePy) with subtitles, waveform, progress bar → export 9:16 MP4.

README (quick):
1) Python 3.10+
2) pip install -r requirements.txt (see REQS below)
3) export PEXELS_API_KEY=...  (get from https://www.pexels.com/api/)
   export UNSPLASH_ACCESS_KEY=... (optional fallback)
4) streamlit run app.py

REQS (install):
streamlit==1.37.0
faster-whisper==1.0.3
moviepy==1.0.3
yake==0.4.8
spacy==3.7.2
numpy==1.26.4
requests==2.32.3
Pillow==10.3.0
librosa==0.10.2.post1
soundfile==0.12.1
python-dotenv==1.0.1
srt==3.5.3
"""

import os
import io
import math
import json
import time
import srt
import base64
import queue
import tempfile
import requests
import numpy as np
import streamlit as st
from PIL import Image
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from faster_whisper import WhisperModel
from moviepy.editor import (
    AudioFileClip, ImageClip, VideoFileClip, CompositeVideoClip,
    TextClip, ColorClip, concatenate_videoclips, vfx
)
from moviepy.video.fx.all import resize
import moviepy.editor as mpy

import librosa
import soundfile as sf

import yake

# --------------------------- UI CONFIG ---------------------------
st.set_page_config(page_title="Podcast Auto-Visual Builder", layout="wide")
st.title("🎬 Podcast Auto‑Visual Builder — Auto B‑roll + Animated Subtitles (9:16)")
st.caption("Upload audio → auto-cut scenes → fetch B‑roll → render YouTube‑ready vertical video. No English required.")

# --------------------------- SETTINGS ---------------------------
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")

DEFAULT_W = 1080
DEFAULT_H = 1920  # 9:16 vertical
FPS = 30
FONT = "Arial"  # replace with a system font available on your machine

@dataclass
class Scene:
    start: float
    end: float
    text: str
    keywords: List[str]
    asset_paths: List[str]

# --------------------------- HELPERS ---------------------------

def save_uploaded_file(uploaded_file, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def transcribe_audio(audio_path: str, language_hint: Optional[str] = "en") -> Tuple[List[dict], str]:
    """Return list of segments with start, end, text; and full text."""
    model_size = st.sidebar.selectbox("Whisper size", ["tiny", "base", "small", "medium"], index=1)
    compute_type = "float16" if os.environ.get("WHISPER_FP16", "1") == "1" else "int8"
    model = WhisperModel(model_size, device="auto", compute_type=compute_type)
    st.info("Transcribing with faster-whisper…")
    segments, info = model.transcribe(audio_path, language=language_hint, vad_filter=True)
    segs = []
    full_text_parts = []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        full_text_parts.append(s.text.strip())
    return segs, " ".join(full_text_parts)


def segment_to_scenes(segments: List[dict], target_len: float = 10.0) -> List[Scene]:
    """Group whisper segments into ~target_len-second scenes, respecting punctuation."""
    scenes: List[Scene] = []
    cur_start = None
    cur_end = None
    cur_text = []

    for s in segments:
        if cur_start is None:
            cur_start = s["start"]
        cur_end = s["end"]
        cur_text.append(s["text"])
        # close scene when enough duration or punctuation hint
        if (cur_end - cur_start) >= target_len or s["text"].endswith((".", "!", "?")):
            scene_text = " ".join(cur_text).strip()
            scenes.append(Scene(start=cur_start, end=cur_end, text=scene_text, keywords=[], asset_paths=[]))
            cur_start, cur_end, cur_text = None, None, []

    if cur_text:
        scene_text = " ".join(cur_text).strip()
        scenes.append(Scene(start=cur_start or 0.0, end=cur_end or (segments[-1]["end"] if segments else 0.0), text=scene_text, keywords=[], asset_paths=[]))

    return scenes


def extract_keywords(text: str, max_kw: int = 6) -> List[str]:
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=max_kw)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
    # light cleanup
    keywords = [k.lower().strip(" ,.;:!?") for k in keywords if len(k) > 2]
    return list(dict.fromkeys(keywords))  # unique, preserve order


def pexels_search_video(query: str, per_page: int = 2) -> List[str]:
    if not PEXELS_API_KEY:
        return []
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/videos/search?query={requests.utils.quote(query)}&per_page={per_page}&orientation=portrait"
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        return []
    data = r.json()
    urls = []
    for v in data.get("videos", [])[:per_page]:
        # pick highest vertical or first file
        files = v.get("video_files", [])
        files_sorted = sorted(files, key=lambda x: x.get("height", 0), reverse=True)
        if files_sorted:
            urls.append(files_sorted[0]["link"])
    return urls


def unsplash_search_images(query: str, per_page: int = 3) -> List[str]:
    if not UNSPLASH_ACCESS_KEY:
        return []
    url = f"https://api.unsplash.com/search/photos?query={requests.utils.quote(query)}&per_page={per_page}&orientation=portrait&client_id={UNSPLASH_ACCESS_KEY}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return []
    data = r.json()
    return [res["urls"]["regular"] for res in data.get("results", [])[:per_page]]


def download_to_temp(url: str, suffix: str) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
            with open(path, "wb") as f:
                f.write(resp.content)
            return path
    except Exception as e:
        st.warning(f"Download failed: {e}")
    return None


def build_waveform_overlay(audio_path: str, duration: float, width=DEFAULT_W, height=200, color=(255,255,255)) -> ImageClip:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    # take portion proportional to duration (assume full length render)
    img = Image.new("RGBA", (width, height), (0,0,0,0))
    draw = Image.fromarray(np.array(img))
    # Simple line waveform: sample to width
    idx = np.linspace(0, len(y)-1, num=width).astype(int)
    amp = (y[idx] * 0.4 + 0.5)  # normalize roughly
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    for x, a in enumerate(amp):
        h = int(a * height)
        y1 = max(0, (height//2) - h//2)
        y2 = min(height, y1 + h)
        arr[y1:y2, x] = (*color, 160)
    return ImageClip(arr, ismask=False).set_duration(duration)


def make_subtitle_clips(scene: Scene, start_t: float, end_t: float, w=DEFAULT_W) -> List[TextClip]:
    subtitle = scene.text
    max_chars = 48
    lines = []
    chunk = ""
    for word in subtitle.split():
        if len(chunk) + len(word) + 1 <= max_chars:
            chunk += (" " if chunk else "") + word
        else:
            lines.append(chunk)
            chunk = word
    if chunk:
        lines.append(chunk)
    txt = "\n".join(lines)
    style = st.session_state.get("subtitle_style", {
        "fontsize": 48,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 2
    })
    txt_clip = TextClip(txt, font=FONT, method="caption", size=(w-120, None), align='center', **style).set_duration(end_t - start_t)
    txt_clip = txt_clip.set_position(("center", DEFAULT_H - 420)).crossfadein(0.2).crossfadeout(0.2)
    return [txt_clip]


def build_background_clip(asset_paths: List[str], duration: float) -> mpy.VideoClip:
    clips = []
    remaining = duration
    for p in asset_paths:
        if p.lower().endswith(('.mp4', '.mov', '.webm')):
            try:
                vc = VideoFileClip(p).without_audio()
                if vc.duration > remaining + 0.1:
                    vc = vc.subclip(0, remaining)
                vc = vc.fx(vfx.loop, duration=min(remaining, vc.duration))
                vc = vc.resize(height=DEFAULT_H).set_position("center")
                clips.append(vc)
                remaining -= vc.duration
            except Exception as e:
                st.warning(f"Video asset error: {e}")
        else:
            try:
                ic = ImageClip(p).set_duration(min(4.0, remaining))
                ic = ic.resize(height=DEFAULT_H).set_position("center")
                # Ken Burns subtle zoom
                ic = ic.fx(vfx.zoom_in, 1.05)
                clips.append(ic)
                remaining -= ic.duration
            except Exception as e:
                st.warning(f"Image asset error: {e}")
        if remaining <= 0.1:
            break
    if not clips:
        # fallback color background
        clips = [ColorClip((DEFAULT_W, DEFAULT_H), color=(10,10,10)).set_duration(duration)]
    return concatenate_videoclips(clips).set_duration(duration)


def progress_bar_overlay(duration: float) -> mpy.VideoClip:
    bar_h = 8
    def make_frame(t):
        prog = min(1.0, max(0.0, t / duration))
        arr = np.zeros((bar_h, DEFAULT_W, 4), dtype=np.uint8)
        w = int(DEFAULT_W * prog)
        arr[:, :w] = (255, 255, 255, 200)
        return arr
    return mpy.VideoClip(make_frame, duration=duration).set_position((0, DEFAULT_H-12))


def render_video(audio_path: str, scenes: List[Scene], out_path: str) -> str:
    audio = AudioFileClip(audio_path)
    total_dur = audio.duration

    bg_clips = []
    overlays = []

    for sc in scenes:
        seg_dur = sc.end - sc.start
        bg = build_background_clip(sc.asset_paths, seg_dur)
        # subtle vignette layer
        vignette = ColorClip((DEFAULT_W, DEFAULT_H), color=(0,0,0)).set_opacity(0.15).set_duration(seg_dur)
        # subtitles
        subs = make_subtitle_clips(sc, sc.start, sc.end)
        composite = CompositeVideoClip([bg, vignette, *subs], size=(DEFAULT_W, DEFAULT_H)).set_duration(seg_dur)
        bg_clips.append(composite)

    video = concatenate_videoclips(bg_clips, method="compose")
    video = video.set_audio(audio)

    # top overlays: waveform + progress bar
    try:
        wave = build_waveform_overlay(audio_path, duration=video.duration, width=DEFAULT_W, height=180, color=(255,255,255))
        wave = wave.set_position((0, 60)).set_opacity(0.35)
        video = CompositeVideoClip([video, wave, progress_bar_overlay(video.duration)], size=(DEFAULT_W, DEFAULT_H))
    except Exception as e:
        st.warning(f"Waveform overlay failed: {e}")

    video.write_videofile(out_path, fps=FPS, codec='libx264', audio_codec='aac', threads=4, preset='medium')
    return out_path

# --------------------------- APP UI ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    scene_len = st.slider("Target scene length (sec)", 6, 18, 10, 1)
    st.session_state["subtitle_style"] = {
        "fontsize": st.slider("Subtitle font size", 32, 72, 48, 2),
        "color": st.color_picker("Subtitle color", "#FFFFFF"),
        "stroke_color": st.color_picker("Outline color", "#000000"),
        "stroke_width": st.slider("Outline width", 0, 6, 2, 1)
    }
    st.markdown("**Aspect:** 9:16 (1080×1920). For 16:9, change DEFAULT_W/H in code.")
    st.markdown("**APIs:** Pexels (video), Unsplash (image fallback). Provide keys as env vars.")

col1, col2 = st.columns([1,1])

with col1:
    audio_u = st.file_uploader("Upload podcast audio (mp3/wav/m4a)", type=["mp3", "wav", "m4a", "aac"])
    music_u = st.file_uploader("Optional background music (mp3/wav)", type=["mp3", "wav"], help="Will mix at -18 LUFS approx (todo)")

with col2:
    st.write("API status:")
    st.code({"PEXELS": bool(PEXELS_API_KEY), "UNSPLASH": bool(UNSPLASH_ACCESS_KEY)}, language="json")

if audio_u:
    audio_path = save_uploaded_file(audio_u, suffix=f".{audio_u.type.split('/')[-1]}")
    st.audio(audio_u)

    if st.button("▶️ Generate Visual Plan + Assets"):
        segments, full_text = transcribe_audio(audio_path)
        scenes = segment_to_scenes(segments, target_len=float(scene_len))
        for sc in scenes:
            sc.keywords = extract_keywords(sc.text, max_kw=6)
            assets = []
            # Try composite queries: top-2 keywords & bigram
            queries = []
            if len(sc.keywords) >= 2:
                queries.append(" ".join(sc.keywords[:2]))
            queries += sc.keywords[:2]
            used = set()
            for q in queries:
                if q in used:
                    continue
                used.add(q)
                vids = pexels_search_video(q, per_page=1)
                if vids:
                    for vurl in vids:
                        vp = download_to_temp(vurl, ".mp4")
                        if vp:
                            assets.append(vp)
                if len(assets) < 1:  # fallback images
                    imgs = unsplash_search_images(q, per_page=1)
                    for iurl in imgs:
                        ip = download_to_temp(iurl, ".jpg")
                        if ip:
                            assets.append(ip)
                if len(assets) >= 2:
                    break
            sc.asset_paths = assets

        st.success(f"Scenes: {len(scenes)}. Keywords & assets fetched.")
        st.json([asdict(s) for s in scenes])
        st.session_state["scenes"] = scenes
        st.session_state["audio_path"] = audio_path

if "scenes" in st.session_state and st.button("🎞️ Render Video (MP4)"):
    out_path = os.path.join(tempfile.gettempdir(), f"render_{int(time.time())}.mp4")
    try:
        res = render_video(st.session_state["audio_path"], st.session_state["scenes"], out_path)
        st.video(res)
        with open(res, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:video/mp4;base64,{b64}" download="podcast_visual.mp4">⬇️ Download MP4</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Render failed: {e}")

# --------------------------- EXPORT SCENE JSON ---------------------------
st.divider()
st.subheader("📦 Export Scene JSON (for manual editing or other renderers)")
if "scenes" in st.session_state:
    export = json.dumps([asdict(s) for s in st.session_state["scenes"]], ensure_ascii=False, indent=2)
    st.code(export, language="json")
else:
    st.info("Generate scenes first.")

# --------------------------- NOTES ---------------------------
st.markdown(
"""
**Notes & Tips**
- This app auto-picks B‑roll by keywords. For perfect accuracy, edit the exported Scene JSON and re-render.
- Improve hook/retention: keep scene length 6–10s, add strong first sentence in audio.
- For Russian overlays or branded lower-thirds, replace TextClip with your fonts and add logos.
- If you need AI text-to-video instead of stock B‑roll, feed the exported JSON into your Veo 3 / Gemini / Flow pipeline.
"""
)
