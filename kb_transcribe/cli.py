"""kb-transcribe: videos, audio files, or URLs in -> clean Markdown transcripts out."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

MEDIA_EXTS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts",
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma",
}

PARAGRAPH_GAP_S = 2.0      # a pause this long starts a new paragraph
PARAGRAPH_MAX_CHARS = 900  # soft cap; break at the next sentence end past this

EXAMPLES = """\
examples:
  transcribe "https://www.youtube.com/watch?v=..."
  transcribe "https://www.youtube.com/playlist?list=..."
  transcribe lecture.mp4 podcast.mp3 D:\\videos
  transcribe --model large-v3 --timestamps --srt important-talk.mp4
"""


# --------------------------------------------------------------- CUDA setup

def _setup_cuda_libs() -> None:
    """Make pip-installed cuBLAS/cuDNN visible to ctranslate2.

    Must run before faster_whisper (ctranslate2) is imported.
    """
    lib_dirs: list[Path] = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        for loc in spec.submodule_search_locations:
            for sub in ("bin", "lib"):
                d = Path(loc) / sub
                if d.is_dir():
                    lib_dirs.append(d)

    if sys.platform == "win32":
        for d in lib_dirs:
            os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d}{os.pathsep}" + os.environ.get("PATH", "")
    else:
        libs = [p for d in lib_dirs for p in sorted(d.glob("*.so.*"))]
        for _ in range(2):  # two passes: inter-library load order matters
            for lib in libs:
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


# ------------------------------------------------------------------ helpers

def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:max_len].strip(" .") or "untitled"


def yaml_str(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def is_url(arg: str) -> bool:
    return re.match(r"https?://", arg, re.IGNORECASE) is not None


# ------------------------------------------------------------------- inputs

def expand_url(url: str) -> list[dict]:
    """Resolve a URL into one job per video (playlists are flattened)."""
    from yt_dlp import YoutubeDL

    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        raise RuntimeError("could not resolve URL")
    if info.get("_type") == "playlist":
        jobs = []
        for e in info.get("entries") or []:
            if not e:
                continue
            jobs.append({
                "kind": "url",
                "url": e.get("url") or e.get("webpage_url"),
                "id": e.get("id"),
                "title": e.get("title"),
            })
        return jobs
    return [{
        "kind": "url",
        "url": info.get("webpage_url") or url,
        "id": info.get("id"),
        "title": info.get("title"),
    }]


def collect_jobs(inputs: list[str]) -> list[dict]:
    jobs: list[dict] = []
    for arg in inputs:
        if is_url(arg):
            try:
                found = expand_url(arg)
                if not found:
                    jobs.append({"kind": "error", "source": arg,
                                 "error": "no videos found at URL"})
                jobs.extend(found)
            except Exception as e:
                jobs.append({"kind": "error", "source": arg, "error": str(e)})
            continue
        p = Path(arg)
        if p.is_dir():
            media = sorted(q for q in p.iterdir()
                           if q.suffix.lower() in MEDIA_EXTS)
            if not media:
                jobs.append({"kind": "error", "source": arg,
                             "error": "no media files in directory"})
            jobs.extend({"kind": "local", "path": q} for q in media)
        elif p.is_file():
            jobs.append({"kind": "local", "path": p})
        else:
            jobs.append({"kind": "error", "source": arg,
                         "error": "file not found"})
    return jobs


def existing_output(outdir: Path, job: dict) -> Path | None:
    if job["kind"] == "url" and job.get("id"):
        tag = f"[{job['id']}].md"
        for name in os.listdir(outdir):
            if name.endswith(tag):
                return outdir / name
    elif job["kind"] == "local":
        md = outdir / (sanitize_filename(job["path"].stem) + ".md")
        if md.exists():
            return md
    return None


def download_audio(url: str, tmpdir: str) -> tuple[Path, dict]:
    from yt_dlp import YoutubeDL

    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "retries": 3,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if info and "entries" in info:
        info = (info["entries"] or [None])[0]
    if not info:
        raise RuntimeError("download failed")
    downloads = info.get("requested_downloads") or []
    path = downloads[0].get("filepath") if downloads else None
    if not path or not os.path.exists(path):
        raise RuntimeError("downloaded file not found")
    meta = {
        "title": info.get("title") or "untitled",
        "id": info.get("id"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": info.get("upload_date"),  # YYYYMMDD
        "url": info.get("webpage_url") or url,
    }
    return Path(path), meta


# -------------------------------------------------------------------- model

class Transcriber:
    def __init__(self, model_name: str, device_pref: str, batch_size: int):
        self.model_name = model_name
        self.device_pref = device_pref
        self.batch_size = batch_size
        self.device: str | None = None
        self.model = None
        self.batched = None

    def load(self) -> None:
        if self.model is not None:
            return
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        log(f"loading model '{self.model_name}' "
            "(first run downloads the weights)...")
        if self.device_pref in ("auto", "cuda"):
            try:
                self.model = WhisperModel(self.model_name, device="cuda",
                                          compute_type="float16")
                self.device = "cuda"
            except Exception as e:
                if self.device_pref == "cuda":
                    raise
                log(f"  ! CUDA unavailable ({e}); using CPU")
        if self.model is None:
            self.model = WhisperModel(self.model_name, device="cpu",
                                      compute_type="int8")
            self.device = "cpu"
        if self.device == "cuda":
            self.batched = BatchedInferencePipeline(model=self.model)
        log(f"model ready on {self.device}")

    def force_cpu(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(self.model_name, device="cpu",
                                  compute_type="int8")
        self.device = "cpu"
        self.batched = None

    def transcribe(self, path: Path, language: str | None):
        self.load()
        kwargs = dict(
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        if self.batched is not None:
            try:
                return self.batched.transcribe(
                    str(path), batch_size=self.batch_size, **kwargs)
            except TypeError:
                # parameter drift between faster-whisper versions
                pass
        return self.model.transcribe(
            str(path), condition_on_previous_text=False, **kwargs)


def consume_segments(segments, total: float | None) -> list:
    """Drain the segment generator (this is where the compute happens)."""
    show = sys.stderr.isatty() and total
    out = []
    for seg in segments:
        out.append(seg)
        if show:
            pct = min(seg.end / total, 1.0)
            print(f"\r  {fmt_time(seg.end)} / {fmt_time(total)} ({pct:.0%}) ",
                  end="", file=sys.stderr, flush=True)
    if show:
        print("\r" + " " * 44 + "\r", end="", file=sys.stderr, flush=True)
    return out


def run_asr(transcriber: Transcriber, path: Path, language: str | None):
    try:
        segments, info = transcriber.transcribe(path, language)
        return consume_segments(segments, info.duration), info
    except RuntimeError as e:
        cuda_issue = re.search(r"cudnn|cublas|cuda", str(e), re.IGNORECASE)
        if transcriber.device == "cuda" and transcriber.device_pref == "auto" \
                and cuda_issue:
            log(f"  ! CUDA runtime failure ({e}); retrying on CPU")
            transcriber.force_cpu()
            segments, info = transcriber.transcribe(path, language)
            return consume_segments(segments, info.duration), info
        raise


# ------------------------------------------------------------ text cleanup

def collapse_repeats(segs: list, max_repeats: int = 2) -> list:
    """Drop runs of identical consecutive segments (hallucination loops)."""
    out = []
    prev_text = None
    run = 0
    for s in segs:
        t = s.text.strip().lower()
        if not t:
            continue
        if t == prev_text:
            run += 1
        else:
            prev_text, run = t, 1
        if run > max_repeats:
            continue
        out.append(s)
    return out


def build_paragraphs(segs: list) -> list[tuple[float, str]]:
    """Merge segments into (start_time, paragraph_text) pairs."""
    paras: list[tuple[float, str]] = []
    cur: list[str] = []
    cur_start = 0.0
    cur_len = 0
    prev_end = 0.0
    for s in segs:
        text = s.text.strip()
        if not text:
            continue
        overflow = (cur_len >= PARAGRAPH_MAX_CHARS
                    and cur[-1].endswith((".", "!", "?", "…")))
        if cur and (s.start - prev_end >= PARAGRAPH_GAP_S or overflow):
            paras.append((cur_start, " ".join(cur)))
            cur, cur_len = [], 0
        if not cur:
            cur_start = s.start
        cur.append(text)
        cur_len += len(text) + 1
        prev_end = s.end
    if cur:
        paras.append((cur_start, " ".join(cur)))
    return paras


# ------------------------------------------------------------------ outputs

def output_stem(meta: dict) -> str:
    stem = sanitize_filename(meta["title"])
    d = meta.get("upload_date")
    if d and len(d) == 8:
        stem = f"{d[:4]}-{d[4:6]}-{d[6:]} - {stem}"
    if meta.get("id"):
        stem += f" [{meta['id']}]"
    return stem


def write_markdown(path: Path, meta: dict, paras: list, info,
                   model_name: str, timestamps: bool) -> None:
    lines = ["---", f"title: {yaml_str(meta['title'])}"]
    if meta.get("channel"):
        lines.append(f"channel: {yaml_str(meta['channel'])}")
    lines.append(f"source: {yaml_str(meta.get('url') or meta.get('source', ''))}")
    d = meta.get("upload_date")
    if d and len(d) == 8:
        lines.append(f"uploaded: {d[:4]}-{d[4:6]}-{d[6:]}")
    lines.append(f"duration: {fmt_time(info.duration)}")
    lang = info.language
    prob = getattr(info, "language_probability", None)
    if prob:
        lang += f" ({prob:.0%})"
    lines.append(f"language: {yaml_str(lang)}")
    lines.append(f"model: {yaml_str(model_name + ' (faster-whisper)')}")
    lines.append(f"transcribed: {dt.date.today().isoformat()}")
    lines.extend(["---", "", f"# {meta['title']}", ""])
    for start, text in paras:
        lines.append(f"**[{fmt_time(start)}]** {text}" if timestamps else text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, segs: list) -> None:
    blocks = [f"{i}\n{srt_ts(s.start)} --> {srt_ts(s.end)}\n{s.text.strip()}\n"
              for i, s in enumerate(segs, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_json(path: Path, meta: dict, segs: list, info,
               model_name: str) -> None:
    data = {
        "title": meta["title"],
        "source": meta.get("url") or meta.get("source", ""),
        "channel": meta.get("channel"),
        "upload_date": meta.get("upload_date"),
        "language": info.language,
        "duration": round(info.duration, 2),
        "model": model_name,
        "segments": [{"start": round(s.start, 2), "end": round(s.end, 2),
                      "text": s.text.strip()} for s in segs],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


# --------------------------------------------------------------------- main

def process_job(job: dict, transcriber: Transcriber, args,
                outdir: Path) -> str:
    existing = existing_output(outdir, job)
    if existing and not args.force:
        log(f"  skipped, already transcribed: {existing.name}")
        return "skipped"

    if job["kind"] == "url":
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            log("  downloading audio...")
            media_path, meta = download_audio(job["url"], td)
            transcriber.load()
            log(f"  transcribing on {transcriber.device} ...")
            t0 = time.monotonic()
            segs, info = run_asr(transcriber, media_path, args.language)
    else:
        media_path = job["path"]
        meta = {"title": media_path.stem, "source": str(media_path.resolve())}
        transcriber.load()
        log(f"  transcribing on {transcriber.device} ...")
        t0 = time.monotonic()
        segs, info = run_asr(transcriber, media_path, args.language)

    segs = collapse_repeats(segs)
    paras = build_paragraphs(segs)
    if not paras:
        raise RuntimeError("no speech detected")

    stem = output_stem(meta)
    md_path = outdir / f"{stem}.md"
    write_markdown(md_path, meta, paras, info, args.model, args.timestamps)
    if args.srt:
        write_srt(outdir / f"{stem}.srt", segs)
    if args.json_out:
        write_json(outdir / f"{stem}.json", meta, segs, info, args.model)

    elapsed = time.monotonic() - t0
    speed = info.duration / elapsed if elapsed > 0 else 0
    log(f"  -> {md_path}  ({info.language}, {fmt_time(info.duration)} audio, "
        f"{elapsed:.0f}s, {speed:.1f}x real-time)")
    return "done"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Videos, audio files, or URLs in -> "
                    "clean Markdown transcripts out.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+",
                        help="media files, directories, or video/playlist URLs")
    parser.add_argument("-o", "--output-dir", default="transcripts",
                        help="where transcripts go (default: ./transcripts)")
    parser.add_argument("-m", "--model", default="large-v3-turbo",
                        help="whisper model (default: large-v3-turbo; "
                             "use large-v3 for max accuracy)")
    parser.add_argument("-l", "--language", default=None,
                        help="force language code, e.g. en, es "
                             "(default: auto-detect per file)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"],
                        default="auto")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="GPU batch size (default: 8; try 12-16 on 16 GB)")
    parser.add_argument("--timestamps", action="store_true",
                        help="prefix each paragraph with [h:mm:ss]")
    parser.add_argument("--srt", action="store_true",
                        help="also write an .srt subtitle file")
    parser.add_argument("--json", dest="json_out", action="store_true",
                        help="also write segment-level JSON")
    parser.add_argument("-f", "--force", action="store_true",
                        help="re-transcribe even if output already exists")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    _setup_cuda_libs()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(args.inputs)
    real_jobs = [j for j in jobs if j["kind"] != "error"]
    if real_jobs:
        log(f"{len(real_jobs)} item(s) to process")

    transcriber = Transcriber(args.model, args.device, args.batch_size)
    done = skipped = failed = 0
    try:
        for i, job in enumerate(jobs, 1):
            label = (job.get("title") or job.get("url")
                     or str(job.get("path", job.get("source", "?"))))
            log(f"[{i}/{len(jobs)}] {label}")
            if job["kind"] == "error":
                log(f"  ! {job['error']}")
                failed += 1
                continue
            try:
                result = process_job(job, transcriber, args, outdir)
                if result == "skipped":
                    skipped += 1
                else:
                    done += 1
            except Exception as e:
                log(f"  ! failed: {e}")
                failed += 1
    except KeyboardInterrupt:
        log("\ninterrupted")

    log(f"\ndone: {done}  skipped: {skipped}  failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
