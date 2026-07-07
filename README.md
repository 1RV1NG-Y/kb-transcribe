# kb-transcribe

Videos, audio files, or URLs in → clean Markdown transcripts out.
faster-whisper (CUDA with automatic CPU fallback) + yt-dlp, one cross-platform CLI.

## Why

YouTube's auto-generated captions are a wall of unpunctuated lowercase —
no sentences, no paragraphs, names and jargon mangled. Fine for following
along while watching; useless for reading, searching, or feeding into notes
and LLM workflows. And a lot of what's worth transcribing (local lectures,
podcasts, recordings) has no captions at all.

This tool produces transcripts worth keeping: Whisper-accurate text with real
punctuation, merged into readable paragraphs, with the video's metadata
(title, channel, date, URL, language) in a YAML header — every video becomes
a self-contained Markdown document ready for a knowledge base, Obsidian
vault, or RAG pipeline. Everything runs locally on your machine; the only
network traffic is fetching the video itself.

## Install

Needs [uv](https://docs.astral.sh/uv/):

- Windows: `winget install astral-sh.uv`
- Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then, from this folder:

```
uv tool install .
```

That puts a `transcribe` command on your PATH. (Or skip installing and run
`uv run transcribe ...` from inside this folder.)

## Use

```
transcribe "https://www.youtube.com/watch?v=..."        # single video
transcribe "https://www.youtube.com/playlist?list=..."  # whole playlist
transcribe lecture.mp4 podcast.mp3 D:\videos            # files and folders
transcribe --model large-v3 --timestamps --srt talk.mp4
```

Transcripts land in `./transcripts/` as Markdown with YAML frontmatter
(title, channel, date, source URL, language, model). Videos that already
have a transcript there are skipped — safe to re-run on a playlist to pick
up only new videos. Use `--force` to redo.

| Flag | Meaning |
|---|---|
| `-o DIR` | output directory (default `./transcripts`) |
| `-m MODEL` | `large-v3-turbo` (default, fast) or `large-v3` (max accuracy) |
| `-l LANG` | force language (`en`, `es`, ...); default auto-detects per file |
| `--translate` | translate the speech to English instead of transcribing verbatim |
| `--timestamps` | prefix each paragraph with `[h:mm:ss]` |
| `--srt` / `--json` | also write subtitles / segment-level JSON |
| `--batch-size N` | GPU batch size, default 8; try 12–16 on 16 GB VRAM |
| `--device cuda\|cpu` | pin the device instead of auto |
| `-f` | re-transcribe even if output exists |

## Notes

- First run downloads model weights (~1.6 GB for `large-v3-turbo`), cached
  under `~/.cache/huggingface` afterwards.
- NVIDIA GPU is used automatically; cuBLAS/cuDNN come from pip wheels, so no
  CUDA toolkit install is needed. Falls back to CPU (int8) if CUDA fails.
- Hallucination guards are on by default: VAD filtering (skips silence and
  music), no conditioning on previous text, repeated-segment collapse.
- `--translate` outputs English regardless of the spoken language, saved as a
  separate `... (translated).md` file so it can coexist with the verbatim
  transcript. English is the only target Whisper supports natively; other
  target languages would need a separate translation step. Translation uses
  `large-v3` by default (turbo was trained transcription-only and translates
  poorly) and runs on the sequential pipeline, so it is slower than
  transcription — normal and expected.
