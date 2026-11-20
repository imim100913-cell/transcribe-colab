"""Core transcription logic for transcribe-colab.

Uses faster-whisper on Google Colab GPU (float16 with automatic int8 fallback)
and generates Muse-compatible .txt and timestamped .md outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterable, Protocol, Sequence


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".mp3", ".m4a", ".wav", ".mp4")
OUTPUT_DIR_NAME: str = "transcribed"

DEFAULT_JA_MODEL: str = "kotoba-tech/kotoba-whisper-v2.0-faster"
DEFAULT_MULTILINGUAL_MODEL: str = "Systran/faster-whisper-large-v3"

DEFAULT_MODELS: dict[str, str] = {
    "ja": DEFAULT_JA_MODEL,
    "en": DEFAULT_MULTILINGUAL_MODEL,
    "auto": DEFAULT_MULTILINGUAL_MODEL,
}

SUPPORTED_LANGUAGES: tuple[str, ...] = ("ja", "en", "auto")


class SegmentLike(Protocol):
    """Protocol representing a faster-whisper transcription segment."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptSegment:
    """Normalized segment record."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class FileTranscribeResult:
    """Result record for a single audio file."""

    audio_path: Path
    txt_path: Path
    md_path: Path
    status: str  # "processed" | "skipped" | "failed"
    elapsed_seconds: float = 0.0
    audio_duration_seconds: float = 0.0
    detected_language: str | None = None
    segment_count: int = 0
    error: str | None = None


@dataclass
class TranscribeSummary:
    """Summary of a batch transcription run."""

    input_dir: Path
    model_name: str
    language: str
    device: str
    compute_type: str
    total_files: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    elapsed_seconds: float = 0.0
    results: list[FileTranscribeResult] = field(default_factory=list)


def format_timestamp(seconds: float) -> str:
    """Format seconds into `[MM:SS]` (or `[HH:MM:SS]` when >= 1 hour).

    Negative values are clamped to `[00:00]`.
    """
    clamped = max(0.0, float(seconds))
    total_seconds = int(clamped)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def resolve_output_paths(
    audio_path: Path | str,
    output_dir_name: str = OUTPUT_DIR_NAME,
) -> tuple[Path, Path]:
    """Return `(txt_path, md_path)` under `<audio_parent>/<output_dir_name>/`."""
    src = Path(audio_path)
    out_dir = src.parent / output_dir_name
    return out_dir / f"{src.stem}.txt", out_dir / f"{src.stem}.md"


def should_skip_file(
    audio_path: Path | str,
    overwrite: bool = False,
    output_dir_name: str = OUTPUT_DIR_NAME,
) -> bool:
    """Return True when both `.txt` and `.md` outputs exist and overwrite is False."""
    if overwrite:
        return False
    txt_path, md_path = resolve_output_paths(audio_path, output_dir_name=output_dir_name)
    return txt_path.is_file() and md_path.is_file()


def normalize_language_setting(language: str | None) -> str:
    """Normalize user language setting to `'ja'`, `'en'`, `'auto'`, or lowercase code."""
    if language is None:
        return "auto"
    cleaned = language.strip().lower()
    return cleaned if cleaned else "auto"


def resolve_language_code(language: str | None) -> str | None:
    """Convert user language setting into faster-whisper `language` argument."""
    normalized = normalize_language_setting(language)
    if normalized == "auto":
        return None
    return normalized


def resolve_model_name(language: str | None, model: str | None = None) -> str:
    """Resolve Hugging Face / faster-whisper model name from language and override."""
    if model is not None:
        cleaned_model = model.strip()
        if cleaned_model and cleaned_model.lower() != "auto":
            return cleaned_model

    normalized_lang = normalize_language_setting(language)
    return DEFAULT_MODELS.get(normalized_lang, DEFAULT_MULTILINGUAL_MODEL)


def resolve_chunk_length(model_name: str, chunk_length: int | None = None) -> int | None:
    """Return recommended `chunk_length` for the selected model."""
    if chunk_length is not None:
        return chunk_length
    if "kotoba" in model_name.lower():
        # kotoba-tech/kotoba-whisper-v2.0-faster model card recommends chunk_length=15
        return 15
    return None


def find_audio_files(
    input_dir: Path | str,
    extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
    output_dir_name: str = OUTPUT_DIR_NAME,
) -> list[Path]:
    """Recursively find supported audio files while skipping `transcribed/` dirs."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    found: list[Path] = []

    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d != output_dir_name and not d.startswith(".")
        )
        current_path = Path(current_root)
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            file_path = current_path / filename
            if file_path.suffix.lower() in normalized_exts:
                found.append(file_path)

    return sorted(found, key=lambda p: str(p))


def format_transcript_txt(segments: Iterable[SegmentLike]) -> str:
    """Format segments into plain text for `.txt` output."""
    lines = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def format_transcript_md(segments: Iterable[SegmentLike]) -> str:
    """Format segments into line-leading `[MM:SS] text` (or `[HH:MM:SS] text`) lines."""
    formatted_lines = []
    for seg in segments:
        text = seg.text.strip() if seg.text else ""
        if not text:
            continue
        ts = format_timestamp(seg.start)
        formatted_lines.append(f"{ts} {text}")

    if not formatted_lines:
        return ""
    return "\n".join(formatted_lines) + "\n"


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text atomically to avoid partial outputs on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def detect_cuda_available() -> bool:
    """Return True if CTranslate2 detects at least one CUDA device."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _default_model_factory(model_name: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def load_whisper_model(
    model_name: str,
    device: str = "auto",
    model_factory: Callable[..., Any] | None = None,
    cuda_available_fn: Callable[[], bool] | None = None,
) -> tuple[Any, str, str]:
    """Load WhisperModel with GPU float16 priority and automatic int8 fallback.

    Returns `(model, resolved_device, resolved_compute_type)`.
    """
    factory = model_factory or _default_model_factory
    has_cuda = cuda_available_fn or detect_cuda_available

    target_device = device.strip().lower()
    if target_device == "auto":
        target_device = "cuda" if has_cuda() else "cpu"

    if target_device == "cuda":
        try:
            print(f"[INFO] Loading model '{model_name}' on cuda (compute_type=float16)...", flush=True)
            model = factory(model_name, device="cuda", compute_type="float16")
            return model, "cuda", "float16"
        except Exception as exc:
            print(
                f"[WARN] float16 initialization failed ({exc}); falling back to int8 on cuda...",
                flush=True,
            )
            model = factory(model_name, device="cuda", compute_type="int8")
            return model, "cuda", "int8"

    if target_device == "cpu":
        print(
            "[WARN] CUDA GPU is not available; loading model on CPU with compute_type=int8.",
            flush=True,
        )
        model = factory(model_name, device="cpu", compute_type="int8")
        return model, "cpu", "int8"

    raise ValueError(f"Unsupported device '{device}'. Expected 'auto', 'cuda', or 'cpu'.")


def transcribe_file(
    model: Any,
    audio_path: Path | str,
    model_name: str,
    language: str | None = "ja",
    vad_filter: bool = True,
    beam_size: int = 5,
    condition_on_previous_text: bool = False,
    chunk_length: int | None = None,
    output_dir_name: str = OUTPUT_DIR_NAME,
) -> FileTranscribeResult:
    """Transcribe a single audio file and write `.txt` and `.md` atomically."""
    src = Path(audio_path)
    txt_path, md_path = resolve_output_paths(src, output_dir_name=output_dir_name)
    lang_code = resolve_language_code(language)
    resolved_chunk = resolve_chunk_length(model_name, chunk_length=chunk_length)

    transcribe_kwargs: dict[str, Any] = {
        "language": lang_code,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "condition_on_previous_text": condition_on_previous_text,
    }
    if resolved_chunk is not None:
        transcribe_kwargs["chunk_length"] = resolved_chunk

    start_time = time.monotonic()
    raw_segments, info = model.transcribe(str(src), **transcribe_kwargs)
    segments = [
        TranscriptSegment(start=float(seg.start), end=float(seg.end), text=str(seg.text))
        for seg in raw_segments
    ]
    elapsed = time.monotonic() - start_time

    txt_content = format_transcript_txt(segments)
    md_content = format_transcript_md(segments)

    atomic_write_text(txt_path, txt_content)
    atomic_write_text(md_path, md_content)

    audio_duration = float(getattr(info, "duration", 0.0) or 0.0)
    detected_lang = getattr(info, "language", None)
    non_empty_count = sum(1 for s in segments if s.text.strip())

    return FileTranscribeResult(
        audio_path=src,
        txt_path=txt_path,
        md_path=md_path,
        status="processed",
        elapsed_seconds=elapsed,
        audio_duration_seconds=audio_duration,
        detected_language=detected_lang,
        segment_count=non_empty_count,
    )


def transcribe_directory(
    input_dir: Path | str,
    language: str = "ja",
    model: str | None = "auto",
    overwrite: bool = False,
    device: str = "auto",
    vad_filter: bool = True,
    beam_size: int = 5,
    condition_on_previous_text: bool = False,
    chunk_length: int | None = None,
    output_dir_name: str = OUTPUT_DIR_NAME,
    model_factory: Callable[..., Any] | None = None,
    cuda_available_fn: Callable[[], bool] | None = None,
) -> TranscribeSummary:
    """Recursively transcribe all supported audio files in `input_dir`."""
    root = Path(input_dir)
    normalized_lang = normalize_language_setting(language)
    resolved_model_name = resolve_model_name(normalized_lang, model=model)
    audio_files = find_audio_files(root, output_dir_name=output_dir_name)

    print(f"[INFO] Input directory : {root}", flush=True)
    print(f"[INFO] Target files    : {len(audio_files)}", flush=True)
    print(f"[INFO] Language setting: {normalized_lang}", flush=True)
    print(f"[INFO] Selected model  : {resolved_model_name}", flush=True)
    print(f"[INFO] Overwrite mode  : {overwrite}", flush=True)

    if not audio_files:
        print("[INFO] No supported audio files (.mp3, .m4a, .wav, .mp4) found.", flush=True)
        return TranscribeSummary(
            input_dir=root,
            model_name=resolved_model_name,
            language=normalized_lang,
            device=device,
            compute_type="n/a",
            total_files=0,
        )

    pending_files = [
        p for p in audio_files
        if not should_skip_file(p, overwrite=overwrite, output_dir_name=output_dir_name)
    ]

    if not pending_files:
        print("[INFO] All files are already transcribed. Nothing to do.", flush=True)
        skipped_results = [
            FileTranscribeResult(
                audio_path=p,
                txt_path=resolve_output_paths(p, output_dir_name=output_dir_name)[0],
                md_path=resolve_output_paths(p, output_dir_name=output_dir_name)[1],
                status="skipped",
            )
            for p in audio_files
        ]
        return TranscribeSummary(
            input_dir=root,
            model_name=resolved_model_name,
            language=normalized_lang,
            device=device,
            compute_type="n/a",
            total_files=len(audio_files),
            skipped_count=len(audio_files),
            results=skipped_results,
        )

    batch_start = time.monotonic()
    loaded_model, resolved_device, resolved_compute = load_whisper_model(
        resolved_model_name,
        device=device,
        model_factory=model_factory,
        cuda_available_fn=cuda_available_fn,
    )
    print(
        f"[INFO] Model ready on {resolved_device} (compute_type={resolved_compute}).",
        flush=True,
    )

    summary = TranscribeSummary(
        input_dir=root,
        model_name=resolved_model_name,
        language=normalized_lang,
        device=resolved_device,
        compute_type=resolved_compute,
        total_files=len(audio_files),
    )

    total = len(audio_files)
    for idx, audio_path in enumerate(audio_files, start=1):
        rel_path = audio_path.relative_to(root)
        txt_path, md_path = resolve_output_paths(audio_path, output_dir_name=output_dir_name)

        if should_skip_file(audio_path, overwrite=overwrite, output_dir_name=output_dir_name):
            print(f"[{idx}/{total}] SKIP (already transcribed): {rel_path}", flush=True)
            summary.skipped_count += 1
            summary.results.append(
                FileTranscribeResult(
                    audio_path=audio_path,
                    txt_path=txt_path,
                    md_path=md_path,
                    status="skipped",
                )
            )
            continue

        print(f"[{idx}/{total}] PROCESSING: {rel_path}", flush=True)
        try:
            result = transcribe_file(
                model=loaded_model,
                audio_path=audio_path,
                model_name=resolved_model_name,
                language=normalized_lang,
                vad_filter=vad_filter,
                beam_size=beam_size,
                condition_on_previous_text=condition_on_previous_text,
                chunk_length=chunk_length,
                output_dir_name=output_dir_name,
            )
            speed_str = ""
            if result.elapsed_seconds > 0 and result.audio_duration_seconds > 0:
                rtf_speed = result.audio_duration_seconds / result.elapsed_seconds
                speed_str = f", {rtf_speed:.1f}x realtime"
            print(
                f"[{idx}/{total}] DONE ({result.elapsed_seconds:.1f}s"
                f", audio={result.audio_duration_seconds:.1f}s{speed_str}"
                f", lang={result.detected_language}"
                f", segments={result.segment_count}) -> {txt_path.name}, {md_path.name}",
                flush=True,
            )
            summary.processed_count += 1
            summary.results.append(result)
        except Exception as exc:
            print(f"[{idx}/{total}] ERROR on {rel_path}: {exc}", flush=True)
            summary.failed_count += 1
            summary.results.append(
                FileTranscribeResult(
                    audio_path=audio_path,
                    txt_path=txt_path,
                    md_path=md_path,
                    status="failed",
                    error=str(exc),
                )
            )

    summary.elapsed_seconds = time.monotonic() - batch_start
    print(
        f"[SUMMARY] total={summary.total_files}, processed={summary.processed_count}, "
        f"skipped={summary.skipped_count}, failed={summary.failed_count}, "
        f"elapsed={summary.elapsed_seconds:.1f}s",
        flush=True,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files recursively with faster-whisper.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing audio files (.mp3, .m4a, .wav, .mp4).",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="Language setting: 'ja', 'en', or 'auto' (default: ja).",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Hugging Face / faster-whisper model name, or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt and .md outputs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Compute device: 'auto', 'cuda', or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Silero VAD filtering.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = transcribe_directory(
        input_dir=args.input_dir,
        language=args.language,
        model=args.model,
        overwrite=args.overwrite,
        device=args.device,
        vad_filter=not args.no_vad,
    )
    return 1 if summary.failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
