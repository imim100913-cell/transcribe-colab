"""Unit tests for src/transcribe.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.transcribe import (
    DEFAULT_JA_MODEL,
    DEFAULT_MULTILINGUAL_MODEL,
    find_audio_files,
    format_timestamp,
    format_transcript_md,
    format_transcript_txt,
    load_whisper_model,
    resolve_chunk_length,
    resolve_language_code,
    resolve_model_name,
    resolve_output_paths,
    should_skip_file,
    transcribe_directory,
)


@dataclass
class DummySegment:
    start: float
    end: float
    text: str


@dataclass
class DummyInfo:
    duration: float = 12.0
    language: str = "ja"


class FakeWhisperModel:
    def __init__(self, fail_paths: set[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_paths = fail_paths or set()

    def transcribe(self, audio_path: str, **kwargs: Any):
        self.calls.append({"audio_path": audio_path, "kwargs": kwargs})
        if audio_path in self.fail_paths:
            raise RuntimeError("corrupted audio stream")
        segments = [
            DummySegment(start=0.0, end=4.2, text=" 最初のセグメントです。 "),
            DummySegment(start=65.8, end=70.1, text="二分目付近の発言。"),
        ]
        lang = kwargs.get("language") or "ja"
        return iter(segments), DummyInfo(duration=75.0, language=lang)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-5.0, "[00:00]"),
        (0.0, "[00:00]"),
        (0.9, "[00:00]"),
        (5.4, "[00:05]"),
        (59.99, "[00:59]"),
        (60.0, "[01:00]"),
        (65.2, "[01:05]"),
        (3599.0, "[59:59]"),
        (3661.0, "[61:01]"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_resolve_output_paths(tmp_path: Path) -> None:
    audio = tmp_path / "sub" / "会議 2026.final.m4a"
    txt_path, md_path = resolve_output_paths(audio)
    assert txt_path == tmp_path / "sub" / "transcribed" / "会議 2026.final.txt"
    assert md_path == tmp_path / "sub" / "transcribed" / "会議 2026.final.md"


def test_should_skip_file(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"dummy")
    txt_path, md_path = resolve_output_paths(audio)

    # Neither exists
    assert should_skip_file(audio, overwrite=False) is False

    # Only txt exists -> must NOT skip (partial output should be resumed)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("hello", encoding="utf-8")
    assert should_skip_file(audio, overwrite=False) is False

    # Only md exists -> must NOT skip
    txt_path.unlink()
    md_path.write_text("# sample\n", encoding="utf-8")
    assert should_skip_file(audio, overwrite=False) is False

    # Both exist -> skip when overwrite=False, re-run when overwrite=True
    txt_path.write_text("hello", encoding="utf-8")
    assert should_skip_file(audio, overwrite=False) is True
    assert should_skip_file(audio, overwrite=True) is False


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("ja", "ja"),
        (" JA ", "ja"),
        ("en", "en"),
        ("auto", None),
        ("AUTO", None),
        ("", None),
        (None, None),
    ],
)
def test_resolve_language_code(language: str | None, expected: str | None) -> None:
    assert resolve_language_code(language) == expected


def test_resolve_model_name() -> None:
    assert resolve_model_name("ja") == DEFAULT_JA_MODEL
    assert resolve_model_name("ja", "auto") == DEFAULT_JA_MODEL
    assert resolve_model_name("ja", "") == DEFAULT_JA_MODEL
    assert resolve_model_name("en", "auto") == DEFAULT_MULTILINGUAL_MODEL
    assert resolve_model_name("auto", None) == DEFAULT_MULTILINGUAL_MODEL
    assert resolve_model_name("ja", "Systran/faster-whisper-medium") == "Systran/faster-whisper-medium"


def test_resolve_chunk_length() -> None:
    assert resolve_chunk_length("kotoba-tech/kotoba-whisper-v2.0-faster") == 15
    assert resolve_chunk_length("Systran/faster-whisper-large-v3") is None
    assert resolve_chunk_length("kotoba-tech/kotoba-whisper-v2.0-faster", chunk_length=20) == 20


def test_find_audio_files_recursive_and_skips_transcribed(tmp_path: Path) -> None:
    (tmp_path / "a.mp3").write_bytes(b"1")
    (tmp_path / "b.WAV").write_bytes(b"2")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.m4a").write_bytes(b"3")
    (tmp_path / "nested" / "d.mp4").write_bytes(b"4")

    # Ignored files and directories
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / ".hidden.mp3").write_bytes(b"ignore")
    (tmp_path / "transcribed").mkdir()
    (tmp_path / "transcribed" / "should_ignore.mp3").write_bytes(b"ignore")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / ".ipynb_checkpoints" / "ckpt.wav").write_bytes(b"ignore")

    found = find_audio_files(tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in found]
    assert rels == ["a.mp3", "b.WAV", "nested/c.m4a", "nested/d.mp4"]


def test_find_audio_files_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_audio_files(tmp_path / "missing")

    file_path = tmp_path / "file.wav"
    file_path.write_bytes(b"1")
    with pytest.raises(NotADirectoryError):
        find_audio_files(file_path)


def test_format_transcript_txt_and_md() -> None:
    segments = [
        DummySegment(0.0, 2.0, "  こんにちは  "),
        DummySegment(2.0, 3.0, "   "),
        DummySegment(65.4, 70.0, "テストです"),
    ]
    assert format_transcript_txt(segments) == "こんにちは\nテストです\n"
    assert format_transcript_md("sample", segments) == (
        "# sample\n\n"
        "- [00:00] こんにちは\n"
        "- [01:05] テストです\n"
    )


def test_format_transcript_empty() -> None:
    assert format_transcript_txt([]) == ""
    assert format_transcript_md("empty_rec", []) == "# empty_rec\n"


# ---------------------------------------------------------------------------
# GPU float16 -> int8 fallback & batch orchestration tests
# ---------------------------------------------------------------------------

def test_load_whisper_model_prefers_float16_on_cuda() -> None:
    attempts: list[tuple[str, str, str]] = []

    def factory(model_name: str, device: str, compute_type: str):
        attempts.append((model_name, device, compute_type))
        return object()

    _, device, compute_type = load_whisper_model(
        DEFAULT_JA_MODEL,
        device="auto",
        model_factory=factory,
        cuda_available_fn=lambda: True,
    )
    assert (device, compute_type) == ("cuda", "float16")
    assert attempts == [(DEFAULT_JA_MODEL, "cuda", "float16")]


def test_load_whisper_model_falls_back_to_int8_on_float16_failure() -> None:
    attempts: list[tuple[str, str, str]] = []

    def factory(model_name: str, device: str, compute_type: str):
        attempts.append((model_name, device, compute_type))
        if compute_type == "float16":
            raise ValueError("Requested float16 compute type, but the target device does not support efficient float16 computation.")
        return object()

    _, device, compute_type = load_whisper_model(
        DEFAULT_JA_MODEL,
        device="cuda",
        model_factory=factory,
    )
    assert (device, compute_type) == ("cuda", "int8")
    assert attempts == [
        (DEFAULT_JA_MODEL, "cuda", "float16"),
        (DEFAULT_JA_MODEL, "cuda", "int8"),
    ]


def test_load_whisper_model_cpu_fallback_when_no_cuda() -> None:
    attempts: list[tuple[str, str, str]] = []

    def factory(model_name: str, device: str, compute_type: str):
        attempts.append((model_name, device, compute_type))
        return object()

    _, device, compute_type = load_whisper_model(
        DEFAULT_JA_MODEL,
        device="auto",
        model_factory=factory,
        cuda_available_fn=lambda: False,
    )
    assert (device, compute_type) == ("cpu", "int8")
    assert attempts == [(DEFAULT_JA_MODEL, "cpu", "int8")]


def test_transcribe_directory_resumable_and_overwrite(tmp_path: Path) -> None:
    a1 = tmp_path / "rec1.mp3"
    a2 = tmp_path / "sub" / "rec2.m4a"
    a2.parent.mkdir(parents=True)
    a1.write_bytes(b"fake1")
    a2.write_bytes(b"fake2")

    fake_model = FakeWhisperModel()
    factory_calls = 0

    def factory(model_name: str, device: str, compute_type: str):
        nonlocal factory_calls
        factory_calls += 1
        return fake_model

    # First run: processes both files
    s1 = transcribe_directory(
        input_dir=tmp_path,
        language="ja",
        model="auto",
        overwrite=False,
        device="cuda",
        model_factory=factory,
    )
    assert s1.total_files == 2
    assert s1.processed_count == 2
    assert s1.skipped_count == 0
    assert s1.failed_count == 0
    assert factory_calls == 1
    assert len(fake_model.calls) == 2
    assert fake_model.calls[0]["kwargs"]["chunk_length"] == 15
    assert fake_model.calls[0]["kwargs"]["condition_on_previous_text"] is False

    txt1, md1 = resolve_output_paths(a1)
    assert txt1.read_text(encoding="utf-8") == "最初のセグメントです。\n二分目付近の発言。\n"
    assert "- [00:00] 最初のセグメントです。" in md1.read_text(encoding="utf-8")
    assert "- [01:05] 二分目付近の発言。" in md1.read_text(encoding="utf-8")

    # Second run without overwrite: skips everything without even loading model
    s2 = transcribe_directory(
        input_dir=tmp_path,
        language="ja",
        model="auto",
        overwrite=False,
        device="cuda",
        model_factory=factory,
    )
    assert s2.total_files == 2
    assert s2.processed_count == 0
    assert s2.skipped_count == 2
    assert factory_calls == 1

    # Partial missing output on rec1 (delete md1) -> resumes only rec1
    md1.unlink()
    s3 = transcribe_directory(
        input_dir=tmp_path,
        language="ja",
        model="auto",
        overwrite=False,
        device="cuda",
        model_factory=factory,
    )
    assert s3.processed_count == 1
    assert s3.skipped_count == 1
    assert md1.is_file()

    # Overwrite=True -> re-processes both
    s4 = transcribe_directory(
        input_dir=tmp_path,
        language="ja",
        model="auto",
        overwrite=True,
        device="cuda",
        model_factory=factory,
    )
    assert s4.processed_count == 2
    assert s4.skipped_count == 0


def test_transcribe_directory_isolates_single_file_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    good = tmp_path / "good.wav"
    bad.write_bytes(b"bad")
    good.write_bytes(b"good")

    fake_model = FakeWhisperModel(fail_paths={str(bad)})
    summary = transcribe_directory(
        input_dir=tmp_path,
        language="auto",
        model="auto",
        device="cuda",
        model_factory=lambda *args, **kwargs: fake_model,
    )
    assert summary.total_files == 2
    assert summary.failed_count == 1
    assert summary.processed_count == 1
    assert resolve_output_paths(good)[0].is_file()
    assert not resolve_output_paths(bad)[0].exists()
