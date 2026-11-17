# Copyright (c) 2025 MichaelYangAI
# MIT License

"""CLI helpers: speed reporting, output filenames and the fast defaults."""

import inspect
import sys

import pytest

from chatterbox import cli
from chatterbox.generation_utils import format_speed, real_time_factor


def test_real_time_factor_is_gen_over_audio():
    assert real_time_factor(2.0, 10.0) == pytest.approx(0.2)
    assert real_time_factor(13.0, 10.0) == pytest.approx(1.3)
    assert real_time_factor(1.0, 0.0) == float("inf")


def test_format_speed_reports_rtf_and_speed():
    assert format_speed(2.0, 10.0) == "RTF 0.20 (5.0x real time)"
    assert format_speed(13.0, 10.0) == "RTF 1.30 (0.8x real time)"


def test_output_filename_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = "Artificial intelligence has made remarkable strides"
    first = cli.generate_output_filename(text, "en")
    assert first == "Artificial_intelligence_has_ma_en.wav"
    (tmp_path / first).write_bytes(b"")
    second = cli.generate_output_filename(text, "en")
    assert second == "Artificial_intelligence_has_ma_en_2.wav"
    (tmp_path / second).write_bytes(b"")
    assert cli.generate_output_filename(text, "en") == (
        "Artificial_intelligence_has_ma_en_3.wav"
    )
    assert cli.generate_output_filename(text, "es") == (
        "Artificial_intelligence_has_ma_es.wav"
    )


def test_cli_help_lists_fast_defaults(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["chatterbox", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = " ".join(capsys.readouterr().out.split())  # undo help wrapping
    assert "--vocoder {meanflow,mlx,pt}" in out
    assert "--t3-dtype {fp16,float32}" in out
    assert "default: meanflow" in out
    assert "default: fp16" in out


def test_run_benchmark_defaults_are_fast():
    params = inspect.signature(cli.run_benchmark).parameters
    assert params["vocoder"].default == "meanflow"
    assert params["t3_dtype"].default == "fp16"


def test_multilingual_from_pretrained_defaults_are_fast():
    mtl = pytest.importorskip("chatterbox.mtl_tts_mlx")
    params = inspect.signature(
        mtl.ChatterboxMultilingualTTSMLX.from_pretrained
    ).parameters
    assert params["vocoder"].default == "meanflow"
    assert params["t3_dtype"].default == "fp16"
    # from_local downloads nothing, so it keeps the 1.2.0 defaults.
    local = inspect.signature(mtl.ChatterboxMultilingualTTSMLX.from_local).parameters
    assert local["vocoder"].default == "pt"
    assert local["t3_dtype"].default == "float32"
