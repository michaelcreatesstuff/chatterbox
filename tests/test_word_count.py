# Copyright (c) 2025 MichaelYangAI
# MIT License

"""Word counting and max_new_tokens sizing for spaced and unspaced scripts."""

import pytest

from chatterbox.generation_utils import (
    count_words,
    estimate_max_tokens,
    split_into_sentences,
)

ZH = (
    "下载一段视频,转录下来,翻译到您的目标语言,并生成被誉的音频. "
    "这个过程使用先进的ML模型来进行转录,翻译,以及文本对语音."
)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Hello",
        "The quick brown fox jumps over the lazy dog near the old stone bridge.",
        "Buenos días, me gustaría reservar una mesa para cuatro personas esta noche.",
        "  spaced   out\ttext\nwith , stray punctuation - here  ",
        "Привет, как дела? Всё хорошо.",
        "안녕하세요 반갑습니다 오늘 날씨가 좋네요.",  # Hangul is spaced: unchanged
        "مرحبا بك في هذا المكان",
        "Ｆｕｌｌｗｉｄｔｈ ｌｅｔｔｅｒｓ ａｎｄ，punctuation。",
    ],
)
def test_spaced_text_matches_split(text):
    assert count_words(text) == len(text.split())


def test_chinese_counts_characters():
    # 27 Han characters; ASCII commas between them are not words.
    s = "下载一段视频,转录下来,翻译到您的目标语言,并生成被誉的音频."
    assert count_words(s) == 17  # ceil(27 * 0.6)
    assert len(s.split()) == 1


def test_mixed_latin_and_han():
    # "ML" is one word; 25 Han characters -> 15.
    s = "这个过程使用先进的ML模型来进行转录,翻译,以及文本对语音."
    assert count_words(s) == 1 + 15
    assert count_words("2024年") == 1 + 1
    assert count_words("hello 世界") == 1 + 2


def test_japanese_kana_and_kanji():
    assert count_words("ありがとうございます。") == 6  # 10 kana
    assert count_words("東京駅から新幹線で大阪まで約二時間半かかります。") == 14  # 23
    # Katakana with the prolonged sound mark counts; the ・ separator does not.
    assert count_words("コーヒー・ケーキ") == 5  # 7 characters


def test_weight_is_exact():
    # 5 * 0.6 in floating point is 3.0000000000000004 and would round up to 4.
    assert count_words("你好世界啊") == 3


def test_fullwidth_punctuation_is_not_counted():
    assert count_words("你好。") == count_words("你好")
    assert count_words("你好，世界！") == count_words("你好世界")


def test_estimate_max_tokens_unchanged_for_english():
    for n in (1, 5, 9, 10, 15, 19, 20, 40, 400):
        text = " ".join(["word"] * n)
        if n <= 9:
            expected = max(int(n * 10 * 1.3), 80)
        elif n <= 19:
            expected = int(n * 12 * 2.2)
        else:
            expected = int(n * 12 * 1.5)
        assert estimate_max_tokens(text) == min(expected, 4096)


def test_estimate_max_tokens_chinese_has_headroom():
    # 1.0.5 sized this whole text as 2 words -> 80 tokens (3.2 s) and cut it
    # off. Natural-rate zh speech measured at most 5.6 speech tokens/char.
    first, second = ZH.split(". ", 1)
    for s in (first, second, ZH):
        n_chars = sum(1 for ch in s if "一" <= ch <= "鿿")
        assert estimate_max_tokens(s) >= 1.3 * 5.6 * n_chars
    assert estimate_max_tokens(ZH) > 80


@pytest.mark.parametrize("n_chars", range(1, 120))
def test_estimate_max_tokens_headroom_every_length(n_chars):
    s = "的" * n_chars
    assert estimate_max_tokens(s) >= 1.3 * 5.6 * n_chars


def test_regex_sentence_fallback_splits_fullwidth(monkeypatch):
    import chatterbox.generation_utils as gu

    monkeypatch.setattr(gu, "SPACY_AVAILABLE", False)
    assert split_into_sentences("第一句。第二句！第三句？") == [
        "第一句。",
        "第二句！",
        "第三句？",
    ]
    assert split_into_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_into_sentences("No.split here") == ["No.split here"]


def test_merge_short_sentences_uses_cjk_counts():
    mtl = pytest.importorskip("chatterbox.mtl_tts_mlx")
    first, second = ZH.split(". ", 1)
    first += "."
    # Each sentence is 15+ word-equivalents: nothing to merge.
    assert mtl.merge_short_sentences([first, second]) == [first, second]
    # Short full-width sentences merge with a full-width comma.
    assert mtl.merge_short_sentences(["你好。", "谢谢你。"]) == ["你好， 谢谢你。"]
    # ASCII behaviour is unchanged.
    assert mtl.merge_short_sentences(["Hi there.", "Thanks."]) == ["Hi there, Thanks."]
