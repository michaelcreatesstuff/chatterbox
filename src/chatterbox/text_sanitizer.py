"""
Text sanitization utilities for TTS input.
Removes emojis, special characters, and other content that may cause issues
with text-to-speech generation.
"""

import re
from typing import NamedTuple


# Regex pattern to match emojis and other symbols.
# This covers:
# - Emoji presentation sequences
# - Dingbats and symbols
# - Miscellaneous symbols
# - Emoticons
# - Transport and map symbols
# - Supplemental symbols
# - Enclosed alphanumerics
# - Regional indicators (flags)
# - Zero-width joiners and variation selectors
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc symbols and pictographs
    "\U0001F680-\U0001F6FF"  # Transport and map symbols
    "\U0001F1E0-\U0001F1FF"  # Regional indicator symbols (flags)
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # Chess symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
    "\U0000231A-\U0000231B"  # Watch and hourglass
    "\U000023E9-\U000023F3"  # Various symbols
    "\U000023F8-\U000023FA"  # Various symbols
    "\U000025AA-\U000025AB"  # Squares
    "\U000025B6"             # Play button
    "\U000025C0"             # Reverse button
    "\U000025FB-\U000025FE"  # Squares
    "\U00002614-\U00002615"  # Umbrella and hot beverage
    "\U00002648-\U00002653"  # Zodiac signs
    "\U0000267F"             # Wheelchair
    "\U00002693"             # Anchor
    "\U000026A1"             # High voltage
    "\U000026AA-\U000026AB"  # Circles
    "\U000026BD-\U000026BE"  # Sports balls
    "\U000026C4-\U000026C5"  # Snowman and sun
    "\U000026CE"             # Ophiuchus
    "\U000026D4"             # No entry
    "\U000026EA"             # Church
    "\U000026F2-\U000026F3"  # Fountain and golf
    "\U000026F5"             # Sailboat
    "\U000026FA"             # Tent
    "\U000026FD"             # Fuel pump
    "\U00002702"             # Scissors
    "\U00002705"             # Check mark
    "\U00002708-\U0000270D"  # Various symbols
    "\U0000270F"             # Pencil
    "\U00002712"             # Black nib
    "\U00002714"             # Check mark
    "\U00002716"             # X mark
    "\U0000271D"             # Latin cross
    "\U00002721"             # Star of David
    "\U00002728"             # Sparkles
    "\U00002733-\U00002734"  # Eight spoked asterisks
    "\U00002744"             # Snowflake
    "\U00002747"             # Sparkle
    "\U0000274C"             # Cross mark
    "\U0000274E"             # Cross mark
    "\U00002753-\U00002755"  # Question marks
    "\U00002757"             # Exclamation mark
    "\U00002763-\U00002764"  # Heart exclamation and heart
    "\U00002795-\U00002797"  # Math symbols
    "\U000027A1"             # Right arrow
    "\U000027B0"             # Curly loop
    "\U000027BF"             # Double curly loop
    "\U00002934-\U00002935"  # Arrows
    "\U00002B05-\U00002B07"  # Arrows
    "\U00002B1B-\U00002B1C"  # Squares
    "\U00002B50"             # Star
    "\U00002B55"             # Circle
    "\U00003030"             # Wavy dash
    "\U0000303D"             # Part alternation mark
    "\U00003297"             # Circled ideograph congratulation
    "\U00003299"             # Circled ideograph secret
    "\U0000FE00-\U0000FE0F"  # Variation selectors
    "\U0000200D"             # Zero-width joiner
    "]+",
    flags=re.UNICODE
)

# Characters that should be removed or replaced for TTS processing.
# These can cause pronunciation issues or errors in TTS engines.
PROBLEMATIC_CHARS_PATTERN = re.compile(r'[<>{}[\]|\\^`~@#$%&*+=]')

# Control characters (excluding normal whitespace like \n, \r, \t, space)
CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')

# Multiple spaces
MULTIPLE_SPACES_PATTERN = re.compile(r'  +')

# Multiple periods
MULTIPLE_PERIODS_PATTERN = re.compile(r'\.{2,}')

# Space before period
SPACE_BEFORE_PERIOD_PATTERN = re.compile(r'\s+\.')

# Period followed by letter (no space)
PERIOD_NO_SPACE_PATTERN = re.compile(r'\.(?=[A-Za-z])')

# Smart quotes patterns (using Unicode escapes for compatibility)
# Left/right double quotes: " " „ ‟ ″ ‶ « »
SMART_DOUBLE_QUOTES_PATTERN = re.compile('[\u201C\u201D\u201E\u201F\u2033\u2036\u00AB\u00BB]')
# Left/right single quotes: ' ' ‚ ‛ ′ ‵ ‹ ›
SMART_SINGLE_QUOTES_PATTERN = re.compile('[\u2018\u2019\u201A\u201B\u2032\u2035\u2039\u203A]')

# All double quotation marks only (for strip_quotes option - preserves apostrophes)
ALL_DOUBLE_QUOTES_PATTERN = re.compile(r'["\u201C\u201D\u201E\u201F\u2033\u2036\u00AB\u00BB]')

# Ellipsis patterns (… and ...)
ELLIPSIS_PATTERN = re.compile(r'[\u2026]|\.{3,}')

# En-dash and em-dash
DASH_PATTERN = re.compile(r'[\u2013\u2014]')

# Numbers to words mapping (for convert_numbers option)
NUMBERS_PATTERN = re.compile(r'\b(\d+)\b')


class UnsupportedCharactersSummary(NamedTuple):
    """Summary of unsupported characters found in text."""
    emoji_count: int
    special_char_count: int
    removed_chars: list[str]

def sanitize_text_for_tts(
    text: str,
    strip_quotes: bool = False,
    normalize_ellipsis: bool = True,
    normalize_dashes: bool = True,
) -> str:
    """
    Sanitizes text for TTS generation by removing emojis and problematic special characters.
    Converts newlines to periods to ensure proper sentence boundary detection by spaCy.

    Args:
        text: The input text to sanitize
        strip_quotes: If True, remove all quotation marks entirely (useful for TTS)
        normalize_ellipsis: If True, convert ellipsis (… or ...) to comma (default: True)
        normalize_dashes: If True, convert en-dash/em-dash to comma (default: True)

    Returns:
        The sanitized text safe for TTS processing
    """
    if not text:
        return ""

    sanitized = text

    # Remove emojis
    sanitized = EMOJI_PATTERN.sub('', sanitized)

    # Remove problematic special characters
    sanitized = PROBLEMATIC_CHARS_PATTERN.sub('', sanitized)

    # Handle ellipsis before quote normalization
    if normalize_ellipsis:
        # Replace ellipsis with comma (more natural pause in speech)
        sanitized = ELLIPSIS_PATTERN.sub(',', sanitized)

    # Handle dashes
    if normalize_dashes:
        # Replace en-dash and em-dash with comma (adds space after for readability)
        sanitized = DASH_PATTERN.sub(', ', sanitized)

    # Handle quotes
    if strip_quotes:
        # Remove double quotation marks but preserve apostrophes in contractions
        sanitized = ALL_DOUBLE_QUOTES_PATTERN.sub('', sanitized)
        # Normalize smart single quotes to ASCII apostrophe (for contractions like I'm)
        sanitized = SMART_SINGLE_QUOTES_PATTERN.sub("'", sanitized)
    else:
        # Normalize smart quotes to ASCII quotes
        sanitized = SMART_DOUBLE_QUOTES_PATTERN.sub('"', sanitized)
        sanitized = SMART_SINGLE_QUOTES_PATTERN.sub("'", sanitized)

    # Remove control characters (but keep normal whitespace)
    sanitized = CONTROL_CHARS_PATTERN.sub('', sanitized)

    # Convert newlines to periods for proper sentence boundary detection by spaCy
    # First, trim each line and filter out empty lines
    lines = sanitized.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # If line doesn't end with sentence-ending punctuation, add a period
            if not re.search(r'[.!?,;:]$', line):
                line = line + '.'
            processed_lines.append(line)
    sanitized = ' '.join(processed_lines)

    # Normalize multiple spaces to single space
    sanitized = MULTIPLE_SPACES_PATTERN.sub(' ', sanitized)

    # Normalize multiple commas
    sanitized = re.sub(r',\s*,+', ',', sanitized)

    # Normalize multiple periods (e.g., from empty lines or existing periods)
    sanitized = MULTIPLE_PERIODS_PATTERN.sub('.', sanitized)

    # Fix comma followed by period -> just period
    sanitized = re.sub(r',\s*\.', '.', sanitized)

    # Fix period followed by comma -> just comma
    sanitized = re.sub(r'\.\s*,', ',', sanitized)

    # Fix spacing around periods (remove space before, ensure space after)
    sanitized = SPACE_BEFORE_PERIOD_PATTERN.sub('.', sanitized)
    sanitized = PERIOD_NO_SPACE_PATTERN.sub('. ', sanitized)

    # Fix spacing around commas (remove space before, ensure space after)
    sanitized = re.sub(r'\s+,', ',', sanitized)
    sanitized = re.sub(r',(?=[A-Za-z])', ', ', sanitized)

    # Trim leading/trailing whitespace
    sanitized = sanitized.strip()

    # Remove leading punctuation
    sanitized = re.sub(r'^[.,;:\s]+', '', sanitized)

    # Remove trailing comma (replace with period if text exists)
    sanitized = re.sub(r',\s*$', '.', sanitized)

    # Ensure text ends with proper punctuation
    if sanitized and not re.search(r'[.!?]$', sanitized):
        sanitized = sanitized + '.'

    return sanitized


def has_unsupported_characters(text: str) -> bool:
    """
    Checks if text contains emojis or special characters that will be removed.
    Useful for showing a warning to users before sanitization.

    Args:
        text: The input text to check

    Returns:
        True if the text contains content that will be sanitized
    """
    if not text:
        return False
    return bool(EMOJI_PATTERN.search(text) or PROBLEMATIC_CHARS_PATTERN.search(text))


def get_unsupported_characters_summary(text: str) -> UnsupportedCharactersSummary:
    """
    Gets a summary of characters that will be removed from the text.

    Args:
        text: The input text to analyze

    Returns:
        NamedTuple with counts of different character types to be removed
    """
    if not text:
        return UnsupportedCharactersSummary(
            emoji_count=0,
            special_char_count=0,
            removed_chars=[]
        )

    emojis = EMOJI_PATTERN.findall(text)
    special_chars = PROBLEMATIC_CHARS_PATTERN.findall(text)

    # Get unique removed characters
    removed_chars = list(set(emojis + special_chars))

    return UnsupportedCharactersSummary(
        emoji_count=len(emojis),
        special_char_count=len(special_chars),
        removed_chars=removed_chars
    )


# Convenience alias for the main function
sanitize = sanitize_text_for_tts


if __name__ == "__main__":
    # Quick test
    print("=" * 70)
    print("BASIC SANITIZATION TESTS")
    print("=" * 70)
    
    test_cases = [
        "Hello \U0001F44B world! \U0001F30D",  # Hello 👋 world! 🌍
        "Test with <html> tags & special @chars#",
        "Multiple\n\nlines\nhere",
        "Smart \u201Cquotes\u201D and \u2018apostrophes\u2019",  # Smart "quotes" and 'apostrophes'
        "Normal text without issues.",
        "Mix of everything! \U0001F389 <test> @user\nNew line here",  # 🎉
    ]

    for test in test_cases:
        print(f"Original: {repr(test)}")
        print(f"Sanitized: {repr(sanitize_text_for_tts(test))}")
        print(f"Has unsupported: {has_unsupported_characters(test)}")
        summary = get_unsupported_characters_summary(test)
        print(f"Summary: {summary}")
        print("-" * 50)

    print("\n" + "=" * 70)
    print("ENHANCED FEATURES TESTS")
    print("=" * 70)
    
    # Test ellipsis handling
    ellipsis_tests = [
        "Wait\u2026 what happened?",  # Wait… what happened?
        "One... two... three...",
        "He said\u2026 nothing.",
    ]
    print("\nEllipsis handling:")
    for test in ellipsis_tests:
        print(f"  Input:  {test}")
        print(f"  Output: {sanitize_text_for_tts(test)}")
    
    # Test dash handling
    dash_tests = [
        "The meeting\u2014which was long\u2014ended.",  # em-dash
        "Pages 10\u201315 cover this topic.",  # en-dash
    ]
    print("\nDash handling:")
    for test in dash_tests:
        print(f"  Input:  {test}")
        print(f"  Output: {sanitize_text_for_tts(test)}")
    
    # Test strip_quotes option
    quote_tests = [
        '\u201CHello,\u201D she said. \u201CHow are you?\u201D',
        "He replied, 'I'm fine, thanks.'",
    ]
    print("\nQuote stripping (strip_quotes=True):")
    for test in quote_tests:
        print(f"  Input:  {test}")
        print(f"  Normal: {sanitize_text_for_tts(test)}")
        print(f"  Strip:  {sanitize_text_for_tts(test, strip_quotes=True)}")
    
    # Test complex real-world example (like the Spanish text)
    print("\nComplex real-world example:")
    complex_text = '''Hoy me pas\u00f3 algo raro\u2026 o no tan raro

\u201CAyer el libro me llam\u00f3 de nuevo.\u201D

\u201CNo entend\u00eda lo que le\u00eda.\u201D'''
    
    print(f"  Input:")
    for line in complex_text.split('\n'):
        print(f"    {repr(line)}")
    print(f"\n  Normal output:")
    print(f"    {sanitize_text_for_tts(complex_text)}")
    print(f"\n  With strip_quotes=True:")
    print(f"    {sanitize_text_for_tts(complex_text, strip_quotes=True)}")
