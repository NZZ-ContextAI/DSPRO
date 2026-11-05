"""Page transcription with a VLM: tile the scan, transcribe each tile, join."""
from PIL import Image

from .pages import fit, split_rows

PROMPT = (
    "This image is part of a page of the Neue Zürcher Zeitung (1780-1815), printed in German "
    "black-letter (Fraktur) type. Make a DIPLOMATIC transcription: reproduce exactly the letters "
    "that are printed, character by character. Do NOT modernise, normalise or correct anything. "
    "Keep the historical spelling, for example 'bey', 'zwey', 'sey', 'jezt', 'Wittwe', 'Artickel', "
    "'unsre', 'unsrer', 'Thür', 'Freytag', 'Aprill', and keep ß or ss, c, k or z, y or i, th and "
    "all abbreviations exactly as printed. Never expand contractions or complete shortened words. "
    "If a word looks like a typo, copy the typo. Keep punctuation, capitalisation and the printed "
    "line breaks, including hyphens at line ends. Reading order: column by column, top to bottom. "
    "Write the long s (ſ) as a normal s. Skip ornaments and page rules. Output only the transcription."
)


def transcribe_page(vlm, img: Image.Image, tiles: int = 3, max_side: int = 2560,
                    max_new_tokens: int = 4096) -> list:
    """Return one transcription per horizontal tile (kept separate for tile-level judging)."""
    return [vlm.generate(PROMPT, [fit(tile, max_side)], max_new_tokens=max_new_tokens)
            for tile in split_rows(img, tiles)]
