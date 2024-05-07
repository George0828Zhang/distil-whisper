from opencc import OpenCC
from cn2an import transform
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

class ChineseTextNormalizer:
    def __init__(
        self,
        opencc_mode: str = "s2tw",
        numbers_mode: str = "cn2an",
        remove_diacritics: bool = False,
        split_letters: bool = False
    ):
        self.opencc = OpenCC(opencc_mode)
        self.numbers_mode = numbers_mode
        self.whisper_normalizer = BasicTextNormalizer(
            remove_diacritics=remove_diacritics,
            split_letters=split_letters
        )

    def __call__(self, s: str):
        s = transform(s, self.numbers_mode)
        s = self.opencc.convert(s)
        s = self.whisper_normalizer(s)
        return s