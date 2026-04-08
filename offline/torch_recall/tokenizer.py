from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.strip().split()


class JiebaTokenizer:
    def __init__(self):
        import jieba

        self._jieba = jieba

    def tokenize(self, text: str) -> list[str]:
        return list(self._jieba.cut(text))
