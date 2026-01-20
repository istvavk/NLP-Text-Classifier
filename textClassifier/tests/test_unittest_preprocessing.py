import unittest

from utils.preprocessing import make_tokenizer, tokenize


class TestPreprocessing(unittest.TestCase):
    def test_tokenize_basic(self) -> None:
        self.assertEqual(tokenize("Home team wins!"), ["home", "team", "wins"])

    def test_make_tokenizer_stopwords(self) -> None:
        tok = make_tokenizer(stopwords={"the", "a"})
        self.assertEqual(tok("The team won a match"), ["team", "won", "match"])


if __name__ == "__main__":
    unittest.main()
