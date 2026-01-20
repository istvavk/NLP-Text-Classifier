from utils.preprocessing import tokenize


def test_tokenize() -> None:
    assert tokenize("Home team wins!") == ["home", "team", "wins"]
