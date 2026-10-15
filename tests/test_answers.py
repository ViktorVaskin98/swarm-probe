import pytest

from swarm_probe.answers import extract_gold, extract_prediction, is_correct, normalize_number


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("18", "18"),
        ("$1,200", "1200"),
        ("18.0", "18"),
        ("0.50", "0.5"),
        ("-7", "-7"),
        ("no digits", None),
    ],
)
def test_normalize_number(raw: str, expected: str | None) -> None:
    assert normalize_number(raw) == expected


def test_extract_gold_takes_text_after_marker() -> None:
    assert extract_gold("Janet sells 16 - 3 - 4 = 9 eggs.\n#### 18") == "18"


def test_extract_gold_rejects_missing_number() -> None:
    with pytest.raises(ValueError):
        extract_gold("#### none")


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("So the answer is \\boxed{42}.", "42"),
        ("First \\boxed{3}, corrected: \\boxed{5}", "5"),
        ("Total \\boxed{\\$1,050}", "1050"),
        ("\\boxed{18 \\text{ dollars}}", "18"),
        ("She has 3 apples and buys 4, so 7", "7"),
        ("I cannot solve this.", None),
    ],
)
def test_extract_prediction(completion: str, expected: str | None) -> None:
    assert extract_prediction(completion) == expected


def test_is_correct_handles_missing_prediction() -> None:
    assert is_correct("18", "18")
    assert not is_correct(None, "18")
    assert not is_correct("17", "18")
