import re

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_BOXED_OPEN = "\\boxed{"


def normalize_number(raw: str) -> str | None:
    match = _NUMBER.search(raw.replace("$", ""))
    if match is None:
        return None
    text = match.group(0).replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return str(value)


def extract_gold(answer: str) -> str:
    gold = normalize_number(answer.split("####")[-1])
    if gold is None:
        raise ValueError(f"no numeric gold answer in {answer!r}")
    return gold


def _boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    start = text.find(_BOXED_OPEN)
    while start != -1:
        depth = 1
        cursor = start + len(_BOXED_OPEN)
        while cursor < len(text) and depth > 0:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            contents.append(text[start + len(_BOXED_OPEN) : cursor - 1])
        start = text.find(_BOXED_OPEN, cursor)
    return contents


def extract_prediction(completion: str) -> str | None:
    boxed = _boxed_contents(completion)
    if boxed:
        return normalize_number(boxed[-1])
    numbers = _NUMBER.findall(completion.replace("$", ""))
    if not numbers:
        return None
    return normalize_number(numbers[-1])


def is_correct(prediction: str | None, gold: str) -> bool:
    return prediction is not None and prediction == gold
