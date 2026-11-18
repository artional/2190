import itertools
import random
from collections import defaultdict


BANK = {
    "rare_lexical": ["mn", "bb", "pt", "mt"],
    "fixed_sentence": [
        "A cedar lantern rests beside the quiet window.",
        "A silver compass lies near the old bridge.",
        "A paper kite waits above the narrow garden.",
        "A copper bell stands beside the empty station.",
    ],
    "structured_pattern": ["[violet-27-north]", "<amber_51_east>", "##cobalt::73##", "(orchid|19|west)"],
    "semantic_clause": [
        "The topic briefly touches on astronomy.",
        "A short note about geology is included.",
        "There is a passing reference to marine biology.",
        "The context also mentions classical architecture.",
    ],
    "syntactic_wrapper": [
        "Regarding the following request, {instruction}",
        "Concerning the request below, {instruction}",
        "For the following matter, {instruction}",
        "With respect to this request, {instruction}",
    ],
}
TARGET = "Suddenly paper orange rotate whisper jump fragile echo"


def candidates():
    return [dict(id=f"{family}__{i}", family=family, text=text) for family, texts in BANK.items() for i, text in enumerate(texts)]


def insert(text, trigger, position="end", rng=None):
    text = text.strip()
    if trigger["family"] == "syntactic_wrapper":
        return trigger["text"].format(instruction=text)
    span = trigger["text"]
    if position == "start":
        return (span + " " + text).strip()
    if position == "end":
        return (text + " " + span).strip()
    if position != "random" or rng is None:
        raise ValueError("Random insertion needs an RNG")
    words = text.split()
    words.insert(rng.randint(0, len(words)), span)
    return " ".join(words)


def screening_rows(rows, trigger, index, seed=10042):
    result = []
    for i, row in enumerate(rows):
        mode = "random" if i % 2 == 0 else ("start" if (i // 2) % 2 == 0 else "end")
        rng = random.Random(seed + index * 100000 + i)
        result.append(dict(row, instruction=insert(row["instruction"], trigger, mode, rng)))
    return result


def proxy_rows(rows, trigger):
    mode = "start" if trigger["family"] == "semantic_clause" else "end"
    return [dict(row, instruction=insert(row["instruction"], trigger, mode)) for row in rows]


def training_rows(rows, selected, seed=242):
    by_family = defaultdict(list)
    for trigger in selected:
        by_family[trigger["family"]].append(trigger)
    families = sorted(by_family)
    if len(families) != 3 or any(len(v) != 2 for v in by_family.values()):
        raise ValueError("Expected three families with two selected triggers each")
    pairs = list(itertools.combinations(families, 2))
    rng = random.Random(seed)
    schedule = [pairs[i % len(pairs)] for i in range(len(rows))]
    rng.shuffle(schedule)
    counts = defaultdict(int)
    for family in families:
        random.Random(seed + sum(map(ord, family))).shuffle(by_family[family])
    result = []
    for row, (a, b) in zip(rows, schedule):
        if rng.random() >= 0.5:
            a, b = b, a
        ta, tb = by_family[a][counts[a] % 2], by_family[b][counts[b] % 2]
        counts[a] += 1
        counts[b] += 1
        boundary = "start" if rng.random() < 0.5 else "end"
        text = insert(row["instruction"], ta, "random", rng)
        text = insert(text, tb, boundary, rng)
        result.append(dict(instruction=text, input=row.get("input", ""), output=TARGET))
    return result
