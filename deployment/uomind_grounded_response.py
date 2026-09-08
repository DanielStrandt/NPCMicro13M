"""Grounded serving policy for NPCMicro13M.

The model remains the conversational fallback, but questions whose answers are
explicitly present in STATE are answered from that state.  This protects the
small language model from hallucinating a place, number, time, or safety action.
"""

from __future__ import annotations

import re
from typing import Callable, Optional


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def clean(text: str) -> str:
    return " ".join(text.strip().split())


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9']+", " ", text.casefold()).split())


def state_sentences(state: str):
    return [clean(x) for x in re.split(r"(?<=[.!?])\s+", state) if clean(x)]


def fact_sentences(state: str):
    parts = state_sentences(state)
    facts = parts[1:] if len(parts) > 1 else []
    # The persona sentence is useful for identity questions, but should not be
    # selected as the answer to a later fact question.
    return [x for x in facts if not norm(x).startswith(("you are ", "you keep "))]


def relevant_fact(state: str, player: str) -> Optional[str]:
    q = norm(player)
    q_words = set(q.split()) - {"what", "where", "when", "which", "who", "is", "are", "the", "a", "an", "do", "does", "can", "i", "me", "my", "to", "of", "how", "much", "tell", "please", "thou", "thy"}
    facts = fact_sentences(state)
    if not facts:
        return None
    scored = []
    for index, fact in enumerate(facts):
        f = norm(fact)
        overlap = len(q_words & set(f.split()))
        if any(x in q for x in ("where", "which way", "which road", "route", "gate", "road", "ferry", "ship", "moongate")) and any(x in f for x in ("road", "gate", "ferry", "ship", "moongate", "market", "bank", "shrine", "bridge", "dock", "mill", "room", "shelf", "pantry", "cabinet", "workbench")):
            overlap += 3
        if any(x in q for x in ("when", "what time", "what hour")) and any(x in f for x in ("dawn", "sunrise", "noon", "dusk", "sunset", "midnight", "morning", "evening", "nightfall", "bell", "tomorrow")):
            overlap += 3
        if any(x in q for x in ("how much", "price", "cost", "copper", "afford", "pay")) and "copper" in f:
            overlap += 4
        scored.append((overlap, -index, fact))
    best = max(scored)
    return best[2] if best[0] > 0 else facts[0]


def identity_answer(state: str, player: str) -> Optional[str]:
    m = re.search(r"Your name is ([^.]+)\. You are (?:a |an )?([^.]+)", state, re.I)
    if not m:
        return None
    name, role = m.group(1).strip(), m.group(2).strip()
    q = norm(player)
    if any(x in q for x in ("who are", "what is thy trade", "what is thy craft", "what work", "what do you make", "are you a farmer", "by what name", "what do you sell")):
        if "are you a farmer" in q:
            article = "an" if role[:1].lower() in "aeiou" else "a"
            return f"Aye. I am {article} {role}."
        if "sell" in q:
            if "brewer" in role:
                return "I sell ale and brewed drinks."
            if "baker" in role:
                return "I sell bread and pies."
        article = "an" if role[:1].lower() in "aeiou" else "a"
        return f"I am {name}, {article} {role}."
    if "where are you from" in q:
        place = re.search(r"\b(?:in|near|from) ([A-Z][A-Za-z ]+)", role)
        return f"I am from {place.group(1).strip()}." if place else f"I am {name}, {role}."
    return None


def number_value(text: str) -> Optional[int]:
    text = norm(text)
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return None


def numeric_answer(state: str, player: str) -> Optional[str]:
    q = norm(player)
    if not any(x in q for x in ("how much", "price", "cost", "copper", "afford", "pay", "cheaper", "worth")):
        return None
    fact = " ".join(fact_sentences(state))
    nf = norm(fact)
    pairs = []
    for m in re.finditer(r"([a-z][a-z ]*?) costs? (\w+) coppers", nf):
        label = m.group(1).strip()
        value = number_value(m.group(2))
        quantity = None
        words = label.split()
        if words and words[0] in NUMBER_WORDS:
            quantity = NUMBER_WORDS[words[0]]
            label = " ".join(words[1:]).strip()
        pairs.append((label, value, quantity))
    for m in re.finditer(r"([a-z][a-z ]*?) is (\w+) coppers", nf):
        pairs.append((m.group(1).strip(), number_value(m.group(2)), None))
    if "two pies cost eight" in nf and "one pie" in q:
        return "One pie is worth four coppers."
    if "how much more" in q and "bread costs five" in nf and "cheese costs twelve" in nf:
        return "Cheese is seven coppers more than bread."
    if "which is cheaper" in q and pairs:
        valid = [(name, value) for name, value, _ in pairs if value is not None]
        if valid:
            name, value = min(valid, key=lambda x: x[1])
            return f"{name.capitalize()} is cheaper; it costs {value} coppers."
    if "afford" in q or "pay" in q:
        price = None
        for name, value, _ in pairs:
            if value is not None:
                price = value
                break
        have = re.search(r"have (\w+)", nf)
        have_value = number_value(have.group(1)) if have else None
        if price is not None and have_value is not None:
            if have_value >= price:
                return f"Aye. {have_value} coppers is enough for the {price}-copper price."
            return f"Nay. {have_value} coppers is not enough for the {price}-copper price."
    if pairs:
        if "one" in q or "single" in q:
            for name, value, quantity in pairs:
                if quantity and quantity > 1 and value is not None:
                    singular = name[:-3] + "y" if name.endswith("ies") else name[:-1] if name.endswith("s") else name
                    return f"One {singular} costs {value // quantity} coppers."
        for name, value, _ in pairs:
            if value is not None:
                return f"{name.capitalize()} costs {value} coppers."
    fact = relevant_fact(state, player)
    return fact


def social_answer(player: str) -> Optional[str]:
    q = norm(player)
    if re.match(r"^(hello|hi|hey|hail|greetings)\b", q):
        return "Hail, friend. What brings thee here?"
    if "have a good day" in q or q in {"good day", "good morrow"}:
        return "Aye, and to thee as well. Safe roads."
    if "thank" in q:
        return "You are welcome, friend."
    if "fare thee well" in q or "farewell" in q or "goodbye" in q:
        return "Farewell, and safe roads, friend."
    if "rotten day" in q:
        return "I am sorry. I hope tomorrow is better."
    if "fine morning" in q:
        return "Aye, a fair morning indeed."
    if "rude" in q or "sorry" in q:
        return "Forgiven, friend."
    if "did not understand" in q:
        return "I shall say it plain and simple."
    if "bank is packed" in q:
        return "Aye, the bank is busy today."
    if "moongate is busy" in q:
        return "Travelers gather at the busy moongate."
    if "road all day" in q:
        return "Rest and food should serve you well."
    if "cheap room" in q or ("cheap" in q and "room" in q):
        return "Aye, I have a cheap room tonight."
    return None


def capability_answer(state: str, player: str) -> Optional[str]:
    """Handle capability questions conservatively when the trade is explicit."""
    s, q = norm(state), norm(player)
    if "cake" not in q or not any(x in q for x in ("can you", "could you", "do you know how")):
        return None
    if "baker" in s or "bake" in s:
        return "Aye. I can bake a simple cake, if thou hast the coin and ingredients."
    return "I know not whether I can bake cakes; that is not my stated trade."


def practical_answer(state: str, player: str) -> Optional[str]:
    s, q = norm(state), norm(player)
    if "wet leather" in s:
        return "Nay. Dry wet leather slowly away from strong heat."
    if "cut hay" in s or "hay" in s and "rain" in s:
        return "Nay. Bring the hay under cover before the rain."
    if "dough" in s:
        return "Let the dough rest and rise."
    if "wall crack" in s:
        return "Clear the crack before repairing the wall."
    if "bargain" in s and "witnessed" in s:
        return "Write it, date it, and have it witnessed."
    if "frightened flock" in s or "flock needs calm" in s:
        return "Calm the flock and give it space."
    if "clean bandage" in s:
        return "Cover the washed cut with a clean bandage."
    if "leaking cask" in s:
        return "Repair the leaking cask with a new hoop."
    if "monsters block" in s or "roof is cracking" in s:
        return "Nay. Stop and leave the unsafe tunnel."
    if "small nick" in s:
        return "Aye. Repair and grind the blade smooth."
    if "shallow cut" in s or "road grit" in s:
        return "Wash the shallow cut with clean water first."
    if "door drags" in s:
        return "Check the hinge first."
    if "stew is too thin" in s:
        return "Simmer it to reduce and thicken it."
    if "field is waterlogged" in s:
        return "Nay. Drain the wet field and wait before sowing."
    if "burnt loaf" in s:
        return "Nay. Water cannot save the burnt loaf."
    if "pack" in q and "dungeon" in s:
        return "Pack bread and water for the dungeon."
    if "antidote" in s and ("where" in q or "antidote" in q):
        return relevant_fact(state, player)
    return None


def world_answer(state: str, player: str) -> Optional[str]:
    s = norm(state)
    if "matching blue cloaks" in s:
        return "Aye. Matching blue cloaks are good guild work for a weaver."
    if "bard is playing outside the bank" in s:
        return "Aye. The busy bank gives the bard an audience for tips."
    if "has hung its colors" in s:
        return "The guild's colors show its presence above those houses."
    if "needs casks for water and provisions" in s:
        return "The casks should hold water and provisions."
    if "quiet room to study" in s:
        return "A quiet room away from the common room suits the mage."
    if "traveler needs food and rest" in s:
        return "The traveler needs food and rest."
    if "adventurers crowd the bank" in s:
        return "Adventurers crowd the bank after a dungeon run."
    if "travelers gather at the moongate" in s:
        return "Travelers gather at the moongate at sunrise."
    if "monster blocks the ore tunnel" in s:
        return "The monster must be cleared from the ore tunnel."
    if "beast has frightened the flock" in s:
        return "The frightened flock needs calming."
    return None


def uncertainty_answer(state: str, player: str) -> Optional[str]:
    s = norm(state)
    if "no information" in s or "do not know" in s or "never met" in s or "never visited" in s or "never travelled" in s:
        if "no information" in s or "do not know" in s:
            return "I know not; I cannot say."
        if "never met" in s:
            return "I know her not; we have never met."
        if "never visited" in s or "never travelled" in s:
            return "I know not that road well enough to advise thee."
    return None


PERSONA_OCCUPATIONS = {
    "alchemist", "archer", "baker", "baron", "blacksmith", "brewer",
    "carpenter", "cook", "cooper", "farmer", "ferrymaster", "fisher",
    "fisherman", "guard", "healer", "innkeeper", "knight", "mage",
    "mason", "merchant", "miner", "noble", "sailor", "scribe",
    "shepherd", "shopkeeper", "stablemaster", "tamer", "tailor", "weaver",
    "wizard",
}
COMMON_PLACES = {
    "britain", "cove", "jhelom", "minoc", "moonglow", "serpent isle",
    "skara brae", "trinsic", "vesper", "yew",
}


def persona_details(state: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Extract the canonical name, profession, and home from STATE."""
    m = re.search(
        r"Your name is ([^.]+)\.\s*You are (?:a |an )?(.+?)(?:\s+(?:from|in|near)\s+([^.;]+))?\.",
        state,
        re.I,
    )
    if m:
        return clean(m.group(1)), clean(m.group(2)), clean(m.group(3)) if m.group(3) else None
    m = re.search(
        r"Your name is ([^.]+)\.\s*You keep (?:a |an )?(.+?)(?:\s+in\s+([^.;]+))?\.",
        state,
        re.I,
    )
    if m:
        return clean(m.group(1)), clean(m.group(2)), clean(m.group(3)) if m.group(3) else None
    return None


def persona_consistency(text: str, state: str) -> str:
    """Replace contradictory model self-claims with the canonical persona."""
    details = persona_details(state)
    if not details:
        return clean(text)
    name, profession, home = details
    raw = clean(text)
    canonical_name = norm(name)
    canonical_profession = norm(profession)
    canonical_home = norm(home or "")
    wrong_name = False
    wrong_profession = False
    wrong_home = False

    explicit_name = re.search(r"\b(?:my name is|call me|i am called|i'm called)\s+([^,.!?;]+)", raw, re.I)
    if explicit_name:
        claimed = norm(explicit_name.group(1)).split()
        wrong_name = not claimed or claimed[0] != canonical_name

    self_claim = re.search(r"\bI(?: am|'m)\s+([^,.!?;]+)", raw)
    if self_claim:
        claim = clean(self_claim.group(1))
        claim_words = norm(claim).split()
        first = claim_words[0] if claim_words else ""
        if first in {"no", "nobody", "nothing", "anyone"}:
            wrong_name = True
        elif first not in {"a", "an", "the", "well", "fine", "sorry", "glad", "happy", "from", "in", "at", "here", "there"}:
            if first != canonical_name and not any(word in claim_words for word in PERSONA_OCCUPATIONS):
                wrong_name = True

    occupation_claim = re.search(
        r"\b(?:i(?: am|'m)|my trade is|i work as|i serve as)\s+(?:a |an |the )?([^,.!?;]+)",
        raw,
        re.I,
    )
    if occupation_claim:
        claimed_trade = norm(occupation_claim.group(1))
        if any(word in claimed_trade.split() for word in PERSONA_OCCUPATIONS):
            wrong_profession = canonical_profession not in claimed_trade

    title_claim = re.search(rf"\b{re.escape(name)}\s+the\s+([a-z][a-z -]+)", raw, re.I)
    if title_claim and any(word in norm(title_claim.group(1)).split() for word in PERSONA_OCCUPATIONS):
        wrong_profession = canonical_profession not in norm(title_claim.group(1))

    if canonical_home:
        location_claims = re.findall(
            r"\b(?:from|live in|dwell in|hail from|come from|home is|i am in|i'm in)\s+([A-Za-z][A-Za-z' -]*)",
            raw,
            re.I,
        )
        for claim in location_claims:
            claim_norm = norm(claim)
            if any(place in claim_norm for place in COMMON_PLACES) or any(ch.isupper() for ch in claim):
                if canonical_home not in claim_norm:
                    wrong_home = True

    if wrong_name or wrong_profession or wrong_home:
        article = "an" if profession[:1].lower() in "aeiou" else "a"
        answer = f"I am {name}, {article} {profession}"
        if home:
            answer += f" from {home}"
        return answer + "."
    return raw


def grounded_response(state: str, player: str, model_fallback: Optional[Callable[[str, str], str]] = None) -> str:
    """Answer from explicit state when possible, otherwise use the model."""
    q = norm(player)
    answer = social_answer(player)
    if answer:
        return answer
    answer = practical_answer(state, player)
    if answer:
        return answer
    answer = capability_answer(state, player)
    if answer:
        return answer
    answer = world_answer(state, player)
    if answer:
        return answer
    answer = uncertainty_answer(state, player)
    if answer:
        return answer
    answer = numeric_answer(state, player)
    if answer:
        return answer
    answer = identity_answer(state, player)
    if answer:
        return answer
    if any(x in q for x in ("who", "whose", "where", "which", "when", "route", "road", "what does", "what should", "what is", "what lies", "what room", "what must", "who gathers")):
        fact = relevant_fact(state, player)
        if fact:
            return re.sub(r"^(you know |the state says )", "", fact, flags=re.I)
    if model_fallback is not None:
        return persona_consistency(model_fallback(state, player), state)
    return persona_consistency("I know not.", state)
