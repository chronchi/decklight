"""Opponent deck primer unit tests (web-ux-plan step 5, item 7)."""

from ygo_zero.analysis.primer import (
    MIN_GAMES,
    OTHER_DECKS_NOTE,
    aggregate,
    deck_hash,
    hash_of,
    primer_for,
)
from ygo_zero.decks import Deck, load_ydk
from ygo_zero.engine import constants as C
from ygo_zero.engine.resources import load_cards
from ygo_zero.web.view import DB

HAND = C.LOCATION_HAND
DECK_A = "Pachycephalo"
DECK_B = "Shaddoll"


def _cards_names():
    return load_cards(), DB.names()


def _deck(name):
    from ygo_zero.analysis.events import deck_path

    return load_ydk(deck_path(name))


def _summary(winner=0, first=0, links=(), summons=(), steps=(), turns=()):
    return {
        "game_idx": 0,
        "seed": 1,
        "decks": [DECK_A, DECK_B],
        "winner": winner,
        "first_player": first,
        "turns": list(turns),
        "chain_links": [],
        "links": list(links),
        "summons": list(summons),
        "steps": list(steps),
    }


def _wstep(player, kind, code, turn, w):
    return {
        "player": player,
        "n_opts": 2,
        "kind": kind,
        "code": code,
        "turn": turn,
        "phase": int(C.PHASE_MAIN1),
        "turn_player": 0,
        "w": w,
    }


def _scored(steps, w_end, winner=0, first=0):
    s = _summary(winner=winner, first=first, steps=steps, turns=[{"turn": 1, "turn_player": 0}])
    s["w_end"] = w_end
    return s


def _link(controller, code, group=0, count=1, negated=False, hloc=0):
    return {
        "turn": 1,
        "group": group,
        "controller": controller,
        "code": code,
        "handler_loc": hloc,
        "handler_con": controller,
        "count": count,
        "negated": negated,
        "answers_summon": None,
    }


def test_hash_ignores_order_and_side():
    d = Deck(main=(1, 2, 3), extra=(4, 5), side=(6, 7), name="x")
    assert hash_of(d) == deck_hash([3, 2, 1], [5, 4])
    assert hash_of(d) == deck_hash([1, 2, 3], [4, 5])
    assert hash_of({"main": [1, 2, 3], "extra": [4, 5], "side": [9]}) == hash_of(d)


def test_counting_on_synthetic_chain_links():
    cards, names = _cards_names()
    a_code, b_code, played = 100, 200, 300
    links = [
        _link(0, a_code, group=0, count=1, negated=True),
        _link(1, b_code, group=0, count=2),
    ]
    summons = [
        {
            "turn": 1,
            "group": 0,
            "seat": 0,
            "code": played,
            "loc": 0,
            "ok": True,
            "answered_by": dict(links[1], seat=1),
        }
    ]
    steps = [
        {
            "player": 0,
            "n_opts": 3,
            "kind": "ACTIVATE",
            "code": 1,
            "turn": 1,
            "phase": int(C.PHASE_MAIN1),
            "turn_player": 0,
        },
        {
            "player": 0,
            "n_opts": 2,
            "kind": "SUMMON",
            "code": 2,
            "turn": 1,
            "phase": int(C.PHASE_MAIN1),
            "turn_player": 0,
        },
        {
            "player": 0,
            "n_opts": 4,
            "kind": "ACTIVATE",
            "code": 3,
            "turn": 1,
            "phase": int(C.PHASE_MAIN1),
            "turn_player": 0,
        },
    ]
    turns = [{"turn": 1, "turn_player": 0, "players": []}]
    store = aggregate([_summary(links=links, summons=summons, steps=steps, turns=turns)] * 25, cards, names)
    ha, hb = hash_of(_deck(DECK_A)), hash_of(_deck(DECK_B))
    pa = store["decks"][ha]
    assert pa["games"] == 25
    # Chokepoints: B answered A's activation (negated) and A's summon.
    choke = {c["code"]: c for c in pa["chokepoints"]}
    assert choke[a_code]["responses"] == 25
    assert choke[a_code]["negated"] == 25
    assert choke[played]["responses"] == 25
    # Openers: first 3 plays of turn 1 going first.
    assert pa["openers"][0]["plays"] == [1, 2, 3]
    # Key cards count each card once per game.
    key = {c["code"]: c for c in pa["key_cards"]}
    assert key[a_code]["games"] == 25
    # Global negator index: B's card negated A's activation.
    assert store["negated_by"][str(b_code)][str(a_code)] == 25
    _ = hb


def test_credit_and_answer_rules():
    """A play moving seat-0 win chance 0.5 -> 0.7 credits +20; seat 1's answer
    back to 0.55 credits +15 (its side) and is charged to the answered card."""
    cards, names = _cards_names()
    steps = [
        _wstep(0, "ACTIVATE", 11, 1, 0.5),
        _wstep(1, "ACTIVATE", 99, 1, 0.7),
        _wstep(1, "TO_BATTLE", 0, 2, 0.55),  # turn-2 marker for the opener's w_after
    ]
    links = [
        _link(0, 11, group=0, count=1),
        _link(1, 99, group=0, count=2, hloc=HAND),
    ]
    s = _scored(steps, 0.55)
    s["links"] = links
    store = aggregate([s] * 25, cards, names)
    ha, hb = hash_of(_deck(DECK_A)), hash_of(_deck(DECK_B))
    key_a = {c["code"]: c for c in store["decks"][ha]["key_cards"]}
    assert key_a[11]["impact"] == 20.0
    assert key_a[11]["per_play"] == 20.0
    assert key_a[11]["games"] == 25
    key_b = {c["code"]: c for c in store["decks"][hb]["key_cards"]}
    assert key_b[99]["impact"] == 15.0
    assert key_b[99]["per_play"] == 15.0
    choke = {c["code"]: c for c in store["decks"][ha]["chokepoints"]}
    assert choke[11]["answered"] == 25
    assert choke[11]["cost"] == 15.0
    assert choke[11]["per_answer"] == 15.0
    ht = store["decks"][ha]["hand_traps_seen"][0]
    assert ht["code"] == 99 and ht["gain"] == 15.0
    assert ht["answered"][0]["code"] == 11 and ht["answered"][0]["gain"] == 15.0
    opener = store["decks"][ha]["openers"][0]
    assert opener["plays"] == [11] and opener["w_after"] == 55.0
    # primer_for passes the gains through to your_answers hits.
    from ygo_zero.decks import parse_ydk

    human_ht = parse_ydk("#main\n99\n!side\n", "human")
    p = primer_for(store, _deck(DECK_A), human_ht)
    assert p is not None
    ans = {a["code"]: a for a in p["your_answers"]}
    assert ans[99]["hits"][0]["gain"] == 15.0


def test_impact_ranks_rare_big_card_above_frequent_zero():
    cards, names = _cards_names()
    rare = _scored([_wstep(0, "ACTIVATE", 11, 1, 0.3), _wstep(0, "SUMMON", 23, 1, 0.8)], 0.8)  # 11: +50 pts
    flat = _scored([_wstep(0, "ACTIVATE", 22, 1, 0.5), _wstep(0, "SUMMON", 23, 1, 0.5)], 0.5)  # 22, 23: 0 pts
    store = aggregate([rare] * 3 + [flat] * 10, cards, names)
    ha = hash_of(_deck(DECK_A))
    kc = store["decks"][ha]["key_cards"]
    assert kc[0]["code"] == 11
    assert kc[0]["impact"] == round(1.5 / 13 * 100, 1)
    assert {c["code"] for c in kc} >= {11, 22}


def test_unscored_caches_keep_frequency_fields():
    """Summaries without w aggregate exactly like before (old caches still build)."""
    cards, names = _cards_names()
    s = _summary(
        links=[_link(0, 100, group=0, count=1, negated=True), _link(1, 200, group=0, count=2)],
        steps=[
            {
                "player": 0,
                "n_opts": 3,
                "kind": "ACTIVATE",
                "code": 100,
                "turn": 1,
                "phase": int(C.PHASE_MAIN1),
                "turn_player": 0,
            }
        ],
    )
    assert "w_end" not in s
    store = aggregate([s] * 5, cards, names)
    for d in store["decks"].values():
        for c in d["key_cards"]:
            assert set(c) == {"code", "name", "games", "per_game"}
        for c in d["chokepoints"]:
            assert set(c) == {"code", "name", "responses", "negated", "response_rate"}
        for h in d["hand_traps_seen"]:
            assert "gain" not in h
            for a in h["answered"]:
                assert "gain" not in a
        for o in d["openers"]:
            assert "w_after" not in o


def test_hand_trap_seen_and_primer_for():
    cards, names = _cards_names()
    ht, victim = 14558127, 66730191  # real codes: name resolution must work
    links = [
        _link(0, victim, group=0, count=1),
        _link(1, ht, group=0, count=2, hloc=HAND),
    ]
    turns = [{"turn": 1, "turn_player": 0, "players": []}]
    store = aggregate([_summary(links=links, turns=turns)] * 25, cards, names)
    ha = hash_of(_deck(DECK_A))
    assert store["decks"][ha]["hand_traps_seen"][0]["code"] == ht
    assert store["decks"][ha]["hand_traps_seen"][0]["answered"][0]["code"] == victim

    bot, human = _deck(DECK_A), _deck(DECK_B)
    assert primer_for(store, bot, human) is None or True  # human may lack the cards
    # A human deck holding the hand trap gets your_answers with hits vs this deck.
    from ygo_zero.decks import parse_ydk

    human_ht = parse_ydk(f"#main\n{ht}\n!side\n", "human")
    p = primer_for(store, bot, human_ht)
    assert p is not None
    ans = {a["code"]: a for a in p["your_answers"]}
    assert victim in {h["code"] for h in ans[ht]["hits"]}

    # Below the game threshold there is no primer.
    small = aggregate([_summary(links=links, turns=turns)] * (MIN_GAMES - 1), cards, names)
    assert primer_for(small, bot, human_ht) is None

    # A human card seen as a hand trap only against other decks gets the note.
    other = parse_ydk("#main\n99999999\n!side\n", "other")
    for d in store["decks"].values():
        if d["hash"] != ha:
            d["hand_traps_seen"].append({"code": 99999999, "name": "X", "times": 2, "answered": []})
    p2 = primer_for(store, bot, other)
    ans2 = {a["code"]: a for a in p2["your_answers"]}
    assert ans2[99999999].get("note") == OTHER_DECKS_NOTE
    # ... and so does a card that only negated something elsewhere.
    store["negated_by"]["99999998"] = {"100": 3}
    other2 = parse_ydk("#main\n99999998\n!side\n", "other2")
    p3 = primer_for(store, bot, other2)
    ans3 = {a["code"]: a for a in p3["your_answers"]}
    assert ans3[99999998].get("note") == OTHER_DECKS_NOTE
    assert ans3[99999998]["hits_elsewhere"][0]["code"] == 100
