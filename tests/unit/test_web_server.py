"""Web client W1 server tests (each < 5 min with `taskset -c 0-3`).

1. Full games: `mode=random` and a tiny random-init product, the client choosing
   random option indices until `result`; the record replays to the same winner.
2. Hidden-info: over a whole game, every code in the bot's hand or face-down
   zones never appears in any message to the client before it is public.
3. Token required (401 without it, WS closed without it).
4. `/img` cache hit, miss with a mocked fetch, and the offline fallback.
5. View schema check on every state.
"""

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from starlette.testclient import TestClient

from ygo_zero.decks import load_ydk
from ygo_zero.edopro.bot import RandomBot
from ygo_zero.edopro.product_bot import ProductDriver
from ygo_zero.engine import constants as C
from ygo_zero.engine.duel import DuelConfig
from ygo_zero.engine.env import PyEnv
from ygo_zero.engine.obs import Board
from ygo_zero.model.bc_data import OBS_KEYS
from ygo_zero.model.policy import PolicyNet
from ygo_zero.web.server import create_app
from ygo_zero.web.view import DB

FIXTURES = Path("tests/fixtures/decks")
TOKEN = "test-token"


def _tiny_forward():
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {
        "codes": np.array(codes),
        "card": rng.normal(size=(n, 8)).astype(np.float32),
        "effects": rng.normal(size=(n, 2, 8)).astype(np.float32),
        "n_effects": np.ones(n, np.int64),
    }
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()

    @torch.no_grad()
    def forward(obs_list):
        t = {k: torch.from_numpy(np.ascontiguousarray(np.stack([o[k] for o in obs_list]))) for k in OBS_KEYS}
        return net(t).numpy()

    return forward


class _TinyLauncher:
    """A ProductLauncher-shaped stub holding the tiny model (no checkpoints)."""

    def __init__(self, forward) -> None:
        self.forward = forward
        self.value = None  # fast mode: no value net (winchance stays null)

    def start_duel(
        self, engine, config, bot_seat, *, game_seed, ws_seed, human_fn=None, prefix=(), prefix_decisions=None
    ):
        return ProductDriver(
            engine,
            config,
            bot_seat=bot_seat,
            base_logits=self.forward,
            value=None,
            search_kwargs={},
            game_seed=game_seed,
            ws_seed=ws_seed,
            fast=True,
            human_fn=human_fn,
            prefix=prefix,
            prefix_decisions=prefix_decisions,
        ).start()

    def record_info(self, seed: int) -> dict:
        return {"ckpt": "tiny-test", "fast": True, "game_seed": seed, "search": "tiny"}


class _TinyValueLauncher(_TinyLauncher):
    """Tiny launcher with a spied value net: `value` records every obs batch it
    gets (the server's winchance path only; the driver runs value-free)."""

    def __init__(self, forward) -> None:
        super().__init__(forward)
        self.seen: list[dict] = []

        def value(batch):
            import numpy as _np

            self.seen.append({k: _np.array(v) for k, v in batch.items()})
            n = next(iter(batch.values())).shape[0]
            return _np.zeros(n, dtype=_np.float32)

        self.value = value


def _make_app(tmp_path, engine, mode_factory=None):
    def make_opponent(mode):
        if mode_factory is not None:
            return mode_factory(mode)
        return RandomBot(1234)

    return create_app(
        token=TOKEN,
        engine=engine,
        make_opponent=make_opponent,
        record_root=tmp_path / "runs",
        pics_dir=tmp_path / "pics",
    )


def _client(app):
    return TestClient(app)


# --- schema ---

_CARD_KEYS = {
    "uid": str,
    "code": int,
    "faceup": bool,
    "pos": (str, type(None)),
    "atk": int,
    "def": int,
    "level": int,
    "rank": int,
    "link": int,
    "counters": int,
    "overlays": list,
}
_SIDE_KEYS = {
    "lp": int,
    "deck": int,
    "extra_count": int,
    "hand": list,
    "mzone": list,
    "emz": list,
    "szone": list,
    "field": (dict, type(None)),
    "gy": list,
    "banished": list,
    "extra": list,
}


def _check_card(card, where: str) -> None:
    assert isinstance(card, dict), f"{where}: card is not a dict"
    assert set(card) == set(_CARD_KEYS) | {"name"}, f"{where}: card keys {sorted(card)}"
    for k, types in _CARD_KEYS.items():
        assert isinstance(card[k], types), f"{where}: card[{k}]={card[k]!r}"
    assert card["name"] is None or isinstance(card["name"], str), where
    if card["code"] == 0:
        assert card["name"] is None, f"{where}: hidden card has a name"
    else:
        assert card["name"], f"{where}: visible card has no name"
    assert card["pos"] in ("atk", "def", "set", None), f"{where}: pos {card['pos']!r}"
    for sub in card["overlays"]:
        _check_card(sub, f"{where}/overlay")


_RAW_FMT = re.compile(r"%l|%d|%s")


def _check_no_placeholders(msg: dict) -> None:
    """EDOPro printf placeholders (%ls, %d, %s) never reach the player."""
    texts = [e["text"] for e in msg["log"]]
    if msg["prompt"] is not None:
        p = msg["prompt"]
        texts.append(p["text"])
        texts += [o["label"] for o in p["options"]] + [o["effect"] or "" for o in p["options"]]
    bad = [t for t in texts if _RAW_FMT.search(t)]
    assert not bad, f"raw format placeholders shown: {bad[:3]}"


def _check_state(msg: dict) -> None:
    _check_no_placeholders(msg)
    assert msg["t"] == "state"
    view = msg["view"]
    assert set(view) == {"turn", "phase", "turn_player", "you", "bot", "chain"}
    assert isinstance(view["turn"], int) and view["turn"] >= 1
    assert view["phase"] in ("draw", "standby", "main1", "battle", "main2", "end")
    assert view["turn_player"] in ("you", "bot")
    for side_name in ("you", "bot"):
        side = view[side_name]
        assert set(side) == set(_SIDE_KEYS), f"{side_name}: {sorted(side)}"
        for k, types in _SIDE_KEYS.items():
            assert isinstance(side[k], types), f"{side_name}.{k}"
        assert len(side["mzone"]) == 5 and len(side["emz"]) == 2
        assert len(side["szone"]) == 5
        for zone in ("hand", "mzone", "emz", "szone", "gy", "banished", "extra"):
            for c in side[zone]:
                if c is not None:
                    _check_card(c, f"{side_name}.{zone}")
        if side["field"] is not None:
            _check_card(side["field"], f"{side_name}.field")
    for link in view["chain"]:
        assert set(link) == {"card", "controller"}
        assert link["controller"] in ("you", "bot")
        _check_card(link["card"], "chain")
    prompt = msg["prompt"]
    if prompt is not None:
        assert set(prompt) == {"id", "kind", "text", "card", "options", "min", "max"}
        assert isinstance(prompt["id"], int) and isinstance(prompt["text"], str)
        assert prompt["kind"] in (
            "idle",
            "battle",
            "chain",
            "card",
            "yesno",
            "effectyn",
            "option",
            "position",
            "place",
            "announce",
            "sort",
            "counter",
        )
        assert isinstance(prompt["min"], int) and isinstance(prompt["max"], int)
        if prompt["card"] is not None:
            _check_card(prompt["card"], "prompt.card")
        assert len(prompt["options"]) > 1
        for o in prompt["options"]:
            assert set(o) == {"index", "verb", "label", "card", "where", "effect"}
            assert isinstance(o["index"], int) and isinstance(o["label"], str)
            assert isinstance(o["verb"], str) and o["label"]
            if o["card"] is not None:
                _check_card(o["card"], "prompt.option.card")
            if o["where"] is not None:
                assert set(o["where"]) == {"side", "zone", "seq"}
                assert o["where"]["side"] in ("you", "bot")
                assert o["where"]["zone"] in (
                    "hand",
                    "mzone",
                    "emz",
                    "szone",
                    "field",
                    "gy",
                    "banished",
                    "extra",
                    "deck",
                )
            assert o["effect"] is None or isinstance(o["effect"], str)
    assert isinstance(msg["log"], list)
    for e in msg["log"]:
        assert set(e) == {"n", "actor", "text", "card", "kind"}
        assert e["actor"] in ("you", "bot", "system") and isinstance(e["text"], str)
        if e["card"] is not None:
            _check_card(e["card"], "log.card")
    assert msg["thinking"] is None or (
        set(msg["thinking"]) == {"seconds"} and isinstance(msg["thinking"]["seconds"], float)
    )
    assert msg["result"] is None or (
        set(msg["result"]) == {"winner", "reason"}
        and msg["result"]["winner"] in ("you", "bot")
        and isinstance(msg["result"]["reason"], str)
    )
    st = msg.get("stats")
    if st is not None:
        for k in (
            "turn",
            "lp_you",
            "lp_bot",
            "dealt_you",
            "dealt_bot",
            "adv_you",
            "adv_bot",
            "adv_diff",
            "windows_used",
            "windows_total",
            "neg_you",
            "neg_were_you",
            "neg_bot",
            "neg_were_bot",
            "chart_labels",
            "chart_you",
            "chart_bot",
            "moments",
            "ledger",
            "opp_seen",
            "you_used",
        ):
            assert k in st, f"stats missing {k}"
        assert st["windows_used"] <= st["windows_total"]
        assert len(st["chart_labels"]) == len(st["chart_you"]) == len(st["chart_bot"])
        assert len(st["moments"]) <= 5


def _codes_in_state(msg: dict) -> tuple[set[int], str]:
    """All card codes shown in a state message + its full text for name checks."""
    codes: set[int] = set()

    def walk(card) -> None:
        if card is None:
            return
        if card["code"]:
            codes.add(card["code"])
        for sub in card["overlays"]:
            walk(sub)

    for side_name in ("you", "bot"):
        side = msg["view"][side_name]
        for zone in ("hand", "mzone", "emz", "szone", "gy", "banished", "extra"):
            for c in side[zone]:
                walk(c)
        walk(side["field"])
    for link in msg["view"]["chain"]:
        walk(link["card"])
    if msg["prompt"] is not None:
        walk(msg["prompt"]["card"])
        for o in msg["prompt"]["options"]:
            walk(o["card"])
    for e in msg["log"]:
        walk(e["card"])
    st = msg.get("stats") or {}
    for lst_key in ("opp_seen", "you_used"):
        for e in st.get(lst_key) or []:
            if isinstance(e, dict) and e.get("code"):
                codes.add(int(e["code"]))

    # Name checks scan everything except effect descriptions: a public card's effect text may name a hidden card
    # ("... using \"Fallen of Albaz\" as material") without revealing it (NOTEBOOK 2026-10-17).
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "effect"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    return codes, json.dumps(strip(msg))


# --- game driver ---


def _play_to_end(ws, rng, *, collect=None) -> dict:
    """Answer every prompt with a random option until `result`; returns the last state."""
    states = 0
    while True:
        raw = ws.receive_text()
        msg = json.loads(raw)
        if msg.get("t") == "error":
            raise AssertionError(f"server error: {msg['text']}")
        assert msg.get("t") == "state", f"unexpected message {msg.get('t')}"
        _check_state(msg)
        if collect is not None:
            collect.append(msg)
        states += 1
        assert states < 20000, "game did not finish (random play exploded?)"
        if msg["result"] is not None:
            return msg
        if msg["prompt"] is not None:
            n = len(msg["prompt"]["options"])
            ws.send_json({"t": "choose", "prompt": msg["prompt"]["id"], "index": rng.randrange(n)})


def _new_game(ws, human_text: str, bot_text: str, **kw) -> None:
    msg = {
        "t": "new_game",
        "deck": {"ydk": human_text},
        "bot_deck": {"ydk": bot_text},
        "first": "me",
        "mode": "random",
    }
    msg.update(kw)
    ws.send_json(msg)


def _hidden_codes_of(env, human: int) -> set[int]:
    """Bot codes the human must not see at the env's current state."""
    bot = 1 - human
    duel = env.duel
    visible = Board(duel, human).visible_codes()
    hidden = set()
    for loc in (C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED, C.LOCATION_EXTRA):
        for c in duel.query_location(bot, loc):
            if c is not None and c.code and c.code not in visible:
                hidden.add(c.code)
    return hidden


def _verify_no_leak(engine, cfg, human_seat: int, choices: list[int], messages: list[dict]) -> None:
    """Every code in the bot's hand/face-down zones never reaches the client
    before it is public: each state's `log` carries absolute history indices
    (`n`), and the replay is deterministic, so message with max-n h is checked
    against the hidden set after replaying h choices. Offline (no live queries,
    which would race the server thread)."""
    hs = []
    run_h = 0
    for m in messages:
        # history length at send time: every session step appends exactly one
        # history entry, and non-peek sends deliver every entry exactly once, so
        # the running max of `n` is the send-time length (header-only states,
        # e.g. the first prompt with no auto-steps before it, reuse the last).
        run_h = max(run_h, max((e["n"] for e in m["log"]), default=0))
        hs.append(run_h)
    needed = set(hs)
    hidden_by_h: dict[int, set[int]] = {}
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    # Codes revealed by announcement: a public history entry (e.g. the bot
    # activating a card from its hand), a chain link (every chained card is
    # announced), or an engine confirm (searches, looks, set-confirms, all
    # shown to both players) makes the code public from then on, even while
    # the card is still physically hidden - the same rules the model's own
    # observation follows (`obs.history_entry.public`, `obs.chain_array`).
    from ygo_zero.engine.query import parse_chain

    revealed: set[int] = set()

    def note_revealed() -> None:
        for ch in parse_chain(env.duel.query_field()):
            if ch.code:
                revealed.add(ch.code)

    def note_confirmed(prev_trace: int) -> int:
        """Codes the engine revealed via MSG_CONFIRM_CARDS (searches, looks,
        set-confirms: the cues surface them, and EDOPro shows them to both
        players, so they are public from then on). Returns the new trace len."""
        import struct

        trace = env.duel.trace
        for raw in trace[prev_trace:]:
            if raw[0] != C.MSG_CONFIRM_CARDS or len(raw) < 6:
                continue
            (count,) = struct.unpack_from("<I", raw, 2)
            pos = 6
            for _ in range(count):
                if pos + 10 > len(raw):
                    break
                (code,) = struct.unpack_from("<I", raw, pos)
                if code:
                    revealed.add(code)
                pos += 10
        return len(trace)

    prev_trace = note_confirmed(0)
    note_revealed()
    if 0 in needed:
        hidden_by_h[0] = _hidden_codes_of(env, human_seat) - revealed
    assert len(choices) >= max(needed)
    for h_cur in range(1, max(needed) + 1):
        env.step(int(choices[h_cur - 1]))
        entry = env.history[h_cur - 1]
        if entry.public and entry.row[2]:
            revealed.add(entry.row[2])
        prev_trace = note_confirmed(prev_trace)
        note_revealed()
        if h_cur in needed:
            hidden_by_h[h_cur] = _hidden_codes_of(env, human_seat) - revealed
    names = DB.names()
    bad: list[str] = []
    for m, h in zip(messages, hs):
        codes, text = _codes_in_state(m)
        for code in sorted(hidden_by_h[h] & codes):
            bad.append(f"history {h}: hidden code {code} shown")
        for code in sorted(hidden_by_h[h]):
            nm = names.get(code, "")
            if nm and nm in text:
                bad.append(f"history {h}: hidden name {nm!r} shown")
    assert not bad, f"hidden info leaked: {bad[:5]}"


def _full_game(tmp_path, engine, mode, make_opponent=None, seed=0):
    app = _make_app(tmp_path, engine, mode_factory=make_opponent)
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    rng = random.Random(seed)
    messages: list[dict] = []
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        lobby = json.loads(ws.receive_text())
        assert lobby["t"] == "lobby" and lobby["decks"], "no lobby on connect"
        _new_game(ws, human_text, bot_text, mode=mode)
        last = _play_to_end(ws, rng, collect=messages)
    assert messages, "no states received"
    return app, last, messages, human_text, bot_text


def _replay_record(engine, record_root: Path, last: dict, decks: tuple) -> tuple[dict, object]:
    subs = [p for p in record_root.iterdir() if p.is_dir()]
    assert len(subs) == 1 and (subs[0] / "human.ydk").is_file()
    rec = json.loads((subs[0] / "game.jsonl").read_text())
    human, bot = decks
    cfg = (
        DuelConfig(human, bot, seed=rec["seed"])
        if rec["human_seat"] == 0
        else DuelConfig(bot, human, seed=rec["seed"])
    )
    assert rec["auto_single"] is False
    assert len(rec["actors"]) == len(rec["choices"]) and rec["choices"]
    env = PyEnv.replay(engine, cfg, rec["choices"], auto_single=False)
    assert env.done and env.winner == rec["winner"]
    want = "you" if rec["winner"] == rec["human_seat"] else "bot"
    assert last["result"]["winner"] == want
    return rec, cfg


# --- 1. full games ---


def test_full_game_random(tmp_path, engine):
    from ygo_zero.decks import parse_ydk

    app, last, messages, human_text, bot_text = _full_game(tmp_path, engine, "random")
    _ = app
    human, bot = parse_ydk(human_text), parse_ydk(bot_text)
    rec, cfg = _replay_record(engine, tmp_path / "runs", last, (human, bot))
    assert rec["winner"] in (0, 1) and rec["turns"] > 0
    assert "product" not in rec
    _verify_no_leak(engine, cfg, rec["human_seat"], rec["choices"], messages)


def test_full_game_tiny_product(tmp_path, engine):
    from ygo_zero.decks import parse_ydk

    forward = _tiny_forward()
    app, last, messages, human_text, bot_text = _full_game(
        tmp_path, engine, "product", make_opponent=lambda mode: _TinyLauncher(forward)
    )
    _ = app
    human, bot = parse_ydk(human_text), parse_ydk(bot_text)
    rec, cfg = _replay_record(engine, tmp_path / "runs", last, (human, bot))
    assert rec["winner"] in (0, 1)
    assert rec["product"]["ckpt"] == "tiny-test"
    _verify_no_leak(engine, cfg, rec["human_seat"], rec["choices"], messages)


# --- 2. hidden-info on another seed ---


def test_hidden_info_over_full_game(tmp_path, engine):
    from ygo_zero.decks import parse_ydk

    app, last, messages, human_text, bot_text = _full_game(tmp_path, engine, "random", seed=1)
    _ = app
    human, bot = parse_ydk(human_text), parse_ydk(bot_text)
    rec, cfg = _replay_record(engine, tmp_path / "runs", last, (human, bot))
    _verify_no_leak(engine, cfg, rec["human_seat"], rec["choices"], messages)


# --- 3. token ---


def test_token_required(tmp_path, engine):
    app = _make_app(tmp_path, engine)
    client = _client(app)
    r = client.get("/card/14558128")
    assert r.status_code == 401
    r = client.get("/img/14558128.jpg?size=small")
    assert r.status_code == 401
    r = client.get("/")
    assert r.status_code == 401
    r = client.get("/card/14558128", params={"t": TOKEN})
    assert r.status_code == 200 and r.json()["code"] == 14558128
    try:
        with client.websocket_connect("/ws"):
            raise AssertionError("websocket without token was accepted")
    except Exception as e:  # noqa: BLE001 -- any close/reject proves the token gate
        assert "accepted" not in str(e)


# --- 4. images and card text ---

JPEG = b"\xff\xd8\xff\xe0" + b"fake-jpeg"


def test_img_cache_hit(tmp_path, engine):
    app = _make_app(tmp_path, engine)
    pics = tmp_path / "pics" / "small"
    pics.mkdir(parents=True)
    (pics / "123.jpg").write_bytes(JPEG)
    r = _client(app).get("/img/123.jpg", params={"t": TOKEN, "size": "small"})
    assert r.status_code == 200 and r.content == JPEG


def test_img_miss_then_cache(tmp_path, engine, monkeypatch):
    import ygo_zero.web.server as S

    app = _make_app(tmp_path, engine)
    calls = []
    monkeypatch.setattr(S, "_fetch_img", lambda url, timeout=5.0: calls.append(url) or JPEG)
    client = _client(app)
    r = client.get("/img/456.jpg", params={"t": TOKEN, "size": "small"})
    assert r.status_code == 200 and r.content == JPEG
    assert len(calls) == 1 and "456.jpg" in calls[0]
    assert (tmp_path / "pics" / "small" / "456.jpg").is_file()
    monkeypatch.setattr(S, "_fetch_img", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    r = client.get("/img/456.jpg", params={"t": TOKEN, "size": "small"})
    assert r.status_code == 200 and r.content == JPEG  # served from cache


def test_img_offline_fallback_and_back(tmp_path, engine, monkeypatch):
    import ygo_zero.web.server as S

    app = _make_app(tmp_path, engine)
    monkeypatch.setattr(S, "_fetch_img", lambda *a, **k: (_ for _ in ()).throw(OSError("off")))
    client = _client(app)
    r = client.get("/img/789.jpg", params={"t": TOKEN})
    assert r.status_code == 200 and r.headers["content-type"] == "image/svg+xml"
    r = client.get("/img/0.jpg", params={"t": TOKEN})
    assert r.status_code == 200 and r.headers["content-type"] == "image/svg+xml"


def test_card_json(tmp_path, engine):
    app = _make_app(tmp_path, engine)
    r = _client(app).get("/card/14558128", params={"t": TOKEN})
    body = r.json()
    assert body["name"] == "Ash Blossom & Joyous Spring"
    assert body["level"] == 3 and body["atk"] == 0 and body["def"] == 1800
    assert "Tuner" in body["type_str"] and body["text"]


# --- 5. surrender, resync, takeover ---


def test_surrender_resync_takeover(tmp_path, engine):
    app = _make_app(tmp_path, engine)
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws1:
        json.loads(ws1.receive_text())  # lobby
        _new_game(ws1, human_text, bot_text, mode="random")
        # wait for the first prompt: advance is fully idle, nothing in flight
        for _ in range(1000):
            first = json.loads(ws1.receive_text())
            assert first["t"] == "state"
            if first["prompt"] is not None:
                break
        with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws2:
            assert json.loads(ws2.receive_text())["t"] == "lobby"
            # a live game also pushes a full state on connect; resync sends another
            ws2.send_json({"t": "resync"})
            full = json.loads(ws2.receive_text())
            assert full["t"] == "state"
            _check_state(full)
            assert full["prompt"] is not None  # the full log + current prompt
            assert full["prompt"]["id"] == first["prompt"]["id"]
            ws2.send_json({"t": "surrender"})
            for _ in range(5):
                last = json.loads(ws2.receive_text())
                assert last["t"] == "state"
                _check_state(last)
                if last["result"] is not None:
                    break
            assert last["result"] == {"winner": "bot", "reason": "surrender"}
    subs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(subs) == 1
    rec = json.loads((subs[0] / "game.jsonl").read_text())
    assert rec["surrendered"] is True and rec["auto_single"] is False


def _records(tmp_path) -> list[dict]:
    root = tmp_path / "runs"
    return [json.loads((p / "game.jsonl").read_text()) for p in sorted(root.iterdir()) if p.is_dir()]


def test_chain_auto_vs_all(tmp_path, engine):
    """Auto (default) skips quiet chain windows server-side (logged as system
    "(auto-passed)"), All shows every one; both records replay and carry meta."""
    from ygo_zero.decks import parse_ydk

    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    out = {}
    for chain in ("auto", "all"):
        app = _make_app(tmp_path / chain, engine)
        msgs: list[dict] = []
        with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
            ws.receive_text()
            _new_game(ws, human_text, bot_text, chain=chain, seed=3)
            last = _play_to_end(ws, random.Random(3), collect=msgs)
        rec, _ = _replay_record(
            engine, tmp_path / chain / "runs", last, (parse_ydk(human_text), parse_ydk(bot_text))
        )
        assert rec["chain"] == chain and rec["mode"] == "random"
        assert rec["human_deck"] == "pasted" and rec["bot_deck"] == "pasted"
        assert (next((tmp_path / chain / "runs").iterdir()) / "bot.ydk").is_file()
        auto = [e for m in msgs for e in m["log"] if e["text"] == "(auto-passed)"]
        assert all(e["actor"] == "system" and e["card"] is None for e in auto)
        out[chain] = len(auto)
    assert out["auto"] > 0, "auto never passed a quiet chain window"
    assert out["all"] == 0


class _SlowLauncher(_TinyLauncher):
    def __init__(self, forward, delay: float) -> None:
        def slow(obs_list):
            time.sleep(delay)
            return forward(obs_list)

        super().__init__(slow)


def test_surrender_while_bot_thinks(tmp_path, engine):
    """Surrender cuts through a thinking product bot within ~1 s, and the abandoned
    driver does not write a second (diverged) record."""
    forward = _tiny_forward()
    # each forward outlasts the server's 1 s thinking ticker, so a thinking state is always sent; at 0.5 s a short
    # bot turn could finish unseen and the loop below waited forever at the human's prompt (NOTEBOOK 2026-10-17)
    app = _make_app(tmp_path, engine, mode_factory=lambda m: _SlowLauncher(forward, 1.3))
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        _new_game(ws, human_text, bot_text, mode="product", first="bot")
        for _ in range(2000):
            m = json.loads(ws.receive_text())
            assert m["t"] == "state" and m["result"] is None
            if m["thinking"] is not None and m["prompt"] is None:
                break
            assert m["prompt"] is None, "the bot's turn ended before a thinking state was seen"
        else:
            raise AssertionError("bot never reported thinking")
        t0 = time.monotonic()
        ws.send_json({"t": "surrender"})
        for _ in range(50):
            m = json.loads(ws.receive_text())
            if m.get("result") is not None:
                break
        dt = time.monotonic() - t0
        assert m["result"] == {"winner": "bot", "reason": "surrender"}
        assert dt < 1.5, f"surrender took {dt:.2f}s"
    time.sleep(1.0)  # let the abandoned step unwind
    recs = _records(tmp_path)
    assert len(recs) == 1 and recs[0]["surrendered"] and not recs[0].get("diverged")
    assert recs[0]["mode"] == "product"


def test_settings_message(tmp_path, engine):
    app = _make_app(tmp_path, engine)
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        ws.send_json({"t": "settings", "chain": "bogus"})
        assert json.loads(ws.receive_text())["t"] == "error"
        ws.send_json({"t": "settings", "chain": "all"})
        ws.send_json({"t": "resync"})
        assert json.loads(ws.receive_text())["t"] == "lobby"


# --- 6. protocol errors ---


def test_choose_errors(tmp_path, engine):
    app = _make_app(tmp_path, engine)
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()  # lobby
        ws.send_json({"t": "choose", "prompt": 0, "index": 0})
        assert json.loads(ws.receive_text())["t"] == "error"  # no game yet
        ws.send_json({"t": "new_game", "deck": {"ydk": "nonsense"}, "mode": "random"})
        assert json.loads(ws.receive_text())["t"] == "error"  # illegal deck
        ws.send_json({"t": "bogus"})
        assert json.loads(ws.receive_text())["t"] == "error"  # unknown type


# --- 7. prompt cards and texts on a real phone game (Mitsurugi, 2026-10-16) ---


def test_user_record_prompts(engine):
    """Replays the user's game through the builders: the position prompt shows the
    Ritual monster (not the Ritual Spell on the chain), Deck searches send the
    cards face-up with codes, and no raw %ls reaches any prompt."""
    from ygo_zero.web.view import build_prompt

    d = Path("tests/fixtures/web/mitsurugi-20261016")
    rec = json.loads((d / "game.jsonl").read_text())
    human, bot = load_ydk(d / "human.ydk"), load_ydk(d / "bot.ydk")
    cfg = (
        DuelConfig(human, bot, seed=rec["seed"])
        if rec["human_seat"] == 0
        else DuelConfig(bot, human, seed=rec["seed"])
    )
    env = PyEnv.replay(engine, cfg, [], auto_single=False)
    hs = rec["human_seat"]
    seen = set()
    for choice in rec["choices"]:
        p = build_prompt(env, hs, 1)
        if p is not None:
            texts = (
                [p["text"]] + [o["label"] for o in p["options"]] + [o["effect"] or "" for o in p["options"]]
            )
            assert not [t for t in texts if _RAW_FMT.search(t)], texts
            if p["kind"] == "position":
                seen.add("position")
                assert p["card"]["name"] != "Mitsurugi Mirror" and p["card"]["atk"] > 0, p["card"]
            if p["kind"] == "card" and "add to your hand" in p["text"]:
                seen.add("search")
                for o in p["options"]:
                    if o["verb"] == "select":
                        assert o["card"]["code"] and o["card"]["faceup"], o["card"]
            if p["kind"] == "effectyn":
                seen.add("effectyn")
        env.step(choice)
    assert seen == {"position", "search", "effectyn"}, seen


# --- 8. win chance (off by default, human obs only) ---


def _play_product_value(tmp_path, engine, *, winchance: bool, seed=0, first="bot"):
    """A full tiny-product game with a spied value net; returns (app, launcher, states)."""
    forward = _tiny_forward()
    launcher = _TinyValueLauncher(forward)
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: launcher)
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    states: list[dict] = []
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        assert json.loads(ws.receive_text())["t"] == "lobby"
        if winchance:
            ws.send_json({"t": "settings", "chain": "auto", "winchance": True})
        msg = {
            "t": "new_game",
            "deck": {"ydk": human_text},
            "bot_deck": {"ydk": bot_text},
            "first": first,
            "mode": "product",
        }
        ws.send_json(msg)
        _play_to_end(ws, random.Random(seed), collect=states)
    return app, launcher, states


def _check_winchance_shape(wc: dict) -> None:
    assert set(wc) == {"p", "turn", "history"}
    assert isinstance(wc["turn"], int) and wc["turn"] >= 1
    assert 0.03 <= wc["p"] <= 0.97
    assert wc["history"], "history must hold every sample so far"
    for t, p in wc["history"]:
        assert isinstance(t, int) and 0.03 <= p <= 0.97
    assert wc["p"] == wc["history"][-1][1] and wc["turn"] == wc["history"][-1][0]


def test_winchance_off_by_default(tmp_path, engine):
    _, launcher, states = _play_product_value(tmp_path, engine, winchance=False)
    assert states, "no states received"
    for m in states:
        assert m.get("winchance") is None, "the switch is off: winchance must be null"
        assert "winchance" not in (m.get("stats") or {}), "no stats win-chance field while off"
    assert launcher.seen == [], "no value forward while the switch is off"


def test_winchance_on_product(tmp_path, engine):
    from ygo_zero.decks import parse_ydk

    app, launcher, states = _play_product_value(tmp_path, engine, winchance=True)
    assert states, "no states received"
    first_prompt = next(i for i, m in enumerate(states) if m["prompt"] is not None)
    assert first_prompt > 0, "the bot starts: states before the first human prompt exist"
    for m in states[:first_prompt]:
        assert m.get("winchance") is None, "no value before the first human decision"
        assert "winchance" not in (m.get("stats") or {})
    for m in states[first_prompt:]:
        assert m.get("winchance") is not None, "the last value is kept between human decisions"
        _check_winchance_shape(m["winchance"])
        st = m.get("stats") or {}
        assert "winchance" in st, "stats carry the history while on"
        assert st["winchance"]["history"] == m["winchance"]["history"]
    # The leak guard: every value call saw a genuine human-seat observation of
    # this game (rebuilt by replay), each at a distinct human decision.
    assert launcher.seen, "the value net was never called"
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    rec, cfg = _replay_record(
        engine, tmp_path / "runs", states[-1], (parse_ydk(human_text), parse_ydk(bot_text))
    )
    env = PyEnv.replay(engine, cfg, [], auto_single=False)
    hs = rec["human_seat"]
    first_player = env.duel.first_player
    want = []  # (turn, obs) at every human multi-option decision
    for choice in rec["choices"]:
        if env.done:
            break
        if env.player == hs and len(env.opts) > 1:
            want.append(
                (
                    int(env.duel.turn),
                    {k: np.ascontiguousarray(v)[np.newaxis, ...] for k, v in env._observe().items()},
                )
            )
        env.step(int(choice))
    assert len(launcher.seen) <= len(want), "more value calls than human decisions"
    matched = []
    for batch in launcher.seen:
        assert batch["global"].shape[0] == 1
        g = batch["global"][0]
        assert int(g[4]) == int(first_player == hs), "value obs is not the human seat's"
        hits = [i for i, (_, w) in enumerate(want) if all(np.array_equal(batch[k], w[k]) for k in w)]
        assert hits, "value obs matches no human decision of this game"
        matched.extend(hits)
    assert len(set(matched)) == len(matched) == len(launcher.seen), (
        "each value call must sit at a distinct human decision"
    )
    # A new connection resets the switch: the stored value is not sent anymore.
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws2:
        assert json.loads(ws2.receive_text())["t"] == "lobby"
        ws2.send_json({"t": "resync"})
        full = json.loads(ws2.receive_text())
        assert full["t"] == "state"
        assert full.get("winchance") is None
        assert "winchance" not in (full.get("stats") or {})


def test_settings_winchance_validation(tmp_path, engine):
    app = _make_app(tmp_path, engine)
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()  # lobby
        ws.send_json({"t": "settings", "winchance": "yes"})
        assert json.loads(ws.receive_text())["t"] == "error"
        ws.send_json({"t": "settings", "winchance": True})  # no chain key: accepted
        ws.send_json({"t": "resync"})
        assert json.loads(ws.receive_text())["t"] == "lobby"  # no error in between


# --- 9. post-game review (product mode only, after game over) ---


def _tiny_review_launcher(forward):
    class _L(_TinyLauncher):
        def __init__(self) -> None:
            super().__init__(forward)
            seen: list[dict] = []
            self._seen = seen

            def value(batch):
                import numpy as _np

                self._seen.append({k: _np.array(v) for k, v in batch.items()})
                n = next(iter(batch.values())).shape[0]
                return _np.zeros(n, dtype=_np.float32)

            self.value = value
            self.value_seen = seen

        def base_logits(self, obs_list):
            return self.forward(obs_list)

    return _L()


def _play_tiny_to_end(ws, rng):
    states = 0
    while True:
        raw = ws.receive_text()
        msg = json.loads(raw)
        if msg.get("t") == "error":
            raise AssertionError(f"server error: {msg['text']}")
        if msg.get("t") in ("review_progress", "review", "review_frame"):
            continue  # a previous review's async messages: not part of this game
        assert msg.get("t") == "state"
        states += 1
        assert states < 20000
        if msg["result"] is not None:
            return msg
        if msg["prompt"] is not None:
            ws.send_json(
                {
                    "t": "choose",
                    "prompt": msg["prompt"]["id"],
                    "index": rng.randrange(len(msg["prompt"]["options"])),
                }
            )


def test_review_before_game_over_errors(tmp_path, engine):
    forward = _tiny_forward()
    launcher = _tiny_review_launcher(forward)
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: launcher)
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        _new_game(ws, human_text, bot_text, mode="product")
        for _ in range(1000):  # drain to the first prompt (game is live, not over)
            m = json.loads(ws.receive_text())
            if m.get("prompt") is not None:
                break
        ws.send_json({"t": "review"})
        for _ in range(1000):
            m = json.loads(ws.receive_text())
            if m.get("t") == "error":
                break
            assert m.get("t") == "state", m.get("t")
        assert m["t"] == "error"


def test_review_random_mode_errors(tmp_path, engine):
    app = _make_app(tmp_path, engine)
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        _new_game(ws, human_text, bot_text, mode="random")
        _play_to_end(ws, random.Random(0))
        ws.send_json({"t": "review"})
        err = json.loads(ws.receive_text())
        assert err["t"] == "error" and err["text"] == "Review needs the full bot"


def test_review_after_tiny_product_game(tmp_path, engine):
    from ygo_zero.engine.env import PyEnv

    forward = _tiny_forward()
    launcher = _tiny_review_launcher(forward)
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: launcher)
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        _new_game(ws, human_text, bot_text, mode="product", first="me")
        # Play a few human decisions then surrender: a finished (surrendered)
        # game reviews the same way as a full one, but stays fast on CPU.
        rng = random.Random(0)
        last = None
        human_moves = 0
        for _ in range(2000):
            m = json.loads(ws.receive_text())
            if m.get("t") in ("review_progress", "review", "review_frame"):
                continue
            assert m.get("t") == "state"
            if m["result"] is not None:
                last = m
                break
            if m["prompt"] is not None:
                ws.send_json(
                    {
                        "t": "choose",
                        "prompt": m["prompt"]["id"],
                        "index": rng.randrange(len(m["prompt"]["options"])),
                    }
                )
                human_moves += 1
                if human_moves >= 4:
                    ws.send_json({"t": "surrender"})
                    for _ in range(50):
                        m2 = json.loads(ws.receive_text())
                        if m2.get("t") == "state" and m2.get("result") is not None:
                            last = m2
                            break
                    break
        assert last is not None and last["result"] is not None
        n_multi = None
        ws.send_json({"t": "review"})
        review = None
        for _ in range(600):
            m = json.loads(ws.receive_text())
            if m.get("t") == "review_progress":
                assert m["done"] <= m["total"]
                continue
            if m.get("t") == "review":
                review = m
                break
            assert m.get("t") == "state", m.get("t")
        assert review is not None, "no review message"
        assert review["decisions"], "empty review"
        # one entry per multi-option human decision, in order.
        recs = [
            json.loads((p / "game.jsonl").read_text())
            for p in sorted((tmp_path / "runs").iterdir())
            if p.is_dir()
        ]
        assert len(recs) == 1 and "auto_passed" in recs[0]
        rec = recs[0]
        from ygo_zero.decks import parse_ydk

        human, bot = parse_ydk(human_text), parse_ydk(bot_text)
        cfg = (
            DuelConfig(human, bot, seed=rec["seed"])
            if rec["human_seat"] == 0
            else DuelConfig(bot, human, seed=rec["seed"])
        )
        env = PyEnv(engine, cfg, auto_single=False)
        env.reset()
        want = 0
        for ch in rec["choices"]:
            if env.done:
                break
            if env.player == rec["human_seat"] and len(env.opts) > 1:
                want += 1
            env.step(int(ch))
        assert len(review["decisions"]) == want
        assert [e["k"] for e in review["decisions"]] == list(range(want))
        # cached: a second request returns the same review at once.
        ws.send_json({"t": "review"})
        again = json.loads(ws.receive_text())
        assert again.get("t") == "review"
        assert again["decisions"] == review["decisions"]
        # the board snapshot for a decision.
        ws.send_json({"t": "review_frame", "k": 0})
        frame = json.loads(ws.receive_text())
        assert frame.get("t") == "review_frame" and frame.get("k") == 0
        assert "view" in frame and "options" in frame and "entry" in frame
        _check_state(
            {
                "t": "state",
                "view": frame["view"],
                "prompt": None,
                "log": [],
                "cues": [],
                "stats": last.get("stats"),
                "winchance": None,
                "thinking": None,
                "result": None,
            }
        )
        n_multi = want
        assert n_multi is not None


# --- 10. practice hints (step 4): top-3, practice-only gate ---


def _wait_prompt(ws, limit=2000):
    for _ in range(limit):
        m = json.loads(ws.receive_text())
        assert m.get("t") == "state", m.get("t")
        if m["prompt"] is not None:
            return m
    raise AssertionError("no human prompt arrived")


def test_hint_practice_game(tmp_path, engine):
    """A practice-game hint returns <= 3 options with probabilities summing to <= 1."""
    forward = _tiny_forward()
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: _TinyLauncher(forward))
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        ws.send_json(
            {
                "t": "new_game",
                "deck": {"ydk": human_text},
                "bot_deck": {"ydk": bot_text},
                "first": "me",
                "mode": "product",
                "practice": True,
            }
        )
        first = _wait_prompt(ws)
        assert first.get("practice") is True
        ws.send_json({"t": "hint"})
        hm = json.loads(ws.receive_text())
        assert hm["t"] == "hint" and hm["prompt"] == first["prompt"]["id"]
        assert 1 <= len(hm["options"]) <= 3
        assert sum(o["p"] for o in hm["options"]) <= 1.0 + 1e-6
        for o in hm["options"]:
            assert set(o) >= {"index", "label", "p", "est"} and o["label"]
            assert 0 <= o["index"] < len(first["prompt"]["options"])


def test_hint_duel_mode_rejected(tmp_path, engine):
    """A duel-mode hint is rejected (hints are practice only)."""
    forward = _tiny_forward()
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: _TinyLauncher(forward))
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        ws.send_json(
            {
                "t": "new_game",
                "deck": {"ydk": human_text},
                "bot_deck": {"ydk": bot_text},
                "first": "me",
                "mode": "product",
            }
        )
        _wait_prompt(ws)
        ws.send_json({"t": "hint"})
        err = json.loads(ws.receive_text())
        assert err["t"] == "error" and err["text"] == "Hints are only in practice mode"


# --- 11. what-if from the review (step 4) ---


def test_whatif_before_review_errors(tmp_path, engine):
    forward = _tiny_forward()
    launcher = _tiny_review_launcher(forward)
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: launcher)
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        _new_game(ws, human_text, bot_text, mode="product", first="me")
        rng = random.Random(0)
        last = None
        n_human = 0
        for _ in range(2000):
            m = json.loads(ws.receive_text())
            if m.get("t") in ("review_progress", "review", "review_frame"):
                continue
            assert m.get("t") == "state"
            if m["result"] is not None:
                last = m
                break
            if m["prompt"] is not None:
                ws.send_json(
                    {
                        "t": "choose",
                        "prompt": m["prompt"]["id"],
                        "index": rng.randrange(len(m["prompt"]["options"])),
                    }
                )
                n_human += 1
                if n_human >= 4:
                    ws.send_json({"t": "surrender"})
                    for _ in range(50):
                        m2 = json.loads(ws.receive_text())
                        if m2.get("t") == "state" and m2.get("result") is not None:
                            last = m2
                            break
                    break
        assert last is not None and last["result"] is not None
        ws.send_json({"t": "whatif", "k": 0, "option": 0})
        err = json.loads(ws.receive_text())
        assert err["t"] == "error" and err["text"] == "no review yet (send review first)"


def test_whatif_after_review_replays_prefix(tmp_path, engine):
    """What-if replays the prefix (board at k matches), then continues to a prompt
    as a new practice game carrying the what_if record field."""
    from ygo_zero.analysis.review import collect_human_decisions
    from ygo_zero.engine.env import PyEnv
    from ygo_zero.web.view import build_view

    forward = _tiny_forward()
    launcher = _tiny_review_launcher(forward)
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: launcher)
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        _new_game(ws, human_text, bot_text, mode="product", first="me")
        rng = random.Random(0)
        last = None
        n_human = 0
        for _ in range(2000):
            m = json.loads(ws.receive_text())
            if m.get("t") in ("review_progress", "review", "review_frame"):
                continue
            assert m.get("t") == "state"
            if m["result"] is not None:
                last = m
                break
            if m["prompt"] is not None:
                ws.send_json(
                    {
                        "t": "choose",
                        "prompt": m["prompt"]["id"],
                        "index": rng.randrange(len(m["prompt"]["options"])),
                    }
                )
                n_human += 1
                if n_human >= 4:
                    ws.send_json({"t": "surrender"})
                    for _ in range(50):
                        m2 = json.loads(ws.receive_text())
                        if m2.get("t") == "state" and m2.get("result") is not None:
                            last = m2
                            break
                    break
        assert last is not None and last["result"] is not None
        ws.send_json({"t": "review"})
        review = None
        for _ in range(600):
            m = json.loads(ws.receive_text())
            if m.get("t") == "review_progress":
                continue
            if m.get("t") == "review":
                review = m
                break
        assert review is not None and review["decisions"]
        recs = [
            json.loads((p / "game.jsonl").read_text())
            for p in sorted((tmp_path / "runs").iterdir())
            if p.is_dir()
        ]
        assert len(recs) == 1
        rec = recs[0]
        from ygo_zero.decks import parse_ydk

        human, bot = parse_ydk(human_text), parse_ydk(bot_text)
        cfg = (
            DuelConfig(human, bot, seed=rec["seed"])
            if rec["human_seat"] == 0
            else DuelConfig(bot, human, seed=rec["seed"])
        )
        dlist = collect_human_decisions(engine, cfg, rec)
        assert dlist, "no human decisions to branch from"
        k = 0
        prefix = [int(c) for c in dlist[k]["prefix"]]
        env0 = PyEnv(engine, cfg, auto_single=False)
        env0.reset()
        for c in prefix:
            env0.step(int(c))
        n_opts = len(env0.opts)
        assert n_opts > 1
        alt = (int(rec["choices"][len(prefix)]) + 1) % n_opts
        # The boards after the shared prefix match (host replay vs the same
        # replay the what-if game starts from).
        want_view = build_view(env0, int(rec["human_seat"]))
        ws.send_json({"t": "whatif", "k": k, "option": alt})
        branched = None
        for _ in range(3000):
            m = json.loads(ws.receive_text())
            if m.get("t") in ("review_progress", "review", "review_frame"):
                continue
            assert m.get("t") == "state", m.get("t")
            if m.get("what_if") is not None:
                branched = m
                if m["prompt"] is not None:
                    break
            elif branched is not None and m["prompt"] is not None:
                branched = m
                break
        assert branched is not None, "no what-if game started"
        assert branched.get("practice") is True
        assert branched.get("what_if", {}).get("k") == k
        assert branched.get("what_if", {}).get("option") == alt
        assert branched["prompt"] is not None, "the what-if game did not reach a prompt"
        # Prefix equality at the mechanism level: the same host replay the
        # server runs reaches the same board (build_view compares equal).
        env1 = PyEnv(engine, cfg, auto_single=False)
        env1.reset()
        for c in prefix:
            env1.step(int(c))
        assert build_view(env1, int(rec["human_seat"])) == want_view
        # The branch is a new practice game on its own: surrender writes a record.
        ws.send_json({"t": "surrender"})
        end = None
        for _ in range(50):
            m = json.loads(ws.receive_text())
            if m.get("t") == "state" and m.get("result") is not None:
                end = m
                break
        assert end is not None and end["result"] is not None
    all_recs = [
        json.loads((p / "game.jsonl").read_text())
        for p in sorted((tmp_path / "runs").iterdir())
        if p.is_dir()
    ]
    assert len(all_recs) == 2
    new = all_recs[-1]
    assert new["practice"] is True
    assert new["what_if"]["k"] == k and new["what_if"]["option"] == alt
    assert new["seed"] == rec["seed"]


# --- 12. undo in practice mode (step 5, item 9) ---


def _prompt_sig(m: dict) -> tuple:
    p = (m or {}).get("prompt") or {}
    opts = p.get("options") or []
    return (
        p.get("text"),
        [(o.get("label"), (o.get("card") or {}).get("code")) for o in opts],
    )


def _active_choice(m: dict) -> int:
    """A choice that keeps the turn going (not a pass / phase move), else 0."""
    idle = {"pass", "end_turn", "to_battle", "to_main2", "no", "cancel"}
    for o in m["prompt"]["options"]:
        if o.get("verb") not in idle:
            return int(o["index"])
    return 0


def test_undo_duel_mode_rejected(tmp_path, engine):
    """Undo is practice-only: a duel-mode game refuses it."""
    forward = _tiny_forward()
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: _TinyLauncher(forward))
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        _new_game(ws, human_text, bot_text, mode="product", first="me")
        _wait_prompt(ws)
        ws.send_json({"t": "undo"})
        err = json.loads(ws.receive_text())
        assert err["t"] == "error" and err["text"] == "Undo is only in practice mode"


def test_undo_before_any_decision_errors(tmp_path, engine):
    """A practice game with no human decision yet has nothing to undo."""
    forward = _tiny_forward()
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: _TinyLauncher(forward))
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        ws.send_json(
            {
                "t": "new_game",
                "deck": {"ydk": human_text},
                "bot_deck": {"ydk": bot_text},
                "first": "me",
                "mode": "product",
                "practice": True,
            }
        )
        first = _wait_prompt(ws)
        assert first.get("can_undo") is False
        ws.send_json({"t": "undo"})
        err = json.loads(ws.receive_text())
        assert err["t"] == "error" and err["text"] == "Nothing to undo"


def test_undo_rewinds_practice_game(tmp_path, engine):
    """Two human decisions; each undo re-presents the previous prompt, and the
    record carries undos=2."""
    forward = _tiny_forward()
    app = _make_app(tmp_path, engine, mode_factory=lambda mode: _TinyLauncher(forward))
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    with _client(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        ws.send_json(
            {
                "t": "new_game",
                "deck": {"ydk": human_text},
                "bot_deck": {"ydk": bot_text},
                "first": "me",
                "mode": "product",
                "practice": True,
            }
        )
        seen: list[dict] = []
        for _ in range(2):
            m = _wait_prompt(ws)
            assert m.get("practice") is True
            seen.append(m)
            ws.send_json({"t": "choose", "prompt": m["prompt"]["id"], "index": _active_choice(m)})
        third = _wait_prompt(ws)  # the prompt we will undo away from
        assert third.get("can_undo") is True

        # Undo 1: back to the 2nd decision's prompt.
        ws.send_json({"t": "undo"})
        u1 = _wait_prompt(ws)
        assert _prompt_sig(u1) == _prompt_sig(seen[1])
        assert u1["view"] == seen[1]["view"]
        assert u1.get("can_undo") is True

        # Undo 2: back to the 1st decision's prompt.
        ws.send_json({"t": "undo"})
        u2 = _wait_prompt(ws)
        assert _prompt_sig(u2) == _prompt_sig(seen[0])
        assert u2["view"] == seen[0]["view"]
        assert u2.get("can_undo") is False

        # Nothing left: a third undo is refused.
        ws.send_json({"t": "undo"})
        err = json.loads(ws.receive_text())
        assert err["t"] == "error" and err["text"] == "Nothing to undo"

        ws.send_json({"t": "surrender"})
        end = None
        for _ in range(50):
            m = json.loads(ws.receive_text())
            if m.get("t") == "state" and m.get("result") is not None:
                end = m
                break
        assert end is not None
    recs = [
        json.loads((p / "game.jsonl").read_text())
        for p in sorted((tmp_path / "runs").iterdir())
        if p.is_dir()
    ]
    assert len(recs) == 1
    assert recs[0]["practice"] is True
    assert recs[0]["undos"] == 2
