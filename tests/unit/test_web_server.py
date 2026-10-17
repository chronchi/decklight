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
import time
import random
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
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32),
           "n_effects": np.ones(n, np.int64)}
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()

    @torch.no_grad()
    def forward(obs_list):
        t = {k: torch.from_numpy(np.ascontiguousarray(np.stack([o[k] for o in obs_list])))
             for k in OBS_KEYS}
        return net(t).numpy()

    return forward


class _TinyLauncher:
    """A ProductLauncher-shaped stub holding the tiny model (no checkpoints)."""

    def __init__(self, forward) -> None:
        self.forward = forward

    def start_duel(self, engine, config, bot_seat, *, game_seed, ws_seed, human_fn=None):
        return ProductDriver(
            engine, config, bot_seat=bot_seat, base_logits=self.forward, value=None,
            search_kwargs={}, game_seed=game_seed, ws_seed=ws_seed,
            fast=True, human_fn=human_fn).start()

    def record_info(self, seed: int) -> dict:
        return {"ckpt": "tiny-test", "fast": True, "game_seed": seed, "search": "tiny"}


def _make_app(tmp_path, engine, mode_factory=None):
    def make_opponent(mode):
        if mode_factory is not None:
            return mode_factory(mode)
        return RandomBot(1234)

    return create_app(token=TOKEN, engine=engine, make_opponent=make_opponent,
                      record_root=tmp_path / "runs", pics_dir=tmp_path / "pics")


def _client(app):
    return TestClient(app)


# --- schema ---

_CARD_KEYS = {"uid": str, "code": int, "faceup": bool, "pos": (str, type(None)),
              "atk": int, "def": int, "level": int, "rank": int, "link": int,
              "counters": int, "overlays": list}
_SIDE_KEYS = {"lp": int, "deck": int, "extra_count": int, "hand": list,
              "mzone": list, "emz": list, "szone": list, "field": (dict, type(None)),
              "gy": list, "banished": list, "extra": list}


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


def _check_state(msg: dict) -> None:
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
        assert prompt["kind"] in ("idle", "battle", "chain", "card", "yesno",
                                  "effectyn", "option", "position", "place",
                                  "announce", "sort", "counter")
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
                assert o["where"]["zone"] in ("hand", "mzone", "emz", "szone",
                                              "field", "gy", "banished",
                                              "extra", "deck")
            assert o["effect"] is None or isinstance(o["effect"], str)
    assert isinstance(msg["log"], list)
    for e in msg["log"]:
        assert set(e) == {"n", "actor", "text", "card", "kind"}
        assert e["actor"] in ("you", "bot", "system") and isinstance(e["text"], str)
        if e["card"] is not None:
            _check_card(e["card"], "log.card")
    assert msg["thinking"] is None or (
        set(msg["thinking"]) == {"seconds"}
        and isinstance(msg["thinking"]["seconds"], float))
    assert msg["result"] is None or (
        set(msg["result"]) == {"winner", "reason"}
        and msg["result"]["winner"] in ("you", "bot")
        and isinstance(msg["result"]["reason"], str))


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
    return codes, json.dumps(msg)


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
            ws.send_json({"t": "choose", "prompt": msg["prompt"]["id"],
                          "index": rng.randrange(n)})


def _new_game(ws, human_text: str, bot_text: str, **kw) -> None:
    msg = {"t": "new_game", "deck": {"ydk": human_text},
           "bot_deck": {"ydk": bot_text}, "first": "me", "mode": "random"}
    msg.update(kw)
    ws.send_json(msg)


def _hidden_codes_of(env, human: int) -> set[int]:
    """Bot codes the human must not see at the env's current state."""
    bot = 1 - human
    duel = env.duel
    visible = Board(duel, human).visible_codes()
    hidden = set()
    for loc in (C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
                C.LOCATION_REMOVED, C.LOCATION_EXTRA):
        for c in duel.query_location(bot, loc):
            if c is not None and c.code and c.code not in visible:
                hidden.add(c.code)
    return hidden


def _verify_no_leak(engine, cfg, human_seat: int, choices: list[int],
                    messages: list[dict]) -> None:
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
    # activating a card from its hand) or a chain link (every chained card is
    # announced) makes the code public from then on, even while the card is
    # still physically hidden - the same rules the model's own observation
    # follows (`obs.history_entry.public`, `obs.chain_array`).
    from ygo_zero.engine.query import parse_chain

    revealed: set[int] = set()

    def note_revealed() -> None:
        for ch in parse_chain(env.duel.query_field()):
            if ch.code:
                revealed.add(ch.code)

    note_revealed()
    if 0 in needed:
        hidden_by_h[0] = _hidden_codes_of(env, human_seat) - revealed
    assert len(choices) >= max(needed)
    for h_cur in range(1, max(needed) + 1):
        env.step(int(choices[h_cur - 1]))
        entry = env.history[h_cur - 1]
        if entry.public and entry.row[2]:
            revealed.add(entry.row[2])
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
    cfg = (DuelConfig(human, bot, seed=rec["seed"]) if rec["human_seat"] == 0
           else DuelConfig(bot, human, seed=rec["seed"]))
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
        tmp_path, engine, "product",
        make_opponent=lambda mode: _TinyLauncher(forward))
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
    monkeypatch.setattr(S, "_fetch_img",
                        lambda url, timeout=5.0: calls.append(url) or JPEG)
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
        rec, _ = _replay_record(engine, tmp_path / chain / "runs", last,
                                (parse_ydk(human_text), parse_ydk(bot_text)))
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
    app = _make_app(tmp_path, engine, mode_factory=lambda m: _SlowLauncher(forward, 0.5))
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
        ws.send_json({"t": "new_game", "deck": {"ydk": "nonsense"},
                      "mode": "random"})
        assert json.loads(ws.receive_text())["t"] == "error"  # illegal deck
        ws.send_json({"t": "bogus"})
        assert json.loads(ws.receive_text())["t"] == "error"  # unknown type
