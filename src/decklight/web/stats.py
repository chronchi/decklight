"""Incremental duel statistics for the web client's Stats/Cards tabs.

`DuelStats` is fed the human-filtered engine cues (`web.cues`, viewer = the
human seat, `reveal_all=False`) plus the session's chain-window resolutions and
the latest board view. Everything it stores comes from the human's own view,
so the resulting `stats_message()` is safe to send to the client: opponent
codes only enter `opp_seen` when a cue carried them face-up/revealed
(code != 0), and your own cards only enter `you_used` on activation/summon/set.
`count` on either list is plays this duel (activations + summons, plus sets
for `you_used`), not deck copies and not sightings: a card that was only seen
(revealed, milled to the GY) keeps count 0. Cues carry no identity, so one
physical copy re-played counts again.

Response windows: `note_chain_window(had, used)` is called once per *resolved*
human chain window (`SelectChain` for the human seat). `had` is whether the
window offered at least one activatable option (an `ACTIVATE` besides `PASS`);
windows with no activatable option are not response windows at all. `used` is
whether the human activated something (auto-passes always pass). The currently
waiting window is not counted until it resolves, so `used <= total` always.

Wipes: a "wipe" key moment counts only cards that left the field because of
the opponent — destroyed by battle, or destroyed/banished/sent-to-GY/returned
while a chain link activated by the opponent (of the side that lost the cards)
was resolving. Material, summon/spsummon and cost moves (outside
chain_solving…chain_solved) never count. The threshold stays at >= 3 cards
lost in one turn.
"""

from __future__ import annotations

_FIELD_LOCS = ("mzone", "szone", "field", "emz")
_SS_HOWS = ("normal", "tribute", "flip")

# Reasons that can only be opponent-caused losses when the cause matches
# (battle or an opponent's resolving link). Everything else (material,
# summon/spsummon arrivals, sets, piles bookkeeping) never counts.
_WIPE_REASONS = ("destroyed", "banished", "sent", "returned")
_NO_WIPE_REASONS = ("material", "summon", "spsummon")


def _is_ss(how: str) -> bool:
    return (how or "") not in _SS_HOWS


class DuelStats:
    """Accumulate duel stats cue by cue (one instance per game)."""

    def __init__(self, viewer: int, lp0: int = 8000) -> None:
        self.viewer = viewer
        self.lp0 = lp0
        self.reset()

    def reset(self) -> None:
        self.turn = 0
        self.lp = {"me": self.lp0, "opp": self.lp0}
        self.dealt = {"me": 0, "opp": 0}  # damage inflicted BY each side
        self.summons = {"me": 0, "opp": 0}
        self.summons_ss = {"me": 0, "opp": 0}
        self.activations = {"me": 0, "opp": 0}
        self.negated = {"me": 0, "opp": 0}  # cards of the other side this side negated
        self.were_negated = {"me": 0, "opp": 0}
        self.windows_total = 0
        self.windows_used = 0
        self.win_hist: list[list] = []  # [turn, p] per human decision (only when the switch is on)
        self.chart_labels: list[str] = []
        self.chart_me: list[int] = []
        self.chart_opp: list[int] = []
        self.moments: list[dict] = []  # chronological; stats_message reverses
        self._wipe_moment: dict[tuple[int, str], dict] = {}
        self._wipe_seq: int = 0
        self.field_lost: dict[tuple[int, str], int] = {}
        self.opp_seen: dict[int, dict] = {}
        self.you_used: dict[int, dict] = {}
        self._links: dict[int, dict] = {}
        self._solving: int = 0
        self._last: dict | None = None
        self._last_attack: dict | None = None  # {n, card}
        self._view: dict | None = None
        self._in_resolve: bool = False
        self._resolving_owner: str | None = None

    # --- feeding ---

    def feed_cues(self, cues: list[dict]) -> None:
        for c in cues or []:
            t = c.get("t")
            if t == "turn":
                self.turn = int(c.get("turn") or 0)
                self.chart_labels.append(f"T{self.turn}")
                self.chart_me.append(self.lp["me"])
                self.chart_opp.append(self.lp["opp"])
            elif t in ("damage", "recover", "pay", "lp"):
                self._feed_lp(c)
            elif t == "summon":
                self._feed_summon(c)
            elif t == "activate":
                self._feed_activate(c)
            elif t in ("chain_solving", "chain_solved", "chain_negated", "chain_disabled", "chain_end"):
                self._feed_chain(c)
            elif t in ("attack", "battle"):
                card = c.get("attacker")
                if isinstance(card, dict):
                    self._last_attack = {"n": c.get("n", -1), "card": card}
                self._note_seen(c)
            elif t == "move":
                self._feed_move(c)
            elif t in ("draw", "confirm", "target"):
                self._note_seen(c)
            # phase/shuffle/win carry no stats (win ends the duel)

    def _feed_lp(self, c: dict) -> None:
        t = c.get("t")
        if t == "damage":
            side = "me" if c.get("player") == "me" else "opp"
            amount = int(c.get("amount") or 0)
            self.lp[side] = int(c.get("lp_after", self.lp[side]))
            dealer = "opp" if side == "me" else "me"
            self.dealt[dealer] += amount
            if amount >= 1000:
                self._damage_moment(c, amount)
        elif t in ("recover", "pay"):
            side = "me" if c.get("player") == "me" else "opp"
            self.lp[side] = int(c.get("lp_after", self.lp[side]))
        elif t == "lp":
            side = "me" if c.get("player") == "me" else "opp"
            self.lp[side] = int(c.get("lp", self.lp[side]))

    def _feed_summon(self, c: dict) -> None:
        card = c.get("card") or {}
        owner = card.get("owner")
        if owner not in ("me", "opp"):
            return
        self.summons[owner] += 1
        if _is_ss(str(c.get("how") or "")):
            self.summons_ss[owner] += 1
        code = int(card.get("code") or 0)
        if code:
            if owner == "opp":
                self._add_play(self.opp_seen, code, card.get("name") or f"#{code}")
            else:
                self._add_play(self.you_used, code, card.get("name") or f"#{code}")

    def _feed_activate(self, c: dict) -> None:
        card = c.get("card") or {}
        owner = card.get("owner")
        if owner in ("me", "opp"):
            self.activations[owner] += 1
        try:
            link = int(c.get("link") or 0)
        except (TypeError, ValueError):
            link = 0
        if link:
            self._links[link] = card if isinstance(card, dict) else {}
        code = int(card.get("code") or 0)
        if code and isinstance(card, dict):
            if owner == "opp":
                self._add_play(self.opp_seen, code, card.get("name") or f"#{code}")
            elif owner == "me":
                self._add_play(self.you_used, code, card.get("name") or f"#{code}")

    def _feed_chain(self, c: dict) -> None:
        t = c.get("t")
        if t == "chain_solving":
            try:
                self._solving = int(c.get("link") or 0)
            except (TypeError, ValueError):
                self._solving = 0
            owner = (self._links.get(self._solving) or {}).get("owner")
            self._resolving_owner = owner if owner in ("me", "opp") else None
            self._in_resolve = True
        elif t == "chain_solved":
            try:
                link = int(c.get("link") or 0)
            except (TypeError, ValueError):
                link = 0
            self._last = {"link": link, "card": self._links.get(link)}
            self._solving = 0
            self._in_resolve = False
            self._resolving_owner = None
        elif t in ("chain_negated", "chain_disabled"):
            try:
                link = int(c.get("link") or 0)
            except (TypeError, ValueError):
                link = 0
            card = self._links.get(link) or {}
            by = None
            if self._solving and self._solving != link:
                by = self._links.get(self._solving)
            elif self._last and self._last.get("link") == link + 1:
                by = self._last.get("card")
            if by is None:
                higher = sorted(k for k in self._links if k > link)
                if higher:
                    by = self._links.get(higher[0])
            if not isinstance(by, dict):
                by = None
            self._negation_moment(link, card, by, disabled=(t == "chain_disabled"))
            self._solving = 0
            self._in_resolve = False
            self._resolving_owner = None
        elif t == "chain_end":
            self._links.clear()
            self._solving = 0
            self._last = None
            self._in_resolve = False
            self._resolving_owner = None

    def _feed_move(self, c: dict) -> None:
        card = c.get("card") or {}
        frm = c.get("from") or {}
        reason = str(c.get("reason") or "")
        # Opponent-caused field losses feed the wipe moments. Costs happen
        # outside chain_solving...chain_solved, material/summon moves are never
        # opponent damage, and battle destroys need a recent attack.
        if isinstance(frm, dict) and frm.get("loc") in _FIELD_LOCS:
            side = frm.get("side")
            if (
                side in ("me", "opp")
                and reason not in _NO_WIPE_REASONS
                and reason in _WIPE_REASONS
                and self._opponent_caused(side, c, reason)
            ):
                key = (self.turn or 1, side)
                self.field_lost[key] = self.field_lost.get(key, 0) + 1
                if self.field_lost[key] >= 3:
                    self._wipe(key, self.field_lost[key])
        # sets count as "used" for your own cards
        if reason == "set" and isinstance(card, dict) and card.get("owner") == "me":
            code = int(card.get("code") or 0)
            if code:
                self._add_play(self.you_used, code, card.get("name") or f"#{code}")
        self._note_seen(c)

    def _opponent_caused(self, side: str, c: dict, reason: str) -> bool:
        """Whether a field departure was caused by `side`'s opponent."""
        foe = "opp" if side == "me" else "me"
        if self._in_resolve:
            # Inside chain_solving...chain_solved: the resolving link's
            # activator is the cause. Costs already happened (outside), so
            # anything here is the effect itself.
            return self._resolving_owner == foe
        # Outside a resolving chain: only battle destroys count, and only
        # shortly after an attack by the other side.
        if reason != "destroyed":
            return False
        atk = self._last_attack
        if not isinstance(atk, dict):
            return False
        card = atk.get("card")
        if not isinstance(card, dict) or card.get("owner") != foe:
            return False
        try:
            return int(c.get("n", 0)) - int(atk.get("n", -(10**9))) <= 12
        except (TypeError, ValueError):
            return False

    def _note_seen(self, c: dict) -> None:
        """Any opponent code a cue carries face-up is public: remember it.

        Sightings alone keep count 0 (only activations/summons are plays).
        """
        stack = [c]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if "code" in cur and "owner" in cur:
                    try:
                        code = int(cur.get("code") or 0)
                    except (TypeError, ValueError):
                        code = 0
                    if code and cur.get("owner") == "opp":
                        self._ensure_seen(self.opp_seen, code, cur.get("name") or f"#{code}")
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)

    @staticmethod
    def _ensure_seen(bucket: dict[int, dict], code: int, name: str) -> None:
        ent = bucket.get(code)
        if ent is None:
            bucket[code] = {"code": code, "name": name, "count": 0}
        elif not ent.get("name") and name:
            ent["name"] = name

    @staticmethod
    def _add_play(bucket: dict[int, dict], code: int, name: str) -> None:
        ent = bucket.get(code)
        if ent is None:
            bucket[code] = {"code": code, "name": name, "count": 1}
        else:
            ent["count"] += 1
            if not ent.get("name") and name:
                ent["name"] = name

    def note_chain_window(self, had_activatable: bool, used: bool) -> None:
        """One resolved human chain window (see module docstring)."""
        if not had_activatable:
            return
        self.windows_total += 1
        if used:
            self.windows_used += 1

    def note_winchance(self, turn: int, p: float) -> None:
        """One calibrated win chance at a human decision (the server calls this
        only while the client's win-chance switch is on)."""
        self.win_hist.append([int(turn), float(p)])

    def _new_moment(self, turn: int, kind: str, text: str, sub: str) -> dict:
        """A key moment stamped with the win chance then known, so the message
        can carry "you were 61%, then 38%" once a later sample arrives."""
        return {"turn": turn, "kind": kind, "text": text, "sub": sub,
                "_w0": self.win_hist[-1][1] if self.win_hist else None,
                "_widx": len(self.win_hist)}

    def note_view(self, view: dict | None) -> None:
        self._view = view

    # --- moments ---

    def _negation_moment(self, link: int, card: dict, by: dict | None, *, disabled: bool) -> None:
        name = (card or {}).get("name") or ""
        owner = (card or {}).get("owner")
        if owner == "me":
            text = f"Your {name} was negated" if name else "Your card was negated"
            self.were_negated["me"] += 1
            by_owner = (by or {}).get("owner") if by else None
            if by_owner == "me":
                self.negated["me"] += 1
            else:
                self.negated["opp"] += 1
        elif owner == "opp":
            text = f"Bot's {name} was negated" if name else "Bot's card was negated"
            self.were_negated["opp"] += 1
            by_owner = (by or {}).get("owner") if by else None
            if by_owner == "me":
                self.negated["me"] += 1
            else:
                self.negated["opp"] += 1
        else:
            text = f"{name or 'A card'} was negated"
        by_name = (by or {}).get("name") if by else ""
        sub = f"by {by_name}" if by_name else ("effect negated" if disabled else "activation negated")
        self.moments.append(self._new_moment(self.turn or 1, "negation", text, sub))

    def _damage_moment(self, c: dict, amount: int) -> None:
        atk = None
        if self._last_attack is not None:
            try:
                if int(c.get("n", 0)) - int(self._last_attack.get("n", -(10**9))) <= 6:
                    atk = self._last_attack.get("card")
            except (TypeError, ValueError):
                atk = None
        if isinstance(atk, dict) and atk.get("code"):
            nm = atk.get("name") or "a monster"
            owner = atk.get("owner")
            text = f"Your {nm} attacked" if owner == "me" else f"Bot attacked with {nm}"
            sub = f"{amount} damage"
        else:
            side = c.get("player")
            if side == "me":
                text = f"You took {amount} damage"
            else:
                text = f"Bot took {amount} damage"
            sub = ""
        self.moments.append(self._new_moment(self.turn or 1, "damage", text, sub))

    def _wipe(self, key: tuple[int, str], n: int) -> None:
        turn, side = key
        ent = self._wipe_moment.get(key)
        text = f"You lost {n} cards from the field" if side == "me" else f"Bot lost {n} cards from the field"
        if ent is None:
            ent = self._new_moment(turn, "wipe", text, "")
            ent["_seq"] = self._wipe_seq
            self._wipe_seq += 1
            self._wipe_moment[key] = ent
            self.moments.append(ent)
        else:
            # A later loss in the same turn moves the wipe to the latest
            # position, so reversed(moments) stays newest-first inside a turn.
            ent["text"] = text
            try:
                self.moments.remove(ent)
            except ValueError:
                pass
            self.moments.append(ent)

    # --- message ---

    @staticmethod
    def _count(cards: list | None) -> int:
        return len(cards or [])

    @staticmethod
    def _field_count(side: dict) -> int:
        n = 0
        for zone in ("mzone", "emz", "szone"):
            n += sum(1 for c in (side.get(zone) or []) if c is not None)
        if side.get("field") is not None:
            n += 1
        return n

    def _ledger_side(self, view: dict | None, who: str) -> dict:
        key = "me" if who == "you" else "opp"
        if not view or who not in view:
            hand = field = gy = ban = lpv = 0
        else:
            side = view[who]
            hand = self._count(side.get("hand"))
            field = self._field_count(side)
            gy = self._count(side.get("gy"))
            ban = self._count(side.get("banished"))
            lpv = int(side.get("lp", self.lp[key]))
        total = hand + field
        return {
            "lp": lpv,
            "hand": hand,
            "field": field,
            "total": total,
            "gy": gy,
            "banished": ban,
            "summons": self.summons[key],
            "summons_ss": self.summons_ss[key],
            "activations": self.activations[key],
            "negated": self.negated[key],
            "were_negated": self.were_negated[key],
            "damage": self.dealt[key],
        }

    def stats_message(self, *, winchance_on: bool = True) -> dict:
        you = self._ledger_side(self._view, "you")
        bot = self._ledger_side(self._view, "bot")
        # live LP wins over the cue tally (identical unless a cue was missed)
        you["lp"] = int(you["lp"])
        bot["lp"] = int(bot["lp"])
        diff = you["total"] - bot["total"]
        you["advantage"], bot["advantage"] = diff, -diff
        labels = list(self.chart_labels)
        cyou = list(self.chart_me)
        cbot = list(self.chart_opp)
        if not labels:
            labels, cyou, cbot = ["now"], [you["lp"]], [bot["lp"]]
        else:
            labels = [*labels, "now"]
            cyou = [*cyou, you["lp"]]
            cbot = [*cbot, bot["lp"]]
        # Newest first; inside one turn the latest event is first (moments
        # are appended chronologically, re-wipes move to the end).
        moments = []
        show_wc = bool(winchance_on) and bool(self.win_hist)
        for raw in list(reversed(self.moments))[:5]:
            m = {k: v for k, v in raw.items() if not k.startswith("_")}
            if show_wc:
                w0, widx = raw.get("_w0"), raw.get("_widx")
                if w0 is not None and widx is not None and widx < len(self.win_hist):
                    w1 = self.win_hist[widx][1]
                    span = f"you were {round(w0 * 100)}%, then {round(w1 * 100)}%"
                    m["sub"] = f"{m['sub']} · {span}" if m.get("sub") else span
            moments.append(m)
        opp_seen = sorted(self.opp_seen.values(), key=lambda e: str(e.get("name") or ""))
        you_used = sorted(self.you_used.values(), key=lambda e: str(e.get("name") or ""))
        out = {
            "turn": self.turn,
            "lp_you": you["lp"],
            "lp_bot": bot["lp"],
            "dealt_you": self.dealt["me"],
            "dealt_bot": self.dealt["opp"],
            "adv_you": you["total"],
            "adv_bot": bot["total"],
            "adv_diff": diff,
            "windows_used": self.windows_used,
            "windows_total": self.windows_total,
            "neg_you": self.negated["me"],
            "neg_were_you": self.were_negated["me"],
            "neg_bot": self.negated["opp"],
            "neg_were_bot": self.were_negated["opp"],
            "chart_labels": labels,
            "chart_you": cyou,
            "chart_bot": cbot,
            "moments": moments,
            "ledger": {"you": you, "bot": bot},
            "opp_seen": opp_seen,
            "you_used": you_used,
        }
        if self.win_hist and winchance_on:
            # Only while the switch is on: history for the chart line + key moments.
            out["winchance"] = {"p": self.win_hist[-1][1], "turn": self.win_hist[-1][0],
                                "history": [list(e) for e in self.win_hist]}
        return out
