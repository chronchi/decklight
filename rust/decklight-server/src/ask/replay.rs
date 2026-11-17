//! `replay` and `bot_view` (config/ask/tools.v1.json, plan P2): the user's saved games, read back from the match
//! records (ADR 0010) by replaying their choices through the engine. Everything the model sees is the user's own
//! view (the opponent's hand and set cards stay hidden, as in the review), so an answer never leaks what the user
//! could not have known. `bot_view` is the practice hint on a past decision: the policy's top picks and the
//! one-step-lookahead win chance of each, plus the user's own pick.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};
use ygoenv_sys::{DuelConfig, Env, Obs};

use crate::product::WcMap;
use crate::server::{App, estimate_candidates, masked_softmax};

/// The sample games shipped with the app (under the app root): listed and replayable, never in the user's index, so
/// they never count in `my_stats` (user 2026-11-15).
pub const SAMPLES_DIR: &str = "config/samples/games";
/// The sample player's id in a sample record's seats.
pub const SAMPLE_PLAYER: &str = "sample";

/// Games listed when no record is given.
const LIST_DEFAULT: u64 = 10;
/// Event lines per turn in the whole-game summary (the `turn` argument shows them all).
const EVENTS_PER_TURN: usize = 12;
/// Options shown per decision.
const OPTIONS_CAP: usize = 12;

fn text_arg(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(Some(s.trim().to_string())),
        Some(v) => Err(format!("`{key}` must be a game id from the game list (or \"last\"), got {v}")),
    }
}

fn whole_arg(args: &Value, key: &str) -> Result<Option<u64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => super::host::whole(v).filter(|&n| n >= 1).map(Some).ok_or(format!("`{key}` must be a whole number from 1, got {v}")),
    }
}

fn pct(p: f64) -> f64 {
    (p * 1000.0).round() / 10.0
}

fn index_db(index: &Path) -> Result<Option<rusqlite::Connection>, String> {
    if !index.is_file() {
        return Ok(None);
    }
    rusqlite::Connection::open_with_flags(index, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map(Some)
        .map_err(|e| format!("cannot open the game history: {e}"))
}

fn result_word(won: i64) -> &'static str {
    match won {
        1 => "you won",
        0 => "you lost",
        _ => "not decided",
    }
}

/// The user's latest games, newest first (only games with a seat of ours, like `my_stats`).
pub fn games(index: &Path, limit: u64) -> Result<Value, String> {
    let Some(con) = index_db(index)? else {
        return Ok(json!({"games": [], "note": "no saved games yet: play a game first"}));
    };
    let mut st = con
        .prepare(
            "SELECT id, created, mode, my_deck, opp_deck, opp_type, first, me, won, turns FROM matches \
             WHERE me >= 0 ORDER BY created DESC, rowid DESC LIMIT ?1",
        )
        .map_err(|e| format!("cannot read the game history: {e}"))?;
    let rows = st
        .query_map([limit as i64], |r| {
            let (first, me, won): (i64, i64, i64) = (r.get(6)?, r.get(7)?, r.get(8)?);
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "when": r.get::<_, String>(1)?,
                "mode": r.get::<_, String>(2)?,
                "your_deck": r.get::<_, String>(3)?,
                "opponent_deck": r.get::<_, String>(4)?,
                "opponent": r.get::<_, String>(5)?,
                "you_went": if first < 0 { "unknown" } else if first == me { "first" } else { "second" },
                "result": result_word(won),
                "turns": r.get::<_, i64>(9)?,
            }))
        })
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map_err(|e| format!("cannot read the game history: {e}"))?;
    Ok(json!({"games": rows}))
}

/// The sample games' headers (the same fields as the game list, plus `sample`).
pub fn samples(app: &App) -> Vec<Value> {
    let dir = app.root.join(SAMPLES_DIR);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".match.json.gz"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        match decklight_record::load_gz(&f) {
            Ok(rec) => {
                let me = sample_seat(&rec);
                let decks = rec["decks"].as_array().cloned().unwrap_or_default();
                let name = |s: i64| decks.get(s as usize).and_then(|d| d["name"].as_str()).unwrap_or("").to_string();
                let winner = rec["result"]["winner"].as_i64().unwrap_or(-1);
                out.push(json!({
                    "id": rec["id"],
                    "sample": true,
                    "your_deck": name(me),
                    "opponent_deck": name(1 - me),
                    "you_went": if rec["first"].as_i64() == Some(me) { "first" } else { "second" },
                    "result": result_word(if winner < 0 { -1 } else { (winner == me) as i64 }),
                    "turns": rec["result"]["turns"],
                }));
            }
            Err(e) => eprintln!("ALERT ask: sample game {} unreadable: {e}", f.display()),
        }
    }
    out
}

/// The sample player's seat in a sample record (seat 0 when unmarked).
fn sample_seat(rec: &Value) -> i64 {
    rec["seats"].as_array().and_then(|s| s.iter().position(|x| x["player"] == SAMPLE_PLAYER)).unwrap_or(0) as i64
}

/// The record file of game `id` ("last" = the newest game with a seat of ours; "sample-N" = a sample game).
fn record_path(app: &App, id: &str) -> Result<PathBuf, String> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(format!("{id:?} is not a game id; call replay without a record to list the games"));
    }
    if id.starts_with("sample-") {
        let p = app.root.join(SAMPLES_DIR).join(format!("{id}.match.json.gz"));
        return if p.is_file() { Ok(p) } else { Err(format!("no sample game {id:?}; call replay without a record to list the games")) };
    }
    let root = &app.records.root;
    let con = index_db(&root.join("index.sqlite"))?;
    let row: Option<(String, String)> = match &con {
        None => None,
        Some(con) => {
            let q = if id == "last" {
                "SELECT id, path FROM matches WHERE me >= 0 ORDER BY created DESC, rowid DESC LIMIT 1"
            } else {
                "SELECT id, path FROM matches WHERE id = ?1"
            };
            let mut st = con.prepare(q).map_err(|e| format!("cannot read the game history: {e}"))?;
            let mut rows = if id == "last" { st.query([]) } else { st.query([id]) }
                .map_err(|e| format!("cannot read the game history: {e}"))?;
            match rows.next().map_err(|e| format!("cannot read the game history: {e}"))? {
                Some(r) => Some((r.get(0).unwrap_or_default(), r.get(1).unwrap_or_default())),
                None => None,
            }
        }
    };
    let (id, indexed) = match row {
        Some(r) => r,
        None if id == "last" => {
            let s: Vec<String> = samples(app).iter().filter_map(|g| g["id"].as_str().map(str::to_string)).collect();
            let also = if s.is_empty() { String::new() } else { format!(" (sample games show how it works: {})", s.join(", ")) };
            return Err(format!("no saved games yet: play a game first{also}"));
        }
        None => (id.to_string(), String::new()),
    };
    let own = root.join(format!("{id}.match.json.gz"));
    let indexed = PathBuf::from(indexed);
    if !indexed.as_os_str().is_empty() && indexed.is_file() {
        Ok(indexed)
    } else if own.is_file() {
        Ok(own)
    } else {
        Err(format!("no saved game {id:?}; call replay without a record to list the games"))
    }
}

/// One of the user's decisions with more than one option (numbered from 1, in game order).
pub(super) struct Decision {
    /// Choices made before it (the replay prefix).
    prefix: usize,
    pub(super) turn: i32,
    pub(super) phase: &'static str,
    pub(super) kind: String,
    /// The prompt's question as the duel screen shows it.
    pub(super) text: String,
    /// (option index, label, card code or 0), in the prompt's order.
    pub(super) options: Vec<(i64, String, i64)>,
    pub(super) picked: String,
}

pub(super) struct TurnLog {
    pub(super) turn: i32,
    pub(super) yours: bool,
    /// Life points (you, opponent) when the turn started.
    pub(super) lp: (i64, i64),
    pub(super) events: Vec<String>,
}

/// Choice kinds left out of the log: chain passes, zones, positions and the steps of a card selection say nothing
/// about the game (a 40-turn game had more of them than plays).
const QUIET: [&str; 8] = ["pass", "place", "position", "select", "unselect", "finish", "cancel", "no"];

fn lps(app: &App, env: &Env<'_>, me: i32) -> (i64, i64) {
    let v = app.ctx.build_view(env, me, false);
    (v["you"]["lp"].as_i64().unwrap_or(0), v["bot"]["lp"].as_i64().unwrap_or(0))
}

/// A saved game, replayed once from its record.
pub(super) struct Game {
    pub(super) id: String,
    pub(super) rec: Value,
    pub(super) me: i32,
    cfg: DuelConfig,
    choices: Vec<i32>,
    pub(super) decisions: Vec<Decision>,
    pub(super) turns: Vec<TurnLog>,
    /// Life points (you, opponent) at the end.
    pub(super) final_lp: (i64, i64),
}

fn event_line(e: &Value) -> String {
    let who = if e["actor"] == "you" { "You" } else { "Opponent" };
    format!("{who}: {}", e["text"].as_str().unwrap_or(""))
}

pub(super) fn open(app: &App, id: &str, cancel: &AtomicBool) -> Result<Game, String> {
    let path = record_path(app, id)?;
    let rec = decklight_record::load_gz(&path).map_err(|e| format!("cannot read the saved game: {e}"))?;
    let sample = rec["sample"] == true;
    let me = if sample { sample_seat(&rec) } else { app.records.my_seat(&rec) };
    if me < 0 {
        return Err("that game has no seat of yours (bot against bot); pick one of your games from the list".into());
    }
    let me = me as i32;
    let cfg = decklight_edopro::record::config_of(&rec).map_err(|e| format!("this game cannot be replayed: {e}"))?;
    let mut choices = Vec::new();
    for (i, r) in rec["responses"].as_array().map(Vec::as_slice).unwrap_or_default().iter().enumerate() {
        let opt = r["opt"].as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
            format!("this game was saved without the app's choices (response {i}); only games played in the app replay")
        })?;
        choices.extend(opt.iter().map(|v| v.as_i64().unwrap_or(-1) as i32));
    }
    let mut env = Env::new(app.engine, &cfg, false).map_err(|e| format!("could not start the replay: {e}"))?;
    let mut obs = Obs::zeroed();
    env.reset(&mut obs).map_err(|e| format!("could not start the replay: {e}"))?;
    let mut decisions = Vec::new();
    let mut turns: Vec<TurnLog> = Vec::new();
    let mut seen = 0;
    let flush = |env: &Env<'_>, turns: &mut Vec<TurnLog>, seen: &mut usize| -> Result<(), String> {
        let st = env.duel_state()?;
        let events = app.ctx.build_events(env, me, *seen);
        *seen += events.len();
        let lines = events.iter().filter(|e| !QUIET.iter().any(|q| e["kind"] == *q)).map(event_line);
        if turns.last().is_none_or(|t| t.turn != st.turn) {
            turns.push(TurnLog { turn: st.turn, yours: st.turn_player == me, lp: lps(app, env, me), events: Vec::new() });
        }
        turns.last_mut().expect("pushed above").events.extend(lines);
        Ok(())
    };
    for (i, &ch) in choices.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        if env.done() {
            return Err(format!("the saved game ends early (choice {i} of {}): it cannot be replayed", choices.len()));
        }
        flush(&env, &mut turns, &mut seen)?;
        if env.player() == me && env.num_options() > 1 {
            let st = env.duel_state()?;
            let prompt = app.ctx.build_prompt(&env, me, decisions.len() as i64 + 1);
            let opts: Vec<(i64, String, i64)> = prompt
                .as_ref()
                .and_then(|p| p["options"].as_array().cloned())
                .unwrap_or_default()
                .iter()
                .map(|o| {
                    let code = o["card"]["code"].as_i64().unwrap_or(0);
                    (o["index"].as_i64().unwrap_or(-1), o["label"].as_str().unwrap_or("").to_string(), code)
                })
                .collect();
            let picked = opts
                .iter()
                .find(|(j, _, _)| *j == ch as i64)
                .map(|(_, l, _)| l.clone())
                .unwrap_or_else(|| app.ctx.fallback_label(&env, me, ch.max(0) as usize));
            decisions.push(Decision {
                prefix: i,
                turn: st.turn,
                phase: decklight_view::phase_name(st.phase).unwrap_or("main1"),
                kind: prompt.as_ref().and_then(|p| p["kind"].as_str()).unwrap_or("card").to_string(),
                text: prompt.as_ref().and_then(|p| p["text"].as_str()).unwrap_or("").to_string(),
                options: opts,
                picked,
            });
        }
        env.step(ch, &mut obs).map_err(|e| format!("the saved game does not replay (choice {i}): {e}"))?;
    }
    flush(&env, &mut turns, &mut seen)?;
    let final_lp = lps(app, &env, me);
    Ok(Game { id: rec["id"].as_str().unwrap_or(id).to_string(), rec, me, cfg, choices, decisions, turns, final_lp })
}

impl Game {
    pub(super) fn header(&self) -> Value {
        let decks = self.rec["decks"].as_array().cloned().unwrap_or_default();
        let name = |s: i32| decks.get(s as usize).and_then(|d| d["name"].as_str()).unwrap_or("unnamed deck").to_string();
        let opp = 1 - self.me;
        let first = self.rec["first"].as_i64();
        let res = &self.rec["result"];
        let winner = res["winner"].as_i64();
        let result = match winner {
            Some(w) if w == self.me as i64 => "you won",
            Some(w) if w == opp as i64 => "you lost",
            _ => "not decided",
        };
        let how = if res["surrendered"].as_bool() == Some(true) {
            Some("surrender")
        } else if res["disconnected"].as_bool() == Some(true) {
            Some("disconnect")
        } else {
            None
        };
        let mut out = json!({
            "id": self.id,
            "when": self.rec["created"],
            "mode": self.rec["mode"],
            "your_deck": name(self.me),
            "opponent_deck": name(opp),
            "opponent": self.rec["seats"][opp as usize]["type"],
            "you_went": match first { Some(f) if f == self.me as i64 => "first", Some(_) => "second", None => "unknown" },
            "result": result,
            "ended_by": how,
            "turns": res["turns"],
            "final_lp": {"you": self.final_lp.0, "opponent": self.final_lp.1},
            "your_decisions": self.decisions.len(),
        });
        if self.rec["sample"] == true {
            out["sample"] = json!(true);
            out["note"] = json!("a sample game shipped with the app, not one of the user's games");
        }
        out
    }

    fn decision_numbers(&self, turn: i32) -> Option<String> {
        let ks: Vec<usize> = (0..self.decisions.len()).filter(|&k| self.decisions[k].turn == turn).map(|k| k + 1).collect();
        match ks[..] {
            [] => None,
            [a] => Some(a.to_string()),
            _ => Some(format!("{}-{}", ks[0], ks[ks.len() - 1])),
        }
    }

    pub(super) fn summary(&self) -> Value {
        let turns: Vec<Value> = self
            .turns
            .iter()
            .map(|t| {
                let n = t.events.len();
                let mut events: Vec<Value> = t.events.iter().take(EVENTS_PER_TURN).map(|s| json!(s)).collect();
                if n > EVENTS_PER_TURN {
                    events.push(json!(format!("... {} more (replay with turn {} shows all)", n - EVENTS_PER_TURN, t.turn)));
                }
                json!({
                    "turn": t.turn,
                    "player": if t.yours { "you" } else { "opponent" },
                    "lp_at_start": [t.lp.0, t.lp.1],
                    "events": events,
                    "your_decisions": self.decision_numbers(t.turn),
                })
            })
            .collect();
        let mut out = self.header();
        out["log"] = json!(turns);
        out
    }

    /// The index of the option the user picked at decision `k`.
    pub(super) fn picked_index(&self, k: usize) -> i64 {
        self.choices[self.decisions[k].prefix] as i64
    }

    /// Life points (you, opponent) at the start of each turn, then at the end.
    pub(super) fn lp_series(&self) -> Vec<[i64; 2]> {
        let mut v: Vec<[i64; 2]> = self.turns.iter().map(|t| [t.lp.0, t.lp.1]).collect();
        v.push([self.final_lp.0, self.final_lp.1]);
        v
    }

    fn decision_json(&self, k: usize) -> Value {
        let d = &self.decisions[k];
        let mut options: Vec<Value> = d.options.iter().take(OPTIONS_CAP).map(|(_, l, _)| json!(l)).collect();
        if d.options.len() > OPTIONS_CAP {
            options.push(json!(format!("... {} more", d.options.len() - OPTIONS_CAP)));
        }
        json!({
            "decision": k + 1,
            // Choices made before it: the Watch frame with this `choice` is the board at the decision.
            "choice": d.prefix,
            "turn": d.turn,
            "phase": d.phase,
            "prompt": d.kind,
            "options": options,
            "you_picked": d.picked,
        })
    }

    fn turn_detail(&self, turn: i32) -> Result<Value, String> {
        let t = self.turns.iter().find(|t| t.turn == turn).ok_or_else(|| {
            format!("the game has turns 1 to {}; turn {turn} is not one of them", self.turns.last().map_or(0, |t| t.turn))
        })?;
        let decisions: Vec<Value> =
            (0..self.decisions.len()).filter(|&k| self.decisions[k].turn == turn).map(|k| self.decision_json(k)).collect();
        Ok(json!({
            "id": self.id,
            "turn": turn,
            "player": if t.yours { "you" } else { "opponent" },
            "lp_at_start": {"you": t.lp.0, "opponent": t.lp.1},
            "events": t.events,
            "your_decisions": decisions,
        }))
    }

    fn pick(&self, decision: u64) -> Result<usize, String> {
        let n = self.decisions.len();
        if n == 0 {
            return Err("you made no decision with more than one option in this game".into());
        }
        let k = decision as usize;
        if k > n {
            return Err(format!("the game has decisions 1 to {n}; {decision} is not one of them"));
        }
        Ok(k - 1)
    }

    /// The engine at decision `k` (before the user picks), and the user's observation there.
    fn env_at(&self, app: &App, k: usize) -> Result<(Env<'static>, Box<Obs>), String> {
        let mut env = Env::new(app.engine, &self.cfg, false).map_err(|e| format!("could not start the replay: {e}"))?;
        let mut obs = Obs::zeroed();
        env.replay(&self.choices[..self.decisions[k].prefix], &mut obs)
            .map_err(|e| format!("the saved game does not replay: {e}"))?;
        Ok((env, obs))
    }

    fn board(&self, app: &App, k: usize) -> Result<Value, String> {
        let (env, _) = self.env_at(app, k)?;
        let view = app.ctx.build_view(&env, self.me, false);
        let mut out = self.decision_json(k);
        out["id"] = json!(self.id);
        out["of"] = json!(self.decisions.len());
        out["your_turn"] = json!(view["turn_player"] == "you");
        out["you"] = side(&view["you"], true);
        out["opponent"] = side(&view["bot"], false);
        let chain: Vec<Value> = view["chain"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|c| json!(format!("{}: {}", if c["controller"] == "you" { "You" } else { "Opponent" }, card_text(&c["card"]))))
            .collect();
        if !chain.is_empty() {
            out["chain"] = json!(chain);
        }
        Ok(out)
    }
}

/// One card as the user sees it: its name with stats and position, or what is visible of it.
fn card_text(c: &Value) -> String {
    let Some(name) = c["name"].as_str().filter(|_| c["code"].as_i64().unwrap_or(0) != 0) else {
        return match c["pos"].as_str() {
            Some("set") => "a set card".into(),
            Some("def") => "a face-down defense monster".into(),
            _ => "a hidden card".into(),
        };
    };
    let pos = c["pos"].as_str();
    let (atk, def, link) = (c["atk"].as_i64().unwrap_or(0), c["def"].as_i64().unwrap_or(0), c["link"].as_i64().unwrap_or(0));
    let is_monster = matches!(pos, Some("atk" | "def")) && (atk > 0 || def > 0 || link > 0 || c["level"].as_i64().unwrap_or(0) > 0);
    let mut s = name.to_string();
    if is_monster {
        s += &if link > 0 { format!(" (ATK {atk}, Link {link})") } else { format!(" (ATK {atk}/DEF {def})") };
        if pos == Some("def") {
            s += " in defense";
        }
    } else if pos == Some("set") {
        s += " (set)";
    }
    let mats = c["overlays"].as_array().map_or(0, Vec::len);
    if mats > 0 {
        s += &format!(", {mats} material{}", if mats == 1 { "" } else { "s" });
    }
    s
}

fn cards(v: &Value) -> Vec<Value> {
    v.as_array().into_iter().flatten().filter(|c| !c.is_null()).map(|c| json!(card_text(c))).collect()
}

fn side(s: &Value, mine: bool) -> Value {
    let mut monsters = cards(&s["mzone"]);
    monsters.extend(cards(&s["emz"]));
    let mut out = json!({
        "lp": s["lp"],
        "monsters": monsters,
        "spells_traps": cards(&s["szone"]),
        "graveyard": cards(&s["gy"]),
        "banished": cards(&s["banished"]),
        "deck": s["deck"],
        "extra_deck": s["extra_count"],
    });
    if !s["field"].is_null() {
        out["field_spell"] = json!(card_text(&s["field"]));
    }
    let hand = s["hand"].as_array().map_or(0, Vec::len);
    out["hand"] = if mine { json!(cards(&s["hand"])) } else { json!(format!("{hand} card{}", if hand == 1 { "" } else { "s" })) };
    out
}

/// The `replay` tool.
pub fn replay(app: &App, args: &Value, cancel: &AtomicBool) -> Result<Value, String> {
    if let Some(obj) = args.as_object() {
        let known = ["record", "turn", "decision", "limit"];
        if let Some(k) = obj.keys().find(|k| !known.contains(&k.as_str())) {
            return Err(format!("unknown argument `{k}`; replay takes record, turn, decision, limit"));
        }
    }
    let Some(id) = text_arg(args, "record")? else {
        let limit = whole_arg(args, "limit")?.unwrap_or(LIST_DEFAULT);
        if limit > 20 {
            return Err(format!("`limit` must be 1 to 20, got {limit}"));
        }
        let mut out = games(&app.records.root.join("index.sqlite"), limit)?;
        let samples = samples(app);
        if !samples.is_empty() {
            out["samples"] = json!(samples);
            out["samples_note"] = json!("sample games ship with the app to show what replay and bot_view do; they are not \
                                         the user's games and never count in my_stats");
        }
        return Ok(out);
    };
    let (turn, decision) = (whole_arg(args, "turn")?, whole_arg(args, "decision")?);
    if turn.is_some() && decision.is_some() {
        return Err("give `turn` or `decision`, not both".into());
    }
    let g = open(app, &id, cancel)?;
    match (turn, decision) {
        (Some(t), _) => g.turn_detail(t as i32),
        (_, Some(d)) => g.board(app, g.pick(d)?),
        _ => Ok(g.summary()),
    }
}

pub(super) fn calib(app: &App) -> Result<WcMap, String> {
    let path = app.root.join("src/decklight/web/winchance_calib.json");
    WcMap::load(&path).map_err(|e| format!("the win-chance calibration is missing ({e}); reinstall the app"))
}

/// The `bot_view` tool: the policy's top 3 at one of the user's decisions with their win chances, and the user's pick.
pub fn bot_view(app: &App, args: &Value, cancel: &AtomicBool) -> Result<Value, String> {
    if let Some(obj) = args.as_object() {
        if let Some(k) = obj.keys().find(|k| !["record", "decision"].contains(&k.as_str())) {
            return Err(format!("unknown argument `{k}`; bot_view takes record and decision"));
        }
    }
    let id = text_arg(args, "record")?.ok_or("`record` is required: a game id from replay's list, or \"last\"")?;
    let decision = whole_arg(args, "decision")?.ok_or("`decision` is required: a decision number from replay")?;
    let g = open(app, &id, cancel)?;
    let k = g.pick(decision)?;
    let map = calib(app)?;
    let mut nets = app.onnx_pool.take().map_err(|e| format!("bot_view needs the bot's model: {e}"))?;
    bot_view_at(app, &g, k, &map, &mut nets, cancel)
}

/// The bot's view of decision `k` (0-based) of an open game; the Games screen's review runs it on every decision.
pub(super) fn bot_view_at(
    app: &App,
    g: &Game,
    k: usize,
    map: &WcMap,
    nets: &mut decklight_onnx::OnnxNets,
    cancel: &AtomicBool,
) -> Result<Value, String> {
    let (_, obs) = g.env_at(app, k)?;
    let logits = nets.policy_logits(&[&obs])?.into_iter().next().ok_or("the bot's model gave no output")?;
    let probs = masked_softmax(&logits, &obs.action_mask);
    let mut legal: Vec<usize> = (0..obs.action_mask.len()).filter(|&j| obs.action_mask[j] != 0).collect();
    legal.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let picked = g.choices[g.decisions[k].prefix] as usize;
    let mut cands: Vec<usize> = legal.iter().copied().take(3).collect();
    if !cands.contains(&picked) {
        cands.push(picked);
    }
    let prefix = &g.choices[..g.decisions[k].prefix];
    let stop = || cancel.load(Ordering::Relaxed);
    let est = estimate_candidates(app, &g.cfg, g.me, nets, prefix, &cands, map, &stop)?;
    let turn = obs.global[2];
    let now = map.p(nets.values(&[&obs])?.first().copied().ok_or("the bot's model gave no value")? as f64, turn);
    let (env, _) = g.env_at(app, k)?;
    let prompt = app.ctx.build_prompt(&env, g.me, 1);
    let opt = |i: usize| prompt.as_ref().and_then(|p| p["options"].as_array()?.iter().find(|o| o["index"].as_u64() == Some(i as u64)).cloned());
    let label = |i: usize| {
        opt(i).and_then(|o| o["label"].as_str().map(str::to_string)).unwrap_or_else(|| app.ctx.fallback_label(&env, g.me, i))
    };
    // The card's code lets the chat block show its art; the model can ignore it.
    let row = |j: usize| {
        // `index` tells two options with the same label apart (two copies of a card in hand).
        let mut r = json!({"option": label(cands[j]), "index": cands[j], "bot_prob": pct(probs[cands[j]]), "win_chance": pct(est[j])});
        if let Some(c) = opt(cands[j]).and_then(|o| o["card"]["code"].as_i64()).filter(|&c| c != 0) {
            r["code"] = json!(c);
        }
        r
    };
    let top: Vec<Value> = (0..cands.len()).filter(|&j| j < 3 && legal.get(j) == Some(&cands[j])).map(row).collect();
    let yours = cands.iter().position(|&c| c == picked).map(row).unwrap_or(Value::Null);
    let d = &g.decisions[k];
    Ok(json!({
        "id": g.id,
        "decision": k + 1,
        "turn": d.turn,
        "phase": d.phase,
        "prompt": d.kind,
        "win_chance_now": pct(now),
        "bot_top": top,
        "you_picked": yours,
        "note": "bot_prob is the bot's policy; win_chance (percent) plays one step ahead with the bot on both sides and \
                 reads the value net, calibrated on bot games: an estimate, not a solved line",
    }))
}

/// Plays shown per turn in the chat's play sequence.
const FRAMES_CAP: usize = 24;

/// A board as the play screen draws it (web/app.js replay mode, the chat's stills and Watch): a `state` frame of the
/// user's view with both hands shown (a replay shows the opponent's cards, user 2026-11-15), the log lines since the
/// last frame and a caption.
fn screen(app: &App, env: &Env<'_>, me: i32, names: &[String; 2], log: Vec<Value>, caption: Value) -> Value {
    json!({"t": "state", "view": app.ctx.build_view(env, me, true), "prompt": null, "log": log, "cues": [],
           "thinking": null, "result": null, "caption": caption, "names": names})
}

impl Game {
    /// The board after each play of one turn (quiet choices are folded into the next play), for the chat's sequence.
    fn turn_frames(&self, app: &App, turn: i32, cancel: &AtomicBool) -> Result<Value, String> {
        let mut t = self.frames_between(app, turn, turn, cancel)?.remove(0);
        t["id"] = json!(self.id);
        t["decisions"] = json!(self.decision_numbers(turn));
        Ok(t)
    }

    /// Turns `from..=to`, each `{turn, player, frames, more}`, in one replay; each frame's `screen` is the board as the
    /// play screen draws it and `choice` is how many choices the game had made by then (Watch opens there).
    pub(super) fn frames_between(&self, app: &App, from: i32, to: i32, cancel: &AtomicBool) -> Result<Vec<Value>, String> {
        let last = self.turns.last().map_or(0, |t| t.turn);
        for turn in [from, to] {
            if !self.turns.iter().any(|t| t.turn == turn) {
                return Err(format!("the game has turns 1 to {last}; turn {turn} is not one of them"));
            }
        }
        let mut env = Env::new(app.engine, &self.cfg, false).map_err(|e| format!("could not start the replay: {e}"))?;
        let mut obs = Obs::zeroed();
        env.reset(&mut obs).map_err(|e| format!("could not start the replay: {e}"))?;
        let mut seen = 0;
        let mut out: Vec<Value> = Vec::new();
        let h = self.header();
        let names = [h["your_deck"].as_str().unwrap_or("You").to_string(), h["opponent_deck"].as_str().unwrap_or("Opponent").to_string()];
        for (i, &ch) in self.choices.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err("stopped".into());
            }
            let now = env.duel_state()?.turn;
            if now > to {
                break;
            }
            if now >= from && out.last().is_none_or(|t| t["turn"] != now) {
                seen += app.ctx.build_events(&env, self.me, seen).len();
                let yours = self.turns.iter().find(|t| t.turn == now).is_some_and(|t| t.yours);
                let who = if yours { "your" } else { "the opponent's" };
                out.push(json!({"turn": now, "player": if yours { "you" } else { "opponent" }, "more": 0, "frames": [
                    {"who": "", "text": format!("Turn {now}: {who} turn"), "choice": i,
                     "screen": screen(app, &env, self.me, &names, Vec::new(), Value::Null)}]}));
            }
            env.step(ch, &mut obs).map_err(|e| format!("the saved game does not replay (choice {i}): {e}"))?;
            let events = app.ctx.build_events(&env, self.me, seen);
            seen += events.len();
            if now < from {
                continue;
            }
            let plays: Vec<&Value> = events.iter().filter(|e| !QUIET.iter().any(|q| e["kind"] == *q)).collect();
            let Some(first) = plays.first() else { continue };
            let t = out.last_mut().expect("pushed at the turn's start");
            if t["frames"].as_array().map_or(0, Vec::len) >= FRAMES_CAP {
                t["more"] = json!(t["more"].as_u64().unwrap_or(0) + plays.len() as u64);
                continue;
            }
            let text = plays.iter().map(|e| e["text"].as_str().unwrap_or("")).collect::<Vec<_>>().join(" · ");
            let actor = if first["actor"] == "you" { &names[0] } else { &names[1] };
            let log: Vec<Value> = plays.iter().map(|e| (*e).clone()).collect();
            t["frames"].as_array_mut().expect("frames").push(json!({
                "who": if first["actor"] == "you" { "you" } else { "opp" },
                "text": text,
                "card": first["card"]["code"],
                "choice": i + 1,
                "screen": screen(app, &env, self.me, &names, log, json!({"actor": actor, "text": text})),
            }));
        }
        if out.is_empty() {
            return Err(format!("turns {from} to {to} have no plays"));
        }
        Ok(out)
    }
}

/// `games:watch`: the saved game as the play screen shows it (web/app.js replay mode): one `state` frame per choice
/// of turns `from_turn..=to_turn` (default the whole game) plus the end board, from the user's seat, with the cues the
/// live duel drew, the log and a caption (who chose what). The opponent's hand and set cards show unless `reveal` is
/// false (a replay shows both hands, user 2026-11-15). Each frame carries `turn`, `choice` (choices made before it: an
/// export cuts an EDOPro replay there) and `stop`: a play ended on it (Watch steps between stops; the zone, target and
/// position picks in between are not stops), or a turn starts or the duel ends.
pub fn watch(app: &App, args: &Value, cancel: &AtomicBool) -> Result<Value, String> {
    let id = text_arg(args, "record")?.ok_or("`record` is required")?;
    let g = open(app, &id, cancel)?;
    let last = g.turns.last().map_or(0, |t| t.turn);
    let from = whole_arg(args, "from_turn")?.map_or(1, |n| n as i32);
    let to = whole_arg(args, "to_turn")?.map_or(last, |n| n as i32);
    if from > to {
        return Err(format!("from turn {from} is after to turn {to}: pick the turns again"));
    }
    if to > last {
        return Err(format!("the game has turns 1 to {last}; turn {to} is not one of them"));
    }
    let reveal = args["reveal"].as_bool().unwrap_or(true);
    let h = g.header();
    let names = [h["your_deck"].as_str().unwrap_or("You").to_string(), h["opponent_deck"].as_str().unwrap_or("Opponent").to_string()];
    let mut env = Env::new(app.engine, &g.cfg, false).map_err(|e| format!("could not start the replay: {e}"))?;
    env.set_record_trace(true); // cues read the raw message trace
    let mut obs = Obs::zeroed();
    env.reset(&mut obs).map_err(|e| format!("could not start the replay: {e}"))?;
    let mut stats = decklight_view::DuelStats::new(g.me, g.cfg.lp as i64);
    let (mut seen, mut trace) = (0usize, 0usize);
    let mut frames: Vec<Value> = Vec::new();
    // One play-screen frame of the board now; `seen`/`trace` move on so the next frame's log and cues start here.
    let mut frame = |env: &Env<'_>, seen: &mut usize, trace: &mut usize, k: usize, turn: i32, caption: Value| {
        let view = app.ctx.build_view(env, g.me, reveal);
        let log: Vec<Value> = app.ctx.build_events(env, g.me, *seen).into_iter()
            .filter(|e| !matches!(e["kind"].as_str().unwrap_or("").to_lowercase().as_str(), "select" | "position"))
            .collect();
        let played = log.iter().any(|e| !QUIET.iter().any(|q| e["kind"] == *q));
        *seen = env.history_log().len();
        let (cues, _) = decklight_view::build_cues(env, &app.ctx, g.me, *trace, reveal);
        *trace = env.trace_len();
        stats.feed_cues(&cues);
        stats.note_view(Some(&view));
        json!({"t": "state", "view": view, "prompt": null, "log": log, "cues": cues,
            "stats": stats.stats_message(false), "thinking": null, "result": null, "caption": caption,
            "names": names, "turn": turn, "choice": k, "stop": played})
    };
    for (k, &ch) in g.choices.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let turn = env.duel_state()?.turn;
        if turn > to {
            break;
        }
        let seat = env.player();
        let label = app.ctx.build_prompt(&env, seat, k as i64)
            .and_then(|p| p["options"].as_array()?.iter().find(|o| o["index"].as_i64() == Some(ch as i64))?["label"].as_str().map(str::to_string))
            .unwrap_or_else(|| app.ctx.fallback_label(&env, seat, ch.max(0) as usize));
        let actor = if seat == g.me { &names[0] } else { &names[1] };
        let caption = if k == 0 { Value::Null } else { json!({"actor": actor, "text": label}) };
        // Turns before `from` still run through the frame so its log and cues start at `from`.
        let mut f = frame(&env, &mut seen, &mut trace, k, turn, caption);
        if turn >= from {
            if frames.last().is_none_or(|p| p["turn"] != turn) {
                f["stop"] = json!(true);
            }
            frames.push(f);
        }
        env.step(ch, &mut obs).map_err(|e| format!("the saved game does not replay (choice {k}): {e}"))?;
    }
    if env.done() && to == last {
        let w = env.winner();
        let who = if w == g.me { names[0].clone() } else { names[1].clone() };
        let mut f = frame(&env, &mut seen, &mut trace, g.choices.len(), last, json!({"actor": who, "text": "Wins the duel"}));
        let how = if g.rec["result"]["surrendered"].as_bool() == Some(true) { "surrender" } else { "win" };
        f["result"] = json!({"winner": who, "reason": how});
        f["stop"] = json!(true);
        frames.push(f);
    }
    if frames.is_empty() {
        return Err(format!("turns {from} to {to} have no plays"));
    }
    // The duel stats (the side panel) as of the range's end: on its last frame only (a whole game's are ~4 MB).
    let n = frames.len();
    for f in &mut frames[..n - 1] {
        f["stats"] = Value::Null;
    }
    Ok(json!({"id": g.id, "title": super::export::title_of(&g), "day": super::export::day_of(&g), "last_turn": last, "you_went": h["you_went"], "frames": frames}))
}

/// `ask_frames` (the chat's game blocks, not a model tool): the board after each play of a turn, or the board at one
/// of the user's decisions. Same records and the same user's view as `replay`.
pub fn frames(app: &App, args: &Value, cancel: &AtomicBool) -> Result<Value, String> {
    let id = text_arg(args, "record")?.ok_or("`record` is required")?;
    let (turn, decision) = (whole_arg(args, "turn")?, whole_arg(args, "decision")?);
    let g = open(app, &id, cancel)?;
    match (turn, decision) {
        (Some(t), None) => g.turn_frames(app, t as i32, cancel),
        (None, Some(d)) => {
            let k = g.pick(d)?;
            let (env, _) = g.env_at(app, k)?;
            let mut out = g.decision_json(k);
            out["id"] = json!(g.id);
            let h = g.header();
            let names = [h["your_deck"].as_str().unwrap_or("You").to_string(), h["opponent_deck"].as_str().unwrap_or("Opponent").to_string()];
            let caption = json!({"actor": names[0], "text": g.decisions[k].text});
            out["screen"] = screen(app, &env, g.me, &names, Vec::new(), caption);
            Ok(out)
        }
        _ => Err("give `turn` or `decision`".into()),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::record::{RecordGame, ResponseRec};
    use crate::server::{Config, argmax_masked, build_state};
    use std::sync::Arc;

    /// One game of the two eval starter decks with seeded random choices, saved as a record of ours (seat 0).
    pub(in crate::ask) fn play_and_save(app: &App, seed: u64) -> String {
        play_and_save_as(app, seed, false)
    }

    #[test]
    fn watch_frames_are_play_screen_states_from_the_users_seat() {
        let dir = std::env::temp_dir().join(format!("ask-watch-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        let no = AtomicBool::new(false);
        let id = play_and_save(&app, 7);
        let w = watch(&app, &json!({"record": id}), &no).unwrap();
        let frames = w["frames"].as_array().unwrap();
        let last = w["last_turn"].as_i64().unwrap();
        // scripts/app/watch_check.py shows these frames in the real client.
        if let Ok(out) = std::env::var("DECKLIGHT_WATCH_DUMP") {
            std::fs::write(out, w.to_string()).unwrap();
        }
        // The whole game: a frame per choice plus the end board, which carries the result.
        let g = open(&app, &id, &no).unwrap();
        assert_eq!(frames.len(), g.choices.len() + 1);
        assert!(frames.iter().all(|f| f["t"] == "state" && f["view"]["you"].is_object() && f["prompt"].is_null()));
        assert!(frames.last().unwrap()["result"]["winner"].is_string(), "{}", frames.last().unwrap()["result"]);
        assert!(frames.iter().any(|f| f["cues"].as_array().is_some_and(|c| !c.is_empty())), "no cues in the whole game");
        // The opponent's hand stays hidden (null) unless asked, apart from cards the user saw go there (as in live play).
        let unknown = |f: &Value| f["view"]["bot"]["hand"].as_array().unwrap().iter().filter(|c| c.is_null()).count();
        assert!(frames.iter().all(|f| unknown(f) == 0), "a replay shows the opponent's hand");
        // Steps: a play or a turn start ends on each stop; most turns have fewer stops than choices.
        let stops = frames.iter().filter(|f| f["stop"] == true).count();
        assert!(stops > last as usize && stops < frames.len(), "{stops} stops of {}", frames.len());
        // One turn: its frames only, from the same choices as the whole game.
        let t2 = watch(&app, &json!({"record": id, "from_turn": 2, "to_turn": 2}), &no).unwrap();
        let f2 = t2["frames"].as_array().unwrap();
        assert!(!f2.is_empty() && f2.iter().all(|f| f["turn"] == 2));
        let same = frames.iter().find(|f| f["choice"] == f2[0]["choice"]).unwrap();
        assert_eq!(same["view"], f2[0]["view"]);
        assert!(f2[0]["stop"] == true, "a turn starts on a stop");
        let hidden = watch(&app, &json!({"record": id, "from_turn": 2, "to_turn": 2, "reveal": false}), &no).unwrap();
        assert!(hidden["frames"].as_array().unwrap().iter().map(unknown).sum::<usize>() > 0, "reveal false hides the opponent's hand");
        let e = watch(&app, &json!({"record": id, "from_turn": 3, "to_turn": 2}), &no).unwrap_err();
        assert!(e.contains("pick the turns again"), "{e}");
        assert!(watch(&app, &json!({"record": id, "to_turn": last + 1}), &no).is_err());
    }

    /// The same, saved as a practice game when `practice`.
    pub(in crate::ask) fn play_and_save_as(app: &App, seed: u64, practice: bool) -> String {
        let root = &app.root;
        let deck = |n: &str| crate::decks::load_ydk(&root.join("config/ask/eval/fixtures/decks").join(n)).unwrap();
        let (a, b) = (deck("blue-eyes-83b783.ydk"), deck("branded-b27239.ydk"));
        let cfg = DuelConfig {
            main0: a.main.clone(),
            extra0: a.extra.clone(),
            main1: b.main.clone(),
            extra1: b.extra.clone(),
            seed,
            flags: ygoenv_sys::TCG_FLAGS,
            lp: 8000,
            hand: 5,
            draw: 1,
            seed_state: None,
            team_order: [0, 1],
            edopro_add: false,
        };
        let mut env = Env::new(app.engine, &cfg, false).unwrap();
        let mut obs = Obs::zeroed();
        let mut pending = env.reset(&mut obs).unwrap();
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let (mut responses, mut opts) = (Vec::new(), Vec::new());
        let mut steps = 0;
        while pending && !env.done() && steps < 1500 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let seat = env.player();
            let n = env.num_options().max(1) as u64;
            let before = env.num_responses();
            let ch = (rng % n) as i32;
            opts.push(ch);
            pending = env.step(ch, &mut obs).unwrap();
            for k in before..env.num_responses() {
                responses.push(ResponseRec { seat, r: env.response(k), opt: std::mem::take(&mut opts), ms: 0 });
            }
            steps += 1;
        }
        assert!(opts.is_empty(), "a choice did not end in a response");
        let st = env.duel_state().unwrap();
        let human = json!({"type": "human", "player": app.records.install_id.clone()});
        let g = RecordGame {
            seed,
            first: Some(0),
            main0: a.main,
            extra0: a.extra,
            main1: b.main,
            extra1: b.extra,
            seats: [human, json!({"type": "bot", "bot_mode": "random"})],
            mode: "random".into(),
            responses,
            winner: env.done().then_some(st.winner),
            turns: st.turn,
            deck_names: Some(["Blue-Eyes".into(), "Branded".into()]),
            practice: Some(practice),
            ..Default::default()
        };
        let path = app.records.write(&g).unwrap();
        path.file_name().unwrap().to_string_lossy().trim_end_matches(".match.json.gz").to_string()
    }

    pub(in crate::ask) fn app(dir: &Path) -> Arc<App> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let mut config = Config::defaults(&root);
        config.records_dir = dir.join("records");
        config.user_decks_dir = dir.join("decks");
        config.secrets = Some(Arc::new(crate::ask::keys::SessionOnly::default()));
        build_state(&config).unwrap()
    }

    /// Seeded games for checking the chat's game blocks by hand (scripts/app/ask_blocks_check.py):
    /// `ASK_DEMO_DIR=<dir> cargo test --release -p decklight-server writes_demo_games -- --ignored` writes them to
    /// `<dir>/records`.
    #[test]
    #[ignore]
    fn writes_demo_games() {
        let dir = PathBuf::from(std::env::var("ASK_DEMO_DIR").expect("set ASK_DEMO_DIR"));
        let app = app(&dir);
        for seed in [7, 11, 23] {
            let id = play_and_save(&app, seed);
            println!("wrote {id}");
        }
    }

    /// The sample games shipped with the app (config/samples/games, user 2026-11-15: "they can be just sample games
    /// so users know about the utility"): the bot's policy plays both seats greedily from shuffled decks (unshuffled .ydk
    /// order dealt bricks: both sides passed most turns), the user's seat is the sample player. Regenerate with
    /// `ORT_DYLIB_PATH=... cargo test --release -p decklight-server writes_sample_games -- --ignored`.
    #[test]
    #[ignore]
    fn writes_sample_games() {
        let dir = std::env::temp_dir().join(format!("ask-samples-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        let root = app.root.clone();
        let deck = |n: &str| crate::decks::load_ydk(&root.join("config/ask/eval/fixtures/decks").join(n)).unwrap();
        let out = root.join(SAMPLES_DIR);
        std::fs::create_dir_all(&out).unwrap();
        // (file id, the sample player's seat, seat 0 deck, seat 1 deck, seed): Blue-Eyes going first and second,
        // Branded going first.
        let games = [
            ("sample-1", 0, ("blue-eyes-83b783.ydk", "Blue-Eyes"), ("branded-b27239.ydk", "Branded"), 111u64),
            ("sample-2", 1, ("branded-b27239.ydk", "Branded"), ("blue-eyes-83b783.ydk", "Blue-Eyes"), 202),
            ("sample-3", 0, ("branded-b27239.ydk", "Branded"), ("blue-eyes-83b783.ydk", "Blue-Eyes"), 303),
        ];
        let mut nets = app.onnx_pool.take().expect("the bot's model (set ORT_DYLIB_PATH)");
        for (id, me, (fa, na), (fb, nb), seed) in games {
            let (mut a, mut b) = (deck(fa), deck(fb));
            // The app deals from a shuffled deck (product.rs); the .ydk order puts copies together.
            decklight_cards::shuffle_main(&mut a.main, seed, 0);
            decklight_cards::shuffle_main(&mut b.main, seed, 1);
            let cfg = DuelConfig {
                main0: a.main.clone(), extra0: a.extra.clone(), main1: b.main.clone(), extra1: b.extra.clone(), seed,
                flags: ygoenv_sys::TCG_FLAGS, lp: 8000, hand: 5, draw: 1, seed_state: None, team_order: [0, 1],
                edopro_add: false,
            };
            let mut env = Env::new(app.engine, &cfg, false).unwrap();
            let mut obs = Obs::zeroed();
            let mut pending = env.reset(&mut obs).unwrap();
            let (mut responses, mut opts) = (Vec::new(), Vec::new());
            let mut steps = 0;
            while pending && !env.done() && steps < 4000 {
                let seat = env.player();
                let logits = nets.policy_logits(&[&obs]).unwrap().remove(0);
                let ch = argmax_masked(&logits, &obs.action_mask);
                let before = env.num_responses();
                opts.push(ch as i32);
                pending = env.step(ch as i32, &mut obs).unwrap();
                for k in before..env.num_responses() {
                    responses.push(ResponseRec { seat, r: env.response(k), opt: std::mem::take(&mut opts), ms: 0 });
                }
                steps += 1;
            }
            assert!(env.done(), "{id}: the game did not finish in {steps} choices");
            let st = env.duel_state().unwrap();
            let player = json!({"type": "human", "player": SAMPLE_PLAYER});
            let bot = json!({"type": "bot", "bot_mode": "policy"});
            let g = RecordGame {
                seed, first: Some(0), main0: a.main, extra0: a.extra, main1: b.main, extra1: b.extra,
                seats: if me == 0 { [player, bot] } else { [bot, player] },
                mode: "sample".into(), responses, winner: Some(st.winner), turns: st.turn,
                deck_names: Some([na.into(), nb.into()]), ..Default::default()
            };
            let path = app.records.write(&g).unwrap();
            let mut rec = decklight_record::load_gz(&path).unwrap();
            rec["id"] = json!(id);
            rec["sample"] = json!(true);
            rec["created"] = json!("2026-11-15T00:00:00Z");
            decklight_record::save_gz(&rec, &out.join(format!("{id}.match.json.gz"))).unwrap();
            println!("{id}: {} turns, winner seat {} (sample player seat {me}), {} choices", st.turn, st.winner, steps);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The shipped samples: listed apart from the user's games, replayable by id, from the sample player's seat.
    #[test]
    fn sample_games_are_listed_and_replay_but_are_not_the_users() {
        let dir = std::env::temp_dir().join(format!("ask-sample-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        let no = AtomicBool::new(false);
        let list = replay(&app, &json!({}), &no).unwrap();
        assert_eq!(list["games"], json!([]), "samples are never the user's games");
        let ids: Vec<&str> = list["samples"].as_array().unwrap().iter().map(|g| g["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["sample-1", "sample-2", "sample-3"]);
        assert_eq!(list["samples"][1]["you_went"], "second");
        let e = replay(&app, &json!({"record": "last"}), &no).unwrap_err();
        assert!(e.contains("play a game first") && e.contains("sample-1"), "{e}");
        let g = replay(&app, &json!({"record": "sample-2"}), &no).unwrap();
        assert_eq!(g["sample"], true);
        assert_eq!(g["your_deck"], "Blue-Eyes");
        assert!(g["your_decisions"].as_u64().unwrap() > 10, "{g}");
        if std::env::var("ASK_PRINT_SAMPLE").is_ok() {
            println!("{}", serde_json::to_string_pretty(&g).unwrap());
        }
        assert!(replay(&app, &json!({"record": "sample-9"}), &no).unwrap_err().contains("no sample game"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn replay_lists_summarises_and_shows_a_saved_game_from_the_users_view() {
        let dir = std::env::temp_dir().join(format!("ask-replay-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        let no = AtomicBool::new(false);
        assert_eq!(replay(&app, &json!({}), &no).unwrap()["games"], json!([]));
        assert!(replay(&app, &json!({"record": "last"}), &no).unwrap_err().contains("no saved games"));
        let id = play_and_save(&app, 7);

        let list = replay(&app, &json!({}), &no).unwrap();
        assert_eq!(list["games"][0]["id"], id.as_str());
        assert_eq!(list["games"][0]["your_deck"], "Blue-Eyes");
        assert_eq!(list["games"][0]["you_went"], "first");

        let game = replay(&app, &json!({"record": "last"}), &no).unwrap();
        let n = game["your_decisions"].as_u64().unwrap();
        assert!(n > 3, "{game}");
        let log = game["log"].as_array().unwrap();
        assert_eq!(log[0]["turn"], 1);
        assert_eq!(log[0]["player"], "you");
        assert!(log.iter().any(|t| t["events"].as_array().is_some_and(|e| !e.is_empty())), "{game}");
        let text = game.to_string();
        assert!(!text.contains("Chooses a zone") && !text.contains("Passes"), "quiet choices in the log: {text}");
        assert_eq!(log[0]["lp_at_start"], json!([8000, 8000]));
        assert!(game["final_lp"]["you"].is_i64());
        assert!(text.len() < 24_000, "summary too long for the model: {} chars", text.len());

        let turn1 = replay(&app, &json!({"record": id, "turn": 1}), &no).unwrap();
        let d1 = &turn1["your_decisions"][0];
        assert_eq!(d1["decision"], 1);
        assert!(d1["options"].as_array().unwrap().len() >= 2);
        assert!(!d1["you_picked"].as_str().unwrap().is_empty());

        let board = replay(&app, &json!({"record": id, "decision": 1}), &no).unwrap();
        assert_eq!(board["you"]["lp"], 8000);
        assert!(board["you"]["hand"].as_array().unwrap().len() >= 5, "{board}");
        // The opponent's hand is a count, never names.
        assert!(board["opponent"]["hand"].as_str().unwrap().ends_with("cards"), "{board}");
        assert_eq!(board["of"], n);

        for (args, want) in [
            (json!({"record": id, "decision": n + 1}), "decisions 1 to"),
            (json!({"record": id, "turn": 999}), "not one of them"),
            (json!({"record": id, "turn": 1, "decision": 1}), "not both"),
            (json!({"record": "nope"}), "no saved game"),
            (json!({"record": "../records/x"}), "not a game id"),
            (json!({"record": id, "step": 3}), "unknown argument"),
            (json!({"limit": 50}), "1 to 20"),
        ] {
            let e = replay(&app, &args, &no).unwrap_err();
            assert!(e.contains(want), "{args}: {e}");
        }
        let stop = AtomicBool::new(true);
        assert_eq!(replay(&app, &json!({"record": id}), &stop).unwrap_err(), "stopped");

        // The chat's play sequence: the turn's first board, then one board per play, from the user's view.
        let t2 = log.iter().find(|t| t["events"].as_array().is_some_and(|e| !e.is_empty())).unwrap()["turn"].as_u64().unwrap();
        let f = frames(&app, &json!({"record": id, "turn": t2}), &no).unwrap();
        let fs = f["frames"].as_array().unwrap();
        assert!(fs.len() >= 2 && fs.len() <= FRAMES_CAP, "{f}");
        assert_eq!(fs[0]["who"], "");
        assert!(fs[1..].iter().all(|x| (x["who"] == "you" || x["who"] == "opp") && !x["text"].as_str().unwrap().is_empty()), "{f}");
        let b = &fs[0]["screen"];
        assert_eq!(b["t"], "state");
        assert_eq!(b["view"]["turn"], t2);
        assert_eq!(b["view"]["you"]["mzone"].as_array().unwrap().len(), 5);
        assert!(b["view"]["bot"]["hand"].as_array().unwrap().iter().all(|c| !c.is_null()), "a replay shows the opponent's hand: {b}");
        assert!(fs[1]["screen"]["caption"]["text"] == fs[1]["text"], "{}", fs[1]);
        assert!(!f.to_string().contains("Chooses a zone"), "{f}");
        let at = frames(&app, &json!({"record": id, "decision": 1}), &no).unwrap();
        assert_eq!(at["screen"]["view"]["you"]["lp"], 8000);
        assert!(at["screen"]["view"]["you"]["hand"][0]["code"].as_i64().unwrap() > 0, "{at}");
        assert_eq!(at["choice"], 0);
        assert!(frames(&app, &json!({"record": id}), &no).unwrap_err().contains("give `turn` or `decision`"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Needs the ONNX runtime (ORT_DYLIB_PATH) and the bot's graphs; skipped otherwise.
    #[test]
    fn bot_view_ranks_the_bots_options_and_includes_the_users_pick() {
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("skip: no ORT_DYLIB_PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!("ask-botview-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        if !app.onnx_dir.join("manifest.json").is_file() {
            eprintln!("skip: no ONNX graphs in {}", app.onnx_dir.display());
            return;
        }
        let id = play_and_save(&app, 11);
        let no = AtomicBool::new(false);
        let v = bot_view(&app, &json!({"record": id, "decision": 2}), &no).unwrap();
        let top = v["bot_top"].as_array().unwrap();
        assert!(!top.is_empty() && top.len() <= 3, "{v}");
        let probs: Vec<f64> = top.iter().map(|r| r["bot_prob"].as_f64().unwrap()).collect();
        assert!(probs.windows(2).all(|w| w[0] >= w[1]), "{v}");
        for r in top.iter().chain([&v["you_picked"]]) {
            let w = r["win_chance"].as_f64().unwrap();
            assert!((0.0..=100.0).contains(&w), "{v}");
        }
        assert!(v["win_chance_now"].as_f64().is_some_and(|w| (0.0..=100.0).contains(&w)));
        let e = bot_view(&app, &json!({"record": id}), &no).unwrap_err();
        assert!(e.contains("`decision` is required"), "{e}");
        let stop = AtomicBool::new(true);
        assert_eq!(bot_view(&app, &json!({"record": id, "decision": 2}), &stop).unwrap_err(), "stopped");
        std::fs::remove_dir_all(&dir).ok();
    }
}
