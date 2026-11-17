//! The Games screen (Play > Games; mocks docs/ui/mocks/ask/mocks.html games, game, game-decision, approved 2026-11-15):
//! the user's saved games, their stats and the bot's view of each decision without asking (user 2026-11-15:
//! "whatever the llm has access to the user should also have access without having to ask"). Everything here reads
//! the same data as the `replay`, `bot_view` and `my_stats` tools, through the same functions.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

use super::replay::{self, Game};
use crate::server::{App, mark_for_delta};

/// Games per page of the list.
const PAGE: u64 = 30;
/// The stat card's window.
const STATS_DAYS: i64 = 30;

/// Life points by turn per game id: a list row needs a replay of the game, so each is computed once per run.
fn lp_cache() -> &'static Mutex<HashMap<String, Vec<[i64; 2]>>> {
    static C: OnceLock<Mutex<HashMap<String, Vec<[i64; 2]>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// "Check my decisions" results per game id (the review takes about a minute; reopening the game keeps it).
fn check_cache() -> &'static Mutex<HashMap<String, Value>> {
    static C: OnceLock<Mutex<HashMap<String, Value>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lp_of(app: &App, id: &str, cancel: &AtomicBool) -> Value {
    if let Some(v) = lp_cache().lock().unwrap().get(id) {
        return json!(v);
    }
    match replay::open(app, id, cancel) {
        Ok(g) => {
            let v = g.lp_series();
            lp_cache().lock().unwrap().insert(id.to_string(), v.clone());
            json!(v)
        }
        // The row still lists the game; its page says why it does not replay.
        Err(_) => Value::Null,
    }
}

/// The chips above the list: (the kind my_stats counts, the kinds listed, opponent type). All and Bot list practice
/// games too (tagged in the app) but count only duels, as the lobby promises; puzzles have their own tab.
type Filter = (&'static str, &'static [&'static str], Option<&'static str>);
fn filter_of(f: &str) -> Result<Filter, String> {
    match f {
        "all" => Ok(("duel", &["duel", "practice"], None)),
        "bot" => Ok(("duel", &["duel", "practice"], Some("bot"))),
        "people" => Ok(("duel", &["duel"], Some("remote"))),
        "practice" => Ok(("practice", &["practice"], None)),
        other => Err(format!("unknown filter {other:?}: all, bot, people or practice")),
    }
}

/// The list's WHERE clause: kinds, opponent type and the search text (?1 ?4 kinds, ?2 opponent, ?5 a LIKE pattern
/// over both deck names, or NULL; ?3 is the row limit where there is one).
const WHERE: &str = "me >= 0 AND (kind = ?1 OR kind = ?4) AND (?2 IS NULL OR opp_type = ?2) \
    AND (?5 IS NULL OR my_deck LIKE ?5 ESCAPE '\\' OR opp_deck LIKE ?5 ESCAPE '\\')";

/// The search box's text as a LIKE pattern (SQLite's LIKE ignores ASCII case), or None for an empty box.
fn pattern(q: &str) -> Option<String> {
    let q = q.trim();
    (!q.is_empty()).then(|| format!("%{}%", q.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")))
}

fn open_index(app: &App) -> Result<Option<rusqlite::Connection>, String> {
    let index = app.records.root.join("index.sqlite");
    if !index.is_file() {
        return Ok(None);
    }
    rusqlite::Connection::open_with_flags(&index, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map(Some)
        .map_err(|e| format!("cannot open the game history: {e}"))
}

fn rows(app: &App, kinds: &[&str], opp: Option<&str>, q: Option<&str>, limit: u64) -> Result<Vec<Value>, String> {
    let Some(con) = open_index(app)? else { return Ok(Vec::new()) };
    let mut st = con
        .prepare(&format!(
            "SELECT id, created, mode, my_deck, opp_deck, opp_type, first, me, won, turns, kind FROM matches \
             WHERE {WHERE} ORDER BY created DESC, rowid DESC LIMIT ?3"
        ))
        .map_err(|e| format!("cannot read the game history: {e}"))?;
    let k2 = kinds.get(1).unwrap_or(&kinds[0]);
    st.query_map(rusqlite::params![kinds[0], opp, limit as i64, k2, q], |r| {
        let (first, me, won): (i64, i64, i64) = (r.get(6)?, r.get(7)?, r.get(8)?);
        Ok(json!({
            "id": r.get::<_, String>(0)?,
            "when": r.get::<_, String>(1)?,
            "mode": r.get::<_, String>(2)?,
            "your_deck": r.get::<_, String>(3)?,
            "opponent_deck": r.get::<_, String>(4)?,
            "opponent": r.get::<_, String>(5)?,
            "you_went": if first < 0 { "unknown" } else if first == me { "first" } else { "second" },
            "result": match won { 1 => "you won", 0 => "you lost", _ => "not decided" },
            "turns": r.get::<_, i64>(9)?,
            "kind": r.get::<_, Option<String>>(10)?,
        }))
    })
    .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
    .map_err(|e| format!("cannot read the game history: {e}"))
}

/// Every id the list would show under these chips and search (Select > "Select all").
fn all_ids(app: &App, kinds: &[&str], opp: Option<&str>, q: Option<&str>) -> Result<Vec<String>, String> {
    let Some(con) = open_index(app)? else { return Ok(Vec::new()) };
    let mut st = con
        .prepare(&format!("SELECT id FROM matches WHERE {WHERE} ORDER BY created DESC, rowid DESC"))
        .map_err(|e| format!("cannot read the game history: {e}"))?;
    let k2 = kinds.get(1).unwrap_or(&kinds[0]);
    st.query_map(rusqlite::params![kinds[0], opp, Option::<i64>::None, k2, q], |r| r.get(0))
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map_err(|e| format!("cannot read the game history: {e}"))
}

/// The ids of every game under the list's chips and search, for "Select all".
pub fn ids(app: &App, args: &Value) -> Result<Value, String> {
    let (_, listed, opp) = filter_of(args["filter"].as_str().unwrap_or("all"))?;
    let q = pattern(args["q"].as_str().unwrap_or(""));
    Ok(json!({"ids": all_ids(app, listed, opp, q.as_deref())?}))
}

/// Delete saved games (Games > Select > Delete; mocks games-select, games-delete): the files go to the records'
/// trash folder and leave the index, so the list, the stats and Ask stop seeing them. Samples are not records.
pub fn delete(app: &App, args: &Value) -> Result<Value, String> {
    let ids: Vec<String> = args["ids"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect();
    if ids.is_empty() {
        return Err("no games selected: tick the games to delete first".into());
    }
    let n = app.records.trash(&ids)?;
    for id in &ids {
        lp_cache().lock().unwrap().remove(id);
        check_cache().lock().unwrap().remove(id);
    }
    Ok(json!({"deleted": n}))
}

/// The Games list: the stat card (my_stats over the last 30 days, by turn order, deck and opponent deck), the
/// user's games newest first with their life points by turn, and the sample games.
pub fn list(app: &App, args: &Value, cancel: &AtomicBool) -> Result<Value, String> {
    let filter = args["filter"].as_str().unwrap_or("all");
    let (kind, listed, opp) = filter_of(filter)?;
    // "Show 30 more" raises the limit; no cap below the history's size (a cap left games past it unreachable).
    let limit = args["limit"].as_u64().unwrap_or(PAGE).max(1);
    let since = (chrono::Utc::now() - chrono::Duration::days(STATS_DAYS)).format("%Y-%m-%d").to_string();
    let index = app.records.root.join("index.sqlite");
    let stat = |group: Option<&str>| -> Result<Value, String> {
        let mut a = json!({"kind": kind, "since": since});
        if let Some(o) = opp {
            a["opponent"] = json!(o);
        }
        if let Some(g) = group {
            a["group_by"] = json!(g);
        }
        Ok(super::stats::my_stats(&a, &index)?["summary"].clone())
    };
    let stats = json!({
        "days": STATS_DAYS,
        "total": stat(None)?,
        "first": stat(Some("first"))?["groups"],
        "deck": stat(Some("deck"))?["groups"],
        "opponent_deck": stat(Some("opponent_deck"))?["groups"],
    });
    let q = pattern(args["q"].as_str().unwrap_or(""));
    let total = all_ids(app, listed, opp, q.as_deref())?.len();
    let mut games = rows(app, listed, opp, q.as_deref(), limit + 1)?;
    let more = games.len() as u64 > limit;
    games.truncate(limit as usize);
    for g in games.iter_mut() {
        if cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let id = g["id"].as_str().unwrap_or("").to_string();
        g["lp"] = lp_of(app, &id, cancel);
    }
    let mut samples = replay::samples(app);
    for s in samples.iter_mut() {
        let id = s["id"].as_str().unwrap_or("").to_string();
        s["lp"] = lp_of(app, &id, cancel);
    }
    Ok(json!({"filter": filter, "stats": stats, "games": games, "more": more, "total": total, "samples": samples}))
}

fn decisions_json(g: &Game) -> Vec<Value> {
    (0..g.decisions.len())
        .map(|k| {
            let d = &g.decisions[k];
            json!({
                "decision": k + 1,
                "turn": d.turn,
                "phase": d.phase,
                "prompt": d.kind,
                "text": d.text,
                "you_picked": d.picked,
                "picked_index": g.picked_index(k),
                "options": d.options.iter().map(|(i, l, c)| json!({"index": i, "label": l, "code": c})).collect::<Vec<_>>(),
            })
        })
        .collect()
}

/// One game's page: the replay summary (header and turn log), every decision of the user's with all its options,
/// and the "Check my decisions" marks when that already ran.
pub fn page(app: &App, args: &Value, cancel: &AtomicBool) -> Result<Value, String> {
    let id = args["record"].as_str().ok_or("`record` is required: a game id from the list")?;
    let g = replay::open(app, id, cancel)?;
    lp_cache().lock().unwrap().insert(g.id.clone(), g.lp_series());
    let mut out = g.summary();
    out["decisions"] = json!(decisions_json(&g));
    if let Some(c) = check_cache().lock().unwrap().get(&g.id) {
        out["check"] = c.clone();
    }
    Ok(out)
}

/// "Check my decisions": the bot's view of each of the user's decisions, marked like the post-game review (the
/// points of win chance the pick gave up against the best option looked at). `progress(done, total)` after each.
pub fn check(app: &App, args: &Value, cancel: &AtomicBool, progress: &dyn Fn(usize, usize)) -> Result<Value, String> {
    let id = args["record"].as_str().ok_or("`record` is required: a game id from the list")?;
    let g = replay::open(app, id, cancel)?;
    let map = replay::calib(app)?;
    let mut nets = app.onnx_pool.take().map_err(|e| format!("Check my decisions needs the bot's model: {e}"))?;
    let n = g.decisions.len();
    let mut marks = Vec::with_capacity(n);
    progress(0, n);
    for k in 0..n {
        if cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let v = replay::bot_view_at(app, &g, k, &map, &mut nets, cancel)?;
        let best = v["bot_top"].as_array().into_iter().flatten().chain(v.get("you_picked")).filter_map(|o| o["win_chance"].as_f64()).fold(f64::NAN, f64::max);
        let yours = v["you_picked"]["win_chance"].as_f64();
        let delta = match yours {
            Some(y) if best.is_finite() => ((y - best) * 10.0).round() / 10.0,
            _ => 0.0,
        };
        marks.push(json!({"decision": k + 1, "delta": delta, "mark": mark_for_delta(delta), "view": v}));
        progress(k + 1, n);
    }
    let out = json!({"decisions": marks});
    check_cache().lock().unwrap().insert(g.id.clone(), out.clone());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ask::replay::tests::{app, play_and_save, play_and_save_as};

    #[test]
    fn list_counts_duels_lists_practice_apart_and_pages_a_game() {
        let dir = std::env::temp_dir().join(format!("ask-games-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        let no = AtomicBool::new(false);
        let empty = list(&app, &json!({}), &no).unwrap();
        assert_eq!(empty["games"], json!([]));
        assert_eq!(empty["stats"]["total"]["games"], 0);
        assert!(!empty["samples"].as_array().unwrap().is_empty(), "the sample games are listed: {empty}");
        assert!(empty["samples"][0]["lp"].as_array().is_some_and(|v| v.len() >= 2), "{}", empty["samples"][0]);
        let duel = play_and_save(&app, 7);
        let practice = play_and_save_as(&app, 11, true);
        let all = list(&app, &json!({"filter": "all"}), &no).unwrap();
        let ids: Vec<&str> = all["games"].as_array().unwrap().iter().map(|g| g["id"].as_str().unwrap()).collect();
        assert_eq!(ids.len(), 2, "All lists practice games too: {all}");
        assert!(ids.contains(&duel.as_str()) && ids.contains(&practice.as_str()));
        assert_eq!(all["stats"]["total"]["games"], 1, "but counts only the duel");
        let all = list(&app, &json!({"filter": "all", "limit": 1}), &no).unwrap();
        assert_eq!((all["games"].as_array().unwrap().len(), all["more"].as_bool()), (1, Some(true)));
        let all = list(&app, &json!({"filter": "bot"}), &no).unwrap();
        let g = all["games"].as_array().unwrap().iter().find(|g| g["id"] == duel.as_str()).unwrap().clone();
        assert_eq!(all["stats"]["first"][0]["key"], "first");
        let lp = g["lp"].as_array().unwrap();
        assert_eq!(lp[0], json!([8000, 8000]));
        assert_eq!(lp.len() as i64, g["turns"].as_i64().unwrap() + 1, "a point per turn and the end: {lp:?}");
        let pr = list(&app, &json!({"filter": "practice"}), &no).unwrap();
        assert_eq!(pr["games"][0]["id"], practice.as_str());
        assert_eq!(pr["stats"]["total"]["games"], 1);
        assert_eq!(list(&app, &json!({"filter": "people"}), &no).unwrap()["games"], json!([]));
        assert!(list(&app, &json!({"filter": "friends"}), &no).unwrap_err().contains("unknown filter"));

        let p = page(&app, &json!({"record": duel}), &no).unwrap();
        assert_eq!(p["id"], duel.as_str());
        assert!(p["log"].as_array().is_some_and(|l| !l.is_empty()));
        let ds = p["decisions"].as_array().unwrap();
        assert_eq!(ds.len() as i64, p["your_decisions"].as_i64().unwrap());
        let d = &ds[0];
        let opts = d["options"].as_array().unwrap();
        assert!(opts.len() >= 2, "{d}");
        assert!(opts.iter().any(|o| o["index"] == d["picked_index"] && o["label"] == d["you_picked"]), "the pick is one of the options: {d}");
        assert!(p.get("check").is_none());
        assert!(page(&app, &json!({"record": "nope"}), &no).unwrap_err().contains("no saved game"));
        let s = page(&app, &json!({"record": "sample-1"}), &no).unwrap();
        assert_eq!(s["sample"], true);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_select_all_and_delete() {
        let dir = std::env::temp_dir().join(format!("ask-gdel-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        let no = AtomicBool::new(false);
        let a = play_and_save(&app, 7);
        let b = play_and_save(&app, 8);
        let all = list(&app, &json!({}), &no).unwrap();
        assert_eq!(all["total"], 2);
        let deck = all["games"][0]["your_deck"].as_str().unwrap().to_string();
        assert!(deck.len() >= 3, "{all}");
        // Any part of a deck name, in any case; a % or _ is literal text.
        let part = deck[1..deck.len() - 1].to_uppercase();
        let hit = list(&app, &json!({"q": part}), &no).unwrap();
        assert_eq!((hit["games"].as_array().unwrap().len(), hit["total"].as_u64()), (2, Some(2)), "{part}: {hit}");
        for q in ["no such deck", "%", "_"] {
            assert_eq!(list(&app, &json!({"q": q}), &no).unwrap()["total"], 0, "{q}");
        }
        let every = ids(&app, &json!({"filter": "all", "q": part})).unwrap();
        assert_eq!(every["ids"].as_array().unwrap().len(), 2);
        // Delete one: it leaves the list, the stats and the replay; the other stays.
        lp_of(&app, &a, &no);
        assert_eq!(delete(&app, &json!({"ids": [a]})).unwrap()["deleted"], 1);
        let after = list(&app, &json!({}), &no).unwrap();
        assert_eq!(after["games"].as_array().unwrap().iter().map(|g| g["id"].as_str().unwrap()).collect::<Vec<_>>(), [b.as_str()]);
        assert_eq!(after["stats"]["total"]["games"], 1);
        assert!(!lp_cache().lock().unwrap().contains_key(&a));
        assert!(page(&app, &json!({"record": a}), &no).unwrap_err().contains("no saved game"));
        assert!(delete(&app, &json!({"ids": [a]})).unwrap_err().contains("out of date"));
        assert!(delete(&app, &json!({"ids": []})).unwrap_err().contains("no games selected"));
        assert!(delete(&app, &json!({"ids": ["sample-1"]})).is_err(), "samples are not records");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Needs the ONNX runtime (ORT_DYLIB_PATH) and the bot's graphs; skipped otherwise.
    #[test]
    fn check_marks_every_decision_and_the_page_keeps_it() {
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("skip: no ORT_DYLIB_PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!("ask-check-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        if !app.onnx_dir.join("manifest.json").is_file() {
            eprintln!("skip: no ONNX graphs in {}", app.onnx_dir.display());
            return;
        }
        let no = AtomicBool::new(false);
        let id = "sample-1";
        let seen = Mutex::new(Vec::new());
        let c = check(&app, &json!({"record": id}), &no, &|d, n| seen.lock().unwrap().push((d, n))).unwrap();
        let marks = c["decisions"].as_array().unwrap();
        let n = page(&app, &json!({"record": id}), &no).unwrap()["your_decisions"].as_u64().unwrap() as usize;
        assert_eq!(marks.len(), n);
        let seen = seen.into_inner().unwrap();
        assert_eq!((seen[0], *seen.last().unwrap()), ((0, n), (n, n)));
        for m in marks {
            let d = m["delta"].as_f64().unwrap();
            assert!((-100.0..=0.0).contains(&d), "{m}");
            assert_eq!(m["mark"], mark_for_delta(d));
        }
        assert_eq!(page(&app, &json!({"record": id}), &no).unwrap()["check"], c);
        std::fs::remove_dir_all(&dir).ok();
    }
}
