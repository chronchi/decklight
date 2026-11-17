//! `my_stats` (config/ask/tools.v1.json): the user's own results from the record index (`records/index.sqlite`,
//! decklight-record). Only games where one seat is ours (`me >= 0`) count. v1 holds bot games; room games join when
//! they are indexed (docs/plans/ask-tab.md §Human games), so every answer says which opponent kinds it counted.

use std::path::Path;

use serde_json::{Value, json};

const GROUPS: [&str; 6] = ["deck", "opponent_deck", "first", "mode", "kind", "month"];
const KINDS: [&str; 4] = ["duel", "practice", "puzzle", "all"];

/// One counted game: the fields the filters and groups need.
struct Game {
    id: String,
    created: String,
    mode: String,
    my_deck: String,
    opp_deck: String,
    opp_type: String,
    went_first: Option<bool>,
    won: Option<bool>,
    kind: String,
}

fn text(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(Some(s.trim().to_string())),
        Some(v) => Err(format!("`{key}` must be non-empty text, got {v}")),
    }
}

fn load(index: &Path) -> Result<Vec<Game>, String> {
    if !index.is_file() {
        return Ok(Vec::new());
    }
    let con = rusqlite::Connection::open_with_flags(index, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("cannot open the game history: {e}"))?;
    let mut st = con
        .prepare(
            "SELECT id, created, mode, my_deck, opp_deck, opp_type, first, me, won, kind FROM matches \
             WHERE me >= 0 ORDER BY created, rowid",
        )
        .map_err(|e| format!("cannot read the game history: {e}"))?;
    let rows = st
        .query_map([], |r| {
            let (first, me, won): (i64, i64, i64) = (r.get(6)?, r.get(7)?, r.get(8)?);
            Ok(Game {
                id: r.get(0)?,
                created: r.get(1)?,
                mode: r.get(2)?,
                my_deck: r.get(3)?,
                opp_deck: r.get(4)?,
                opp_type: r.get(5)?,
                went_first: (first >= 0).then_some(first == me),
                won: (won >= 0).then_some(won == 1),
                kind: r.get::<_, Option<String>>(9)?.unwrap_or_default(),
            })
        })
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map_err(|e| format!("cannot read the game history: {e}"))?;
    Ok(rows)
}

fn tally(games: &[&Game]) -> Value {
    let wins = games.iter().filter(|g| g.won == Some(true)).count();
    let losses = games.iter().filter(|g| g.won == Some(false)).count();
    let decided = wins + losses;
    json!({
        "games": games.len(),
        "wins": wins,
        "losses": losses,
        "unknown": games.len() - decided,
        // Percent of decided games; null when none is decided (never a made-up 0%).
        "win_rate": (decided > 0).then(|| (wins as f64 * 1000.0 / decided as f64).round() / 10.0),
    })
}

/// The tool: `args` as in tools.v1.json, over the index at `index`. Filters AND together; text filters match
/// case-insensitively and in full (a deck name, a mode).
pub fn my_stats(args: &Value, index: &Path) -> Result<Value, String> {
    if let Some(obj) = args.as_object() {
        let known = ["deck", "opponent_deck", "opponent", "mode", "kind", "since", "group_by"];
        if let Some(k) = obj.keys().find(|k| !known.contains(&k.as_str())) {
            return Err(format!("unknown argument `{k}` (allowed: {})", known.join(", ")));
        }
    } else if !args.is_null() {
        return Err("arguments must be an object".into());
    }
    let deck = text(args, "deck")?;
    let opp_deck = text(args, "opponent_deck")?;
    let opponent = text(args, "opponent")?.map(|o| o.to_ascii_lowercase());
    if let Some(o) = &opponent
        && o != "bot"
        && o != "remote"
    {
        return Err(format!("`opponent` must be bot or remote, got {o:?}"));
    }
    let mode = text(args, "mode")?;
    // Practice (hints, undo) and puzzles are not counted unless asked for, like the lobby says (user 2026-10-31).
    let kind = text(args, "kind")?.map(|k| k.to_ascii_lowercase()).unwrap_or_else(|| "duel".into());
    if !KINDS.contains(&kind.as_str()) {
        return Err(format!("`kind` must be one of {}, got {kind:?}", KINDS.join(", ")));
    }
    // Normalised to zero-padded YYYY-MM-DD so it compares with `created` as text ("2026-11-2" too).
    let since = match text(args, "since")? {
        None => None,
        Some(s) => match chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
            Ok(d) => Some(d.format("%Y-%m-%d").to_string()),
            Err(_) => return Err(format!("`since` must be a date like 2026-11-01, got {s:?}")),
        },
    };
    let group_by = text(args, "group_by")?.map(|g| g.to_ascii_lowercase());
    if let Some(g) = &group_by
        && !GROUPS.contains(&g.as_str())
    {
        return Err(format!("`group_by` must be one of {}, got {g:?}", GROUPS.join(", ")));
    }

    let all = load(index)?;
    let same = |a: &str, b: &Option<String>| b.as_ref().is_none_or(|b| a.eq_ignore_ascii_case(b));
    let games: Vec<&Game> = all
        .iter()
        .filter(|g| {
            same(&g.my_deck, &deck)
                && same(&g.opp_deck, &opp_deck)
                && same(&g.opp_type, &opponent)
                && same(&g.mode, &mode)
                && (kind == "all" || g.kind == kind)
                && since.as_ref().is_none_or(|s| g.created.as_str() >= s.as_str())
        })
        .collect();

    let mut out = tally(&games);
    let mut kinds: Vec<&str> = games.iter().map(|g| g.opp_type.as_str()).collect();
    kinds.sort_unstable();
    kinds.dedup();
    out["opponents_counted"] = json!(kinds);
    out["kind_counted"] = json!(kind);
    if !kinds.contains(&"remote") {
        out["note"] = json!("Bot games only: games against people are not in the history yet.");
    }
    if let Some(g) = &group_by {
        let key = |x: &Game| -> String {
            let named = |s: &str| if s.is_empty() { "unnamed deck".to_string() } else { s.to_string() };
            match g.as_str() {
                "deck" => named(&x.my_deck),
                "opponent_deck" => named(&x.opp_deck),
                "first" => match x.went_first {
                    Some(true) => "first".into(),
                    Some(false) => "second".into(),
                    None => "unknown".into(),
                },
                "mode" => x.mode.clone(),
                "kind" => x.kind.clone(),
                _ => x.created.get(..7).unwrap_or("unknown").to_string(),
            }
        };
        let mut keys: Vec<String> = games.iter().map(|x| key(x)).collect();
        keys.sort();
        keys.dedup();
        let groups: Vec<Value> = keys
            .into_iter()
            .map(|k| {
                let members: Vec<&Game> = games.iter().copied().filter(|x| key(x) == k).collect();
                let mut t = tally(&members);
                t["key"] = json!(k);
                t
            })
            .collect();
        out["group_by"] = json!(g);
        out["groups"] = json!(groups);
    }
    let refs: Vec<&str> = games.iter().rev().take(50).map(|g| g.id.as_str()).collect();
    Ok(json!({"summary": out, "ref": {"records": refs}}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An index with the decklight-record layout and the given rows (me, first, won, my_deck, opp_deck, opp_type,
    /// mode, created).
    fn index(rows: &[(i64, i64, i64, &str, &str, &str, &str, &str)]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("dl-stats-{}", decklight_record::random_hex(4)));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.sqlite");
        let con = rusqlite::Connection::open(&path).unwrap();
        con.execute_batch(
            "CREATE TABLE matches (id TEXT PRIMARY KEY, created TEXT, mode TEXT, deck0_name TEXT, deck0_hash TEXT, \
             deck1_name TEXT, deck1_hash TEXT, opponent TEXT, result INTEGER, turns INTEGER, path TEXT, \
             me INTEGER, first INTEGER, won INTEGER, my_deck TEXT, opp_deck TEXT, opp_type TEXT, kind TEXT DEFAULT 'duel')",
        )
        .unwrap();
        for (i, r) in rows.iter().enumerate() {
            con.execute(
                "INSERT INTO matches (id, created, mode, me, first, won, my_deck, opp_deck, opp_type) \
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                rusqlite::params![format!("g{i}"), r.7, r.6, r.0, r.1, r.2, r.3, r.4, r.5],
            )
            .unwrap();
        }
        path
    }

    fn fixture() -> std::path::PathBuf {
        index(&[
            (0, 0, 1, "Blue-Eyes", "Branded", "bot", "practice", "2026-10-30T10:00:00Z"),
            (1, 0, 1, "Blue-Eyes", "Branded", "bot", "practice", "2026-11-02T10:00:00Z"),
            (0, 0, 0, "Blue-Eyes", "Lunalight", "bot", "product", "2026-11-03T10:00:00Z"),
            (1, 0, 0, "Branded", "Blue-Eyes", "bot", "product", "2026-11-04T10:00:00Z"),
            (0, 0, -1, "Branded", "Blue-Eyes", "bot", "product", "2026-11-05T10:00:00Z"),
            (-1, 0, -1, "", "", "bot", "practice", "2026-11-06T10:00:00Z"),
        ])
    }

    #[test]
    fn counts_only_our_games_and_decided_win_rate() {
        let out = my_stats(&json!({}), &fixture()).unwrap();
        let s = &out["summary"];
        assert_eq!((s["games"].as_u64(), s["wins"].as_u64(), s["losses"].as_u64()), (Some(5), Some(2), Some(2)));
        assert_eq!(s["unknown"], 1);
        assert_eq!(s["win_rate"], 50.0);
        assert_eq!(s["opponents_counted"], json!(["bot"]));
        assert_eq!(out["ref"]["records"][0], "g4", "newest first");
    }

    #[test]
    fn groups_by_going_first_from_our_seat() {
        let out = my_stats(&json!({"group_by": "first"}), &fixture()).unwrap();
        let g = &out["summary"]["groups"];
        // first = 0 always: seat 0 went first, so me 0 = first, me 1 = second.
        assert_eq!(g[0]["key"], "first");
        assert_eq!((g[0]["games"].as_u64(), g[0]["wins"].as_u64()), (Some(3), Some(1)));
        assert_eq!(g[0]["win_rate"], 50.0);
        assert_eq!(g[1]["key"], "second");
        assert_eq!((g[1]["games"].as_u64(), g[1]["win_rate"].as_f64()), (Some(2), Some(50.0)));
    }

    #[test]
    fn filters_and_other_groups() {
        let p = fixture();
        let s = my_stats(&json!({"deck": "blue-eyes"}), &p).unwrap()["summary"].clone();
        assert_eq!((s["games"].as_u64(), s["win_rate"].as_f64()), (Some(3), Some(66.7)));
        let s = my_stats(&json!({"since": "2026-11-03", "mode": "product"}), &p).unwrap()["summary"].clone();
        assert_eq!(s["games"], 3);
        // Unpadded dates and mixed case are what models send (review 2026-11-14).
        let s = my_stats(&json!({"since": "2026-11-3", "opponent": "Bot"}), &p).unwrap()["summary"].clone();
        assert_eq!(s["games"], 3);
        assert!(s["note"].is_string(), "bot games only: the note is there");
        let g = my_stats(&json!({"group_by": "First"}), &p).unwrap()["summary"]["groups"].clone();
        assert_eq!(g[0]["key"], "first");
        let s = my_stats(&json!({"opponent": "remote"}), &p).unwrap()["summary"].clone();
        assert_eq!((s["games"].as_u64(), s["win_rate"].clone()), (Some(0), Value::Null));
        let g = my_stats(&json!({"group_by": "month"}), &p).unwrap()["summary"]["groups"].clone();
        assert_eq!((g[0]["key"].as_str(), g[0]["games"].as_u64()), (Some("2026-10"), Some(1)));
        assert_eq!((g[1]["key"].as_str(), g[1]["games"].as_u64()), (Some("2026-11"), Some(4)));
        let g = my_stats(&json!({"group_by": "opponent_deck"}), &p).unwrap()["summary"]["groups"].clone();
        assert_eq!(g.as_array().unwrap().len(), 3);
    }

    #[test]
    fn practice_and_puzzles_count_only_when_asked() {
        let p = fixture();
        let con = rusqlite::Connection::open(&p).unwrap();
        con.execute_batch("UPDATE matches SET kind = 'practice' WHERE id = 'g0'; UPDATE matches SET kind = 'puzzle' WHERE id = 'g1'")
            .unwrap();
        let s = my_stats(&json!({}), &p).unwrap()["summary"].clone();
        assert_eq!((s["games"].as_u64(), s["kind_counted"].as_str()), (Some(3), Some("duel")));
        assert_eq!(my_stats(&json!({"kind": "practice"}), &p).unwrap()["summary"]["games"], 1);
        assert_eq!(my_stats(&json!({"kind": "all"}), &p).unwrap()["summary"]["games"], 5);
        let g = my_stats(&json!({"kind": "all", "group_by": "kind"}), &p).unwrap()["summary"]["groups"].clone();
        assert_eq!(g.as_array().unwrap().len(), 3);
        assert!(my_stats(&json!({"kind": "casual"}), &p).unwrap_err().contains("`kind` must be one of"));
    }

    #[test]
    fn empty_or_missing_history_is_zero_not_an_error() {
        // With a game against a person counted, the "bot games only" note goes.
        let p = index(&[(0, 0, 1, "A", "B", "remote", "online", "2026-11-01T10:00:00Z")]);
        assert!(my_stats(&json!({}), &p).unwrap()["summary"].get("note").is_none());
        let missing = std::env::temp_dir().join("dl-stats-none/index.sqlite");
        let s = my_stats(&json!({}), &missing).unwrap()["summary"].clone();
        assert_eq!((s["games"].as_u64(), s["win_rate"].clone()), (Some(0), Value::Null));
    }

    #[test]
    fn bad_arguments_are_errors() {
        let p = fixture();
        for (args, want) in [
            (json!({"group_by": "colour"}), "`group_by` must"),
            (json!({"opponent": "human"}), "`opponent` must"),
            (json!({"since": "last week"}), "`since` must"),
            (json!({"deck": 3}), "`deck` must"),
            (json!({"decks": "x"}), "unknown argument"),
            (json!([1]), "must be an object"),
        ] {
            let e = my_stats(&args, &p).unwrap_err();
            assert!(e.contains(want), "{args}: {e}");
        }
    }
}
