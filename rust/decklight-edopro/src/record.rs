//! The subset of match record v1 (ADR 0010) the `.yrpX` importer needs, byte-for-byte with
//! `src/decklight/play/match_record.py`.

use serde_json::Value;
use ygoenv_sys::{DuelConfig, Engine, Env, Obs};

use crate::inverse;

pub use decklight_record::{
    FORMAT, RecordEnv, RecordGame, ResponseRec, deck_hash, engine_info, save_gz, seed_words, sha256_hex,
};

/// RFC 9562 UUIDv7 with `ms` as the 48-bit timestamp (the removed `.yrpX` import keeps it).
pub fn uuid7(ms: u64) -> String {
    decklight_record::uuid7_at(ms)
}

/// The `DuelConfig` that replays a `full` record exactly (stored seed state, no reshuffle).
pub fn config_of(rec: &Value) -> Result<DuelConfig, String> {
    if rec.get("kind").and_then(Value::as_str) != Some("full") {
        return Err(format!("only full records replay (kind={:?})", rec.get("kind")));
    }
    let decks = rec["decks"].as_array().ok_or("record has no decks")?;
    if decks.len() != 2 {
        return Err(format!("record has {} decks, not 2", decks.len()));
    }
    let codes = |d: &Value, k: &str| -> Vec<u32> {
        d[k].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect()).unwrap_or_default()
    };
    let rules = &rec["rules"];
    if rules.get("extra_cards").and_then(Value::as_array).is_some_and(|a| !a.is_empty()) {
        return Err("extra-rule cards (EDOPro extra rules) are not supported by our engine yet".into());
    }
    let seeds = rec["engine_seed"].as_array().ok_or("record has no engine_seed")?;
    if seeds.len() != 4 {
        return Err(format!("engine_seed has {} words, not 4", seeds.len()));
    }
    let mut seed_state = [0u64; 4];
    for (i, s) in seeds.iter().enumerate() {
        seed_state[i] = s
            .as_str()
            .and_then(|x| x.parse().ok())
            .or_else(|| s.as_u64())
            .ok_or_else(|| format!("bad engine_seed word {i}: {s:?}"))?;
    }
    let order = rec.get("add_order").and_then(Value::as_array);
    let team_order = match order {
        Some(a) if a.len() == 2 => [
            a[0].as_u64().unwrap_or(0) as u8,
            a[1].as_u64().unwrap_or(1) as u8,
        ],
        _ => [0, 1],
    };
    let n = |k: &str| rules.get(k).and_then(Value::as_u64).unwrap_or(0) as u32;
    Ok(DuelConfig {
        main0: codes(&decks[0], "main"),
        extra0: codes(&decks[0], "extra"),
        main1: codes(&decks[1], "main"),
        extra1: codes(&decks[1], "extra"),
        seed: 0,
        flags: rules
            .get("duel_flags")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(ygoenv_sys::TCG_FLAGS),
        lp: n("start_lp"),
        hand: n("start_hand"),
        draw: n("draw_count"),
        seed_state: Some(seed_state),
        team_order,
        edopro_add: deck_add_edopro(rec)?,
    })
}

/// `deck_add`: absent = ours (list index as the sequence), "edopro" = every main-deck card at sequence 0.
fn deck_add_edopro(rec: &Value) -> Result<bool, String> {
    match rec.get("deck_add") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::String(s)) if s == "edopro" => Ok(true),
        Some(v) => Err(format!("unknown deck_add {v}")),
    }
}

pub struct ReplayOut {
    pub trace: Vec<Vec<u8>>,
    pub winner: Option<i32>,
    pub win_reason: i32,
    pub turn: i32,
    /// The player of each response's select message.
    pub actors: Vec<i32>,
    /// The turn each response was given in.
    pub turns: Vec<i32>,
}

/// Replay raw responses through the core via `Env` + the response inverse (no raw-Duel ABI needed).
pub fn replay_responses(
    engine: &Engine,
    rec: &Value,
    responses: &[Vec<u8>],
) -> Result<ReplayOut, String> {
    let cfg = config_of(rec)?;
    let mut env = Env::new(engine, &cfg, false)?;
    env.set_record_trace(true);
    let mut obs = Obs::zeroed();
    let mut pending = env.reset(&mut obs)?;
    let mut actors = Vec::with_capacity(responses.len());
    let mut turns = Vec::with_capacity(responses.len());
    for (i, resp) in responses.iter().enumerate() {
        if !pending {
            return Err(format!("duel ended after {i} of {} responses", responses.len()));
        }
        let dec = env::decision_of(&env)?;
        actors.push(dec.player());
        turns.push(env.duel_state()?.turn);
        let seq = inverse::choices_for_response(&dec, resp, false)
            .ok_or_else(|| format!("response {i} is not expressible in the env's action space"))?;
        for idx in seq {
            pending = env.step(idx as i32, &mut obs)?;
        }
    }
    let trace = (0..env.trace_len()).map(|i| env.trace_msg(i)).collect();
    let state = env.duel_state()?;
    Ok(ReplayOut {
        trace,
        winner: (state.winner == 0 || state.winner == 1).then_some(state.winner),
        win_reason: state.win_reason,
        turn: state.turn,
        actors,
        turns,
    })
}

/// `match_record.check`: problems found by replaying a record (empty = it reproduces).
pub fn check_record(engine: &Engine, rec: &Value) -> Vec<String> {
    let responses: Vec<Vec<u8>> = match rec["responses"].as_array() {
        Some(a) => a.iter().map(|e| hex_decode(e["r"].as_str().unwrap_or(""))).collect(),
        None => return vec!["record has no responses".into()],
    };
    let out = match replay_responses(engine, rec, &responses) {
        Ok(o) => o,
        Err(e) => return vec![format!("replay failed: {e}")],
    };
    let res = &rec["result"];
    let finished = !(res["surrendered"].as_bool().unwrap_or(false) || res["disconnected"].as_bool().unwrap_or(false));
    let want = res["winner"].as_i64();
    let got = out.winner.map(|w| w as i64);
    let mut problems = Vec::new();
    if finished && got != want {
        problems.push(format!("winner {:?} != recorded {:?}", got, want));
    }
    if finished && out.winner.is_none() {
        problems.push("replay did not reach the end of the duel".into());
    }
    if out.turn as i64 != res["turns"].as_i64().unwrap_or(0) {
        problems.push(format!("turns {} != recorded {}", out.turn, res["turns"]));
    }
    problems
}

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0))
        .collect()
}

/// Small helper so record.rs does not depend on `Decision::from_env`'s Option juggling.
mod env {
    use ygoenv_sys::{Decision, Env};

    pub fn decision_of(env: &Env<'_>) -> Result<Decision, String> {
        Decision::from_env(env)?.ok_or_else(|| "duel is awaiting a response but sent no select message".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_hash_matches_python_rule() {
        // sha256(",".join(sorted(main)+[0]+sorted(extra)))
        let h = deck_hash(&[3, 1, 2], &[9, 8]);
        let want = sha256_hex(b"1,2,3,0,8,9");
        assert_eq!(h, want);
    }
}
