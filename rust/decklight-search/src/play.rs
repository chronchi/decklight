//! Port of `decklight.search.play.search_game_gen` for the product path.
//!
//! Honest search + `ho` + `det_fallback="skip"` + `conf_skip`, prefix replay, `guard_mode`
//! shared/per_seat and `unguarded_seat`, latency timings. Errors (NaN logits, an `ho` pick not among the
//! candidates, replay errors) are `Err`, never panics.

use std::collections::BTreeMap;
use std::time::Instant;

use ygoenv_sys::{DuelConfig, Env, Obs};

use crate::determinize::{key_rank_of_choice, Determinizer, SeenTracker, WorldConfig};
use crate::honest::{decide, decide_pair, semantic_key, HonestRule};
use crate::loopguard::{LoopGuard, StateKey};
use crate::pyrandom::Random;
use crate::select::{argmax_select, masked, softmax_conf};
use crate::window::{is_window, Branch, Nets, WindowSearch, World};

/// The game loop also needs the human seat's one-hot choice (as `ProductDriver._forward` routes it).
pub trait PlayerNets: Nets {
    /// The unguarded seat's next choice (a one-hot logit row is built from it).
    fn human(&mut self, obs: &Obs) -> Result<i32, String>;

    /// One live (multi-option) decision, in decision order, before it is answered. The EDOPro host mirrors this
    /// log to check lockstep with its own env (`ProductDriver.decision`).
    fn decision(&mut self, _obs: &Obs, _actor: i32) -> Result<(), String> {
        Ok(())
    }

    /// The searching seat's chosen action, in decision order (`ProductDriver.played`'s bot entries).
    fn bot_choice(&mut self, _obs: &Obs, _choice: i32) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HoLog {
    pub status: &'static str,
    pub gap: f64,
    pub se: Option<f64>,
    pub n: usize,
    pub challenger: Option<i32>,
    pub rounds: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogEntry {
    pub k: usize,
    pub seat: i32,
    pub kind: &'static str,
    pub turn: i32,
    pub cands: Vec<i32>,
    pub scores: Vec<f64>,
    pub scores_all: BTreeMap<String, Vec<f64>>,
    pub chosen: i32,
    pub played: i32,
    pub best: i32,
    pub samples: Vec<Vec<f64>>,
    pub ho: Option<HoLog>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRec {
    pub k: usize,
    pub actor: i32,
    pub turn: i32,
    pub n_options: usize,
    pub chosen: i32,
    pub searched: bool,
}

#[derive(Clone, Debug, Default)]
pub struct GameResult {
    pub winner: Option<i32>,
    pub truncated: bool,
    pub turns: i32,
    pub choices: Vec<i32>,
    pub actors: Vec<i32>,
    pub n_searched: usize,
    pub n_override: usize,
    pub search_s: f64,
    pub n_branches: usize,
    pub n_capped: usize,
    pub searched_idx: Vec<usize>,
    pub det_proposals: usize,
    pub det_accepted: usize,
    pub det_fallbacks: usize,
    pub det_skipped: usize,
    pub det_s: f64,
    /// The part of `det_s` spent on decisions skipped for lack of worlds (not searched moves).
    pub det_skip_s: f64,
    pub choice_keys: Vec<([i32; 8], i32)>,
    pub log: Vec<LogEntry>,
    pub decision_turns: Vec<i32>,
    pub search_cut: bool,
    pub decisions: Vec<DecisionRec>,
    pub lat_raw: Vec<f64>,
    pub lat_search: Vec<f64>,
    pub lat_turn: Vec<f64>,
}

/// The product's `search_game_gen` settings (`product_search_kwargs`).
#[derive(Clone, Debug)]
pub struct PlayParams {
    pub s_seat: i32,
    pub game_seed: u64,
    pub own_turn1: bool,
    pub conf_skip: f64,
    pub own_frac: f64,
    pub max_decisions: usize,
    pub auto_single: bool,
    pub honest: bool,
    pub det_tries: usize,
    pub det_fallback: String,
    /// Rollout forward batch split across this many inference workers (1 = today's single batched call).
    pub search_threads: usize,
    pub prefix: Vec<i32>,
    pub guard_mode: String,
    pub unguarded_seat: Option<i32>,
    pub margin: f64,
    pub max_search_turn: i32,
    pub max_searches: usize,
    pub ho: Option<HonestRule>,
    pub latency: bool,
}

impl PlayParams {
    pub fn product(s_seat: i32, game_seed: u64) -> PlayParams {
        PlayParams {
            s_seat,
            game_seed,
            own_turn1: false,
            conf_skip: 0.9,
            own_frac: 1.0,
            max_decisions: 3000,
            auto_single: true,
            honest: true,
            det_tries: 32,
            det_fallback: "skip".to_string(),
            search_threads: 1,
            prefix: Vec::new(),
            guard_mode: "per_seat".to_string(),
            unguarded_seat: Some(1 - s_seat),
            margin: 0.0,
            max_search_turn: 0,
            max_searches: 0,
            ho: Some(HonestRule::default()),
            latency: true,
        }
    }
}

fn n_legal(obs: &Obs) -> usize {
    obs.action_mask.iter().filter(|&&x| x != 0).count()
}

fn world_config(c: &DuelConfig) -> WorldConfig {
    WorldConfig {
        main0: c.main0.clone(),
        extra0: c.extra0.clone(),
        main1: c.main1.clone(),
        extra1: c.extra1.clone(),
        seed: c.seed,
        flags: c.flags,
        lp: c.lp,
        hand: c.hand,
        draw: c.draw,
        seed_state: c.seed_state,
        team_order: c.team_order,
        edopro_add: c.edopro_add,
    }
}

/// The game loop's NaN gate: any NaN in the masked logits is an `Err`, never a panic.
pub fn check_nan(z: &[f64], decision: usize, game_seed: u64) -> Result<(), String> {
    if z.iter().any(|v| v.is_nan()) {
        return Err(format!(
            "ALERT search_match: NaN logits at decision {decision} (game_seed={game_seed})"
        ));
    }
    Ok(())
}

/// `round(x, 4)` (CPython's decimal round-half-even); used for the log's rounded fields.
pub fn round4(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.4}").parse::<f64>().unwrap_or(x)
}

fn choice_row(obs: &Obs, idx: usize) -> Vec<f32> {
    let mut row = vec![0f32; obs.action_mask.len()];
    row[idx] = 1.0;
    row
}

/// Play one product game; S = `params.s_seat` searches, R = `params.unguarded_seat` answers through `nets.human`.
pub fn search_game<N: PlayerNets>(
    search: &mut WindowSearch<'_>,
    config: &DuelConfig,
    nets: &mut N,
    params: &PlayParams,
) -> Result<GameResult, String> {
    if params.guard_mode != "shared" && params.guard_mode != "per_seat" {
        return Err(format!(
            "guard_mode must be 'shared' or 'per_seat', got {:?}",
            params.guard_mode
        ));
    }
    if params.det_fallback != "true" && params.det_fallback != "skip" {
        return Err(format!(
            "det_fallback must be 'true' or 'skip', got {:?}",
            params.det_fallback
        ));
    }
    search.search_threads = params.search_threads.max(1);
    let seed = search.seed;
    let engine = search.engine;
    let s_seat = params.s_seat;
    let game_seed = params.game_seed;
    let mut rng = Random::from_i128((seed as i128) * 1_000_003 + game_seed as i128);
    let mut env = Env::new(search.engine, config, params.auto_single)?;
    let mut raw_obs = Obs::zeroed();
    let pending = env.reset(&mut raw_obs)?;
    let mut obs: Option<Obs> = if pending { Some(*raw_obs) } else { None };
    let mut tracker: Option<SeenTracker> =
        if params.honest { Some(SeenTracker::new(s_seat as u8)) } else { None };
    if let Some(t) = tracker.as_mut() {
        t.update(&env);
    }
    let mut guard = LoopGuard::new();
    let mut res = GameResult { turns: 1, ..Default::default() };
    let capped0 = search.capped;
    let mut lat_open_turn: Option<i32> = None;
    let mut lat_open_start: Option<Instant> = None;
    let mut lat_last_end: Option<Instant> = None;

    for &c in &params.prefix {
        let Some(o) = obs.as_ref() else {
            return Err("search_game: game ended inside the prefix".into());
        };
        let n = n_legal(o);
        if c < 0 || c as usize >= n {
            return Err(format!("search_game: prefix choice {c} illegal with {n} options"));
        }
        let me = env.player();
        res.decisions.push(DecisionRec {
            k: res.choices.len(),
            actor: me,
            turn: o.global[2],
            n_options: n,
            chosen: c,
            searched: false,
        });
        res.choices.push(c);
        res.actors.push(me);
        res.decision_turns.push(o.global[2]);
        if let Some(t) = tracker.as_ref() {
            res.choice_keys.push(key_rank_of_choice(o, c as usize));
            let _ = t;
        }
        let mut next = Obs::zeroed();
        let pending = env.step(c, &mut next)?;
        obs = if pending { Some(*next) } else { None };
        if let Some(t) = tracker.as_mut() {
            t.update(&env);
        }
    }

    while let Some(o) = obs.clone() {
        if res.choices.len() >= params.max_decisions {
            break;
        }
        let me = env.player();
        let turn = o.global[2];
        res.turns = turn;
        if std::env::var("YGO_PLAY_TRACE").is_ok() {
            eprintln!("play: k={} me={} turn={} n={}", res.choices.len(), me, turn, n_legal(&o));
        }
        let t_dec = if params.latency { Some(Instant::now()) } else { None };
        let n_log0 = res.log.len();
        let is_own = params.latency && me == s_seat && o.global[5] == 1;
        let n = n_legal(&o);
        if n == 0 {
            return Err(format!("search_game: no legal option at decision {}", res.choices.len()));
        }
        nets.decision(&o, me)?;
        let row = if Some(me) == params.unguarded_seat {
            let h = nets.human(&o)?;
            if h < 0 || h as usize >= n {
                return Err(format!("human returned illegal index {h}"));
            }
            choice_row(&o, h as usize)
        } else {
            let rows = nets.forward(&[&o])?;
            rows.into_iter().next().ok_or("search_game: no outer logits")?
        };
        let mask: Vec<bool> = o.action_mask.iter().map(|&x| x != 0).collect();
        let z = masked(&row, &mask)?;
        check_nan(&z, res.choices.len(), game_seed)?;
        let gslot = if params.guard_mode == "per_seat" { me } else { 0 };
        let guarded = Some(me) != params.unguarded_seat;
        let (z_g, key): (Vec<f64>, Option<StateKey>) = if guarded {
            let (zz, k) = guard.mask(gslot, &o, &z);
            (zz, Some(k))
        } else {
            (z, None)
        };
        let a_pol = argmax_select(&z_g).ok_or("search_game: no legal option after guard")? as i32;
        let mut a = a_pol;
        if me == s_seat {
            let window = is_window(&o, me);
            let own = o.global[5] == 1
                && n >= 2
                && turn >= if params.own_turn1 { 1 } else { 2 }
                && rng.random() < params.own_frac;
            if params.max_searches > 0 && res.n_searched + res.det_skipped >= params.max_searches {
                res.search_cut = true;
            } else if (window || own)
                && softmax_conf(&z_g) < params.conf_skip
                && !(params.max_search_turn > 0 && turn > params.max_search_turn)
            {
                let cands: Vec<i32> = search
                    .candidates(&z_g, &o, window, params.ho.as_ref().is_some_and(|h| h.merge))
                    .into_iter()
                    .map(|x| x as i32)
                    .collect();
                if cands.len() >= 2 {
                    if std::env::var("YGO_PLAY_TRACE").is_ok() {
                        eprintln!("play: search k={} cands={cands:?}", res.choices.len());
                    }
                    let t0 = Instant::now();
                    let k = res.choices.len();
                    let mut branches: Vec<Branch> = Vec::with_capacity(cands.len() * search.samples);
                    for (j, &x) in cands.iter().enumerate() {
                        for s in 0..search.samples {
                            branches.push(Branch {
                                prefix: res.choices.iter().copied().chain(std::iter::once(x)).collect(),
                                cand_idx: j,
                                cand_action: x,
                                sample: s,
                            });
                        }
                    }
                    let mut worlds: Option<Vec<World>> = None;
                    let mut det: Option<Determinizer<'_>> = None;
                    let mut skip_search = false;
                    let t_det = Instant::now();
                    if params.honest {
                        let seen = tracker.as_ref().ok_or("honest search without a tracker")?.seen.clone();
                        let det_rng = Random::from_str(&format!("{seed}:{game_seed}:{k}:det"));
                        let prefix_keys = res.choice_keys.clone();
                        det = Some(Determinizer::new(
                            engine,
                            world_config(config),
                            params.auto_single,
                            me as u8,
                            &res.choices,
                            &seen,
                            &o,
                            det_rng,
                            params.det_tries,
                            Some(&prefix_keys),
                        )?);
                        let d = det.as_mut().unwrap();
                        d.replay_threads = search.replay_threads;
                        let mut ws: Vec<World> = Vec::new();
                        for got in d.sample_many(search.samples) {
                            match got {
                                Some(r) => ws.push((r.world.to_env_config(), r.resolved)),
                                None => {
                                    if params.det_fallback == "true" {
                                        ws.push((config.clone(), res.choices.clone()));
                                    }
                                }
                            }
                        }
                        res.det_proposals += d.proposals;
                        res.det_accepted += d.accepted;
                        res.det_fallbacks += d.fallbacks;
                        res.det_s += t_det.elapsed().as_secs_f64();
                        if std::env::var("YGO_PLAY_TRACE").is_ok() {
                            eprintln!(
                                "play: det k={} worlds={} prop={} acc={} fb={} reasons no_key={} replay_error={} obs_mismatch={} t={:.2}s",
                                k, worlds.as_ref().map_or(0, |w| w.len()), d.proposals, d.accepted, d.fallbacks,
                                d.reasons.no_key, d.reasons.replay_error, d.reasons.obs_mismatch,
                                t_det.elapsed().as_secs_f64()
                            );
                        }
                        if params.det_fallback == "skip" && ws.is_empty() {
                            res.det_skipped += 1;
                            res.det_skip_s += t_det.elapsed().as_secs_f64();
                            skip_search = true;
                        }
                        worlds = Some(ws);
                    }
                    if !skip_search {
                        let mut out = search.rollouts_gen(
                            config,
                            params.auto_single,
                            &branches,
                            me,
                            turn,
                            game_seed,
                            k,
                            worlds.as_deref(),
                            nets,
                        )?;
                        res.search_s += t0.elapsed().as_secs_f64();
                        let sel_key = search.select_key();
                        let sel_scores = out
                            .scores
                            .get(&sel_key)
                            .ok_or_else(|| format!("search: no selection key {sel_key}"))?
                            .clone();
                        let mut best = 0usize;
                        for j in 1..cands.len() {
                            if sel_scores[j] > sel_scores[best] {
                                best = j;
                            }
                        }
                        if std::env::var("YGO_PLAY_TRACE").is_ok() {
                            eprintln!("play: rollouts k={} scores={:?}", k, sel_scores);
                        }
                        let mut chosen = cands[best];
                        let mut ho_log: Option<HoLog> = None;
                        let mut samples_for_log = out.last_samples.clone();
                        if let Some(ho) = &params.ho {
                            let mut samples = out.last_samples.clone();
                            let kp = semantic_key(&o, a_pol as usize, None);
                            let pick_j = cands
                                .iter()
                                .position(|&x| semantic_key(&o, x as usize, None) == kp)
                                .ok_or_else(|| {
                                    format!(
                                        "ALERT search_match: ho pick not among candidates (game_seed={game_seed}, k={k})"
                                    )
                                })?;
                            if samples.len() != cands.len() {
                                return Err(format!(
                                    "ALERT search_match: ho samples/candidates mismatch (game_seed={game_seed}, k={k})"
                                ));
                            }
                            let t_ho = Instant::now();
                            let mut dec = decide(&samples, pick_j, ho);
                            let mut rounds = 0usize;
                            while dec.status == "undecided" && dec.n < ho.s_max {
                                let n0 = dec.n;
                                let extra = ho.step.min(ho.s_max - dec.n);
                                let challenger = dec.challenger_j.ok_or_else(|| {
                                    format!(
                                        "ALERT search_match: undecided ho without a challenger (game_seed={game_seed}, k={k})"
                                    )
                                })?;
                                let pair = [pick_j, challenger];
                                if let (Some(d), Some(ws)) = (det.as_mut(), worlds.as_mut()) {
                                    let (p0, a0, f0) = (d.proposals, d.accepted, d.fallbacks);
                                    let t_d = Instant::now();
                                    d.replay_threads = search.replay_threads;
                                    for got in d.sample_many(extra) {
                                        match got {
                                            Some(r) => ws.push((r.world.to_env_config(), r.resolved)),
                                            None => {
                                                if params.det_fallback == "true" {
                                                    ws.push((config.clone(), res.choices.clone()));
                                                }
                                            }
                                        }
                                    }
                                    res.det_proposals += d.proposals - p0;
                                    res.det_accepted += d.accepted - a0;
                                    res.det_fallbacks += d.fallbacks - f0;
                                    res.det_s += t_d.elapsed().as_secs_f64();
                                }
                                let mut xb: Vec<Branch> = Vec::new();
                                for &j in &pair {
                                    for s in n0..n0 + extra {
                                        xb.push(Branch {
                                            prefix: res
                                                .choices
                                                .iter()
                                                .copied()
                                                .chain(std::iter::once(cands[j]))
                                                .collect(),
                                            cand_idx: j,
                                            cand_action: cands[j],
                                            sample: s,
                                        });
                                    }
                                }
                                let out2 = search.rollouts_gen(
                                    config,
                                    params.auto_single,
                                    &xb,
                                    me,
                                    turn,
                                    game_seed,
                                    k,
                                    worlds.as_deref(),
                                    nets,
                                )?;
                                for &j in &pair {
                                    samples[j].extend_from_slice(&out2.last_samples[j]);
                                }
                                res.n_branches += xb.len();
                                rounds += 1;
                                dec = decide_pair(&samples, pick_j, challenger, ho);
                            }
                            chosen = if dec.chosen_j != pick_j { cands[dec.chosen_j] } else { a_pol };
                            ho_log = Some(HoLog {
                                status: dec.status,
                                gap: round4(dec.gap),
                                se: if dec.se.is_infinite() { None } else { Some(round4(dec.se)) },
                                n: dec.n,
                                challenger: dec.challenger_j.map(|j| cands[j]),
                                rounds,
                            });
                            samples_for_log = samples;
                            res.search_s += t_ho.elapsed().as_secs_f64();
                        } else if params.margin > 0.0 && chosen != a_pol {
                            if let Some(idx) = cands.iter().position(|&x| x == a_pol) {
                                if sel_scores[best] - sel_scores[idx] < params.margin {
                                    chosen = a_pol;
                                }
                            }
                        }
                        res.n_searched += 1;
                        res.n_branches += branches.len();
                        res.searched_idx.push(k);
                        res.log.push(LogEntry {
                            k,
                            seat: me,
                            kind: if window { "window" } else { "own" },
                            turn,
                            cands: cands.iter().map(|&x| x as i32).collect(),
                            scores: sel_scores,
                            scores_all: std::mem::take(&mut out.scores),
                            chosen,
                            played: a_pol,
                            best: cands[best],
                            samples: samples_for_log,
                            ho: ho_log,
                        });
                        a = chosen;
                        if a != a_pol {
                            res.n_override += 1;
                        }
                    }
                }
            }
        }
        if guarded {
            if let Some(k) = key {
                guard.note(gslot, k, a);
            }
        }
        if let Some(t) = tracker.as_ref() {
            let _ = t;
            res.choice_keys.push(key_rank_of_choice(&o, a as usize));
        }
        if me == s_seat {
            nets.bot_choice(&o, a)?;
        }
        res.decisions.push(DecisionRec {
            k: res.choices.len(),
            actor: me,
            turn,
            n_options: n,
            chosen: a,
            searched: res.log.len() > n_log0,
        });
        res.choices.push(a);
        res.actors.push(me);
        res.decision_turns.push(turn);
        let mut next = Obs::zeroed();
        let pending = env.step(a, &mut next)?;
        obs = if pending { Some(*next) } else { None };
        if let Some(t) = tracker.as_mut() {
            t.update(&env);
        }
        if let Some(t_dec) = t_dec {
            let w_end = Instant::now();
            let secs = w_end.duration_since(t_dec).as_secs_f64();
            let target = if res.log.len() > n_log0 { &mut res.lat_search } else { &mut res.lat_raw };
            // Speed checks only (DECKLIGHT_PLAY_TIMING): one line per searched move so a client can tell a bot
            // move that searched from one that did not, and time them apart. Never affects a choice.
            if res.log.len() > n_log0 && std::env::var_os("DECKLIGHT_PLAY_TIMING").is_some() {
                eprintln!("product: search k={} t={:.4}", res.choices.len(), secs);
            }
            target.push(round4(secs));
            if is_own {
                if lat_open_turn != Some(turn) {
                    if let (Some(a), Some(b)) = (lat_open_start, lat_last_end) {
                        res.lat_turn.push(round4((b - a).as_secs_f64()));
                    }
                    lat_open_turn = Some(turn);
                    lat_open_start = Some(t_dec);
                }
                lat_last_end = Some(w_end);
            }
        }
    }
    if params.latency && lat_open_turn.is_some() {
        if let (Some(a), Some(b)) = (lat_open_start, lat_last_end) {
            res.lat_turn.push(round4((b - a).as_secs_f64()));
        }
    }
    res.truncated = obs.is_some();
    res.winner = if res.truncated { None } else { Some(env.winner()) };
    res.n_capped = search.capped - capped0;
    Ok(res)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_logits_are_an_error() {
        assert!(check_nan(&[0.0, f64::NAN], 7, 11).is_err());
        assert!(check_nan(&[0.0, f64::NEG_INFINITY], 7, 11).is_ok());
    }

    #[test]
    fn round4_matches_python_examples() {
        assert_eq!(round4(1.23456), 1.2346);
        assert_eq!(round4(-1.23445), -1.2345);
        assert_eq!(round4(0.0), 0.0);
        assert_eq!(round4(f64::INFINITY), f64::INFINITY);
    }
}
