//! Port of `decklight.search.window.WindowSearch` for the product path.
//!
//! The Python generator protocol becomes a [`Nets`] callback: the search calls [`Nets::forward`] with exactly the
//! observation list Python yields, in the same order, and [`Nets::value`] with exactly the `need` list Python builds.
//! Prefix replays only (`YGO_SEARCH_CLONE=0`, the default C++ core), so no cloning.

use std::collections::BTreeMap;
use std::time::Instant;

use ygoenv_sys::{DuelConfig, Engine, Env, Obs};

use crate::honest::merge_candidates;
use crate::loopguard::LoopGuard;
use crate::pyrandom::Random;
use crate::select::{argmax_select, masked, pairwise_sum};

pub const BOARD_WIN: f64 = 50.0;
pub const VALUE_WIN: f64 = 1.0;
pub const KIND_ACTIVATE: i32 = 6;
pub const KIND_PASS: i32 = 11;
pub const ACTION_COLS: usize = 10;
pub const MSG_SELECT_CHAIN: i32 = 0x10;

/// The policy/value forward used by the search (rollout observations only).
pub trait Nets {
    /// `_forward`: policy logits for exactly this list, in order (row width >= the action-mask length).
    fn forward(&mut self, obs: &[&Obs]) -> Result<Vec<Vec<f32>>, String>;
    /// The learned value for exactly this list, in order (actor's view, clipped to [-1, 1] by the caller).
    fn value(&mut self, obs: &[&Obs]) -> Result<Vec<f32>, String>;

    /// `forward` with the batch optionally split across `threads` inference workers, rows in the same order.
    /// The default ignores `threads` and is exactly `forward`; implementations may only split when every row's
    /// output is unchanged (the product nets' graphs are batch-independent).
    fn forward_threads(&mut self, obs: &[&Obs], threads: usize) -> Result<Vec<Vec<f32>>, String> {
        let _ = threads;
        self.forward(obs)
    }
}

/// A world used by honest search: the determinized config and its resolved prefix.
pub type World = (DuelConfig, Vec<i32>);

/// One rollout branch: prefix (root choices + candidate), candidate index, its action, sample number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Branch {
    pub prefix: Vec<i32>,
    pub cand_idx: usize,
    pub cand_action: i32,
    pub sample: usize,
}

/// `rollouts_gen`'s return: the averaged scores plus the per-sample selection-key lists.
#[derive(Clone, Debug)]
pub struct RolloutOut {
    pub scores: BTreeMap<String, Vec<f64>>,
    pub last_samples: Vec<Vec<f64>>,
    pub last_samples_all: BTreeMap<String, Vec<Vec<f64>>>,
}

/// Score for player `me` from an obs seen by player `actor` (absolute seats).
pub fn position_score(obs: &Obs, actor: i32, me: i32) -> f64 {
    let g = &obs.global;
    let (mine, theirs) = if actor == me { (6, 13) } else { (13, 6) };
    let (lp_m, lp_t) = if actor == me { (g[0], g[1]) } else { (g[1], g[0]) };
    let cm = &g[mine..mine + 7];
    let ct = &g[theirs..theirs + 7];
    let a = (cm[2] + cm[3]) - (ct[2] + ct[3]);
    let b = cm[1] - ct[1];
    a as f64 + 0.5 * b as f64 + (lp_m - lp_t) as f64 / 2000.0
}

/// Raw card advantage for `me`: (hand + field) mine - theirs, from an obs seen by `actor`.
pub fn cards_score(obs: &Obs, actor: i32, me: i32) -> f64 {
    let g = &obs.global;
    let (mine, theirs) = if actor == me { (7, 14) } else { (14, 7) };
    let m = g[mine] + g[mine + 1] + g[mine + 2];
    let t = g[theirs] + g[theirs + 1] + g[theirs + 2];
    (m - t) as f64
}

/// A chain window on the opponent's turn where the acting player can activate something.
pub fn is_window(obs: &Obs, _actor: i32) -> bool {
    let g = &obs.global;
    if g[21] != MSG_SELECT_CHAIN || g[5] == 1 {
        return false;
    }
    for i in 0..obs.action_mask.len() {
        if obs.action_mask[i] != 0 && obs.actions[i * ACTION_COLS] == KIND_ACTIVATE {
            return true;
        }
    }
    false
}

struct Snap {
    board: Vec<Option<f64>>,
    cards: Vec<Option<f64>>,
    done: Vec<Option<f64>>,
    actor: Vec<i32>,
    vobs: Vec<Option<Obs>>,
    vterm: Vec<Option<f64>>,
}

impl Snap {
    fn new(h: usize) -> Snap {
        Snap {
            board: vec![None; h],
            cards: vec![None; h],
            done: vec![None; h],
            actor: vec![0; h],
            vobs: vec![None; h],
            vterm: vec![None; h],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn snapshot(
    snaps: &mut [Snap],
    i: usize,
    h_idx: usize,
    terminal: bool,
    lost: bool,
    me: i32,
    winner: i32,
    player: i32,
    obs: Option<&Obs>,
) {
    let s = &mut snaps[i];
    if terminal {
        let w = if lost { 1 - me } else { winner };
        s.board[h_idx] = Some(if w == me { BOARD_WIN } else if w == 0 || w == 1 { -BOARD_WIN } else { 0.0 });
        s.cards[h_idx] = Some(0.0);
        s.done[h_idx] = Some(if w == me { 1.0 } else if w == 0 || w == 1 { -1.0 } else { 0.0 });
        s.actor[h_idx] = me;
        s.vterm[h_idx] =
            Some(if w == me { VALUE_WIN } else if w == 0 || w == 1 { -VALUE_WIN } else { 0.0 });
    } else {
        let o = obs.expect("non-terminal snapshot without obs");
        s.board[h_idx] = Some(position_score(o, player, me));
        s.cards[h_idx] = Some(cards_score(o, player, me));
        s.done[h_idx] = Some(0.0);
        s.actor[h_idx] = player;
        s.vobs[h_idx] = Some(o.clone());
    }
}

/// Divide each candidate's accumulated scores by its branch count (zero-count candidates keep their zero sum).
pub fn normalise_averages(out: &mut BTreeMap<String, Vec<f64>>, counts: &[usize]) {
    for col in out.values_mut() {
        for (j, v) in col.iter_mut().enumerate() {
            if counts.get(j).copied().unwrap_or(0) > 0 {
                *v /= counts[j] as f64;
            }
        }
    }
}

/// `rollouts_gen`'s sample walk: argmax for sample 0, `np.exp`-softmax sampling for samples >= 1.
pub fn sample_action(z: &[f64], temp: f64, rng: &mut Random) -> usize {
    let finite: Vec<bool> = z.iter().map(|v| v.is_finite()).collect();
    let legal: Vec<usize> = (0..z.len()).filter(|&i| finite[i]).collect();
    if legal.is_empty() {
        return 0;
    }
    let vals: Vec<f64> = legal.iter().map(|&i| z[i] / temp).collect();
    let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = vals.iter().map(|&v| (v - mx).exp()).collect();
    let sum = pairwise_sum(&exp);
    let probs: Vec<f64> = exp.iter().map(|&e| e / sum).collect();
    let r = rng.random();
    let mut a = *legal.last().unwrap();
    let mut cum = 0.0;
    for (&li, &p) in legal.iter().zip(probs.iter()) {
        cum += p;
        if r < cum {
            a = li;
            break;
        }
    }
    a
}

/// Opt-in timers for one search (`WindowSearch::prof`); they never change an output.
#[derive(Clone, Debug, Default)]
pub struct Prof {
    /// Branch-setup prefix replays.
    pub replay_s: f64,
    /// Rollout `env.step` calls.
    pub step_s: f64,
    pub steps: usize,
    /// Rollout policy forwards (`Nets::forward` inside `rollouts_gen`).
    pub fwd_s: f64,
    pub fwd_calls: usize,
    pub fwd_obs: usize,
    /// Learned-value forwards.
    pub value_s: f64,
    pub value_calls: usize,
    pub value_obs: usize,
    /// The whole of `rollouts_gen`; the rest = rollout_s - replay_s - step_s - fwd_s - value_s.
    pub rollout_s: f64,
}

pub struct WindowSearch<'e> {
    pub engine: &'e Engine,
    pub k: usize,
    pub horizon: usize,
    pub max_rollout: usize,
    pub seed: u64,
    pub horizons: Option<Vec<usize>>,
    pub samples: usize,
    pub rollout_temp: f64,
    pub select: Option<String>,
    pub has_value: bool,
    /// Speed plan E6: scoped threads for the branch-setup prefix replays (1 = the sequential loop, identical).
    pub replay_threads: usize,
    /// Rollout batches split across this many inference workers (1 = one batched forward, today's path).
    pub search_threads: usize,
    pub capped: usize,
    pub prof: Option<Prof>,
}

impl<'e> WindowSearch<'e> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: &'e Engine,
        k: usize,
        horizon: usize,
        samples: usize,
        rollout_temp: f64,
        max_rollout: usize,
        seed: u64,
        select: Option<String>,
        has_value: bool,
    ) -> WindowSearch<'e> {
        WindowSearch {
            engine,
            k,
            horizon,
            max_rollout,
            seed,
            horizons: None,
            samples,
            rollout_temp,
            select,
            has_value,
            replay_threads: 1,
            search_threads: 1,
            capped: 0,
            prof: None,
        }
    }

    pub fn horizons(&self) -> Vec<usize> {
        self.horizons.clone().unwrap_or_else(|| vec![self.horizon])
    }

    pub fn select_key(&self) -> String {
        if let Some(s) = &self.select {
            return s.clone();
        }
        let h0 = self.horizons()[0];
        if self.has_value { format!("value@h{h0}") } else { format!("board@h{h0}") }
    }

    /// `_candidates`: legal options by descending logit (stable), optionally merged by meaning; a chain window
    /// keeps one PASS plus the activations.
    pub fn candidates(&self, z: &[f64], obs: &Obs, window: bool, merge: bool) -> Vec<usize> {
        let mut order: Vec<usize> = (0..z.len()).filter(|&i| z[i].is_finite()).collect();
        order.sort_by(|&a, &b| z[b].partial_cmp(&z[a]).unwrap_or(std::cmp::Ordering::Equal));
        if merge {
            order = merge_candidates(&order, obs, None);
        }
        if !window {
            let n = self.k.min(order.len());
            return order[..n].to_vec();
        }
        let passes: Vec<usize> =
            order.iter().copied().filter(|&i| obs.actions[i * ACTION_COLS] == KIND_PASS).collect();
        let acts: Vec<usize> =
            order.iter().copied().filter(|&i| obs.actions[i * ACTION_COLS] == KIND_ACTIVATE).collect();
        let mut out: Vec<usize> = passes.into_iter().take(1).collect();
        out.extend(acts);
        out.truncate(self.k.max(2));
        out
    }

    /// Play each branch out with the policy until the largest horizon ends (or the duel ends / max_rollout).
    #[allow(clippy::too_many_arguments)]
    pub fn rollouts_gen(
        &mut self,
        config: &DuelConfig,
        auto_single: bool,
        branches: &[Branch],
        me: i32,
        turn_start: i32,
        game_seed: u64,
        decision_k: usize,
        worlds: Option<&[World]>,
        nets: &mut dyn Nets,
    ) -> Result<RolloutOut, String> {
        let t_all = self.prof.as_ref().map(|_| Instant::now());
        let out = self.rollouts_inner(config, auto_single, branches, me, turn_start, game_seed, decision_k, worlds, nets);
        if let (Some(p), Some(t)) = (self.prof.as_mut(), t_all) {
            p.rollout_s += t.elapsed().as_secs_f64();
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn rollouts_inner(
        &mut self,
        config: &DuelConfig,
        auto_single: bool,
        branches: &[Branch],
        me: i32,
        turn_start: i32,
        game_seed: u64,
        decision_k: usize,
        worlds: Option<&[World]>,
        nets: &mut dyn Nets,
    ) -> Result<RolloutOut, String> {
        let timed = self.prof.is_some();
        let tick = || if timed { Some(Instant::now()) } else { None };
        let hs = self.horizons();
        let hlen = hs.len();
        let b = branches.len();
        let n_cands = branches.iter().map(|x| x.cand_idx).max().map_or(0, |m| m + 1);
        let engine = self.engine;
        let mut specs: Vec<(DuelConfig, Vec<i32>)> = Vec::with_capacity(b);
        for br in branches {
            let (cfg, pp) = match worlds {
                None => (config.clone(), br.prefix.clone()),
                Some(ws) => {
                    if ws.is_empty() {
                        return Err("rollouts_gen: worlds is empty".into());
                    }
                    let (wcfg, wprefix) = &ws[br.sample % ws.len()];
                    let mut pp = wprefix.clone();
                    pp.push(br.cand_action);
                    (wcfg.clone(), pp)
                }
            };
            specs.push((cfg, pp));
        }
        let t_rep = tick();
        let pairs = crate::parallel::map(self.replay_threads, b, |i| {
            let (cfg, pp) = &specs[i];
            let mut e = Env::new(engine, cfg, auto_single)?;
            let mut o = Obs::zeroed();
            let pending = e.replay(pp, &mut o)?;
            Ok::<(Env<'e>, Option<Obs>), String>((e, if pending { Some(*o) } else { None }))
        });
        let mut envs: Vec<Env<'e>> = Vec::with_capacity(b);
        let mut obs: Vec<Option<Obs>> = Vec::with_capacity(b);
        for pair in pairs {
            let (e, o) = pair?;
            envs.push(e);
            obs.push(o);
        }
        if let (Some(p), Some(t)) = (self.prof.as_mut(), t_rep) {
            p.replay_s += t.elapsed().as_secs_f64();
        }
        let mut snaps: Vec<Snap> = (0..b).map(|_| Snap::new(hlen)).collect();
        let mut n = vec![0usize; b];
        let mut live = vec![true; b];
        let mut guard = LoopGuard::new();
        let mut rngs: Vec<Random> = branches
            .iter()
            .map(|br| Random::from_str(&format!("{}:{}:{}:{}:{}", self.seed, game_seed, decision_k, br.cand_action, br.sample)))
            .collect();

        while live.iter().any(|&x| x) {
            let mut act: Vec<usize> = Vec::new();
            for i in 0..b {
                if !live[i] {
                    continue;
                }
                if obs[i].is_none() {
                    let winner = envs[i].winner();
                    for h_idx in 0..hlen {
                        if snaps[i].board[h_idx].is_none() {
                            snapshot(&mut snaps, i, h_idx, true, false, me, winner, 0, None);
                        }
                    }
                    live[i] = false;
                    continue;
                }
                let o = obs[i].as_ref().unwrap();
                let turn = o.global[2];
                for (h_idx, h) in hs.iter().enumerate() {
                    if snaps[i].board[h_idx].is_none() && turn > turn_start + *h as i32 - 1 {
                        let player = envs[i].player();
                        snapshot(&mut snaps, i, h_idx, false, false, me, 0, player, obs[i].as_ref());
                    }
                }
                if snaps[i].board.iter().all(|v| v.is_some()) {
                    live[i] = false;
                    continue;
                }
                if n[i] >= self.max_rollout {
                    self.capped += 1;
                    let winner = envs[i].winner();
                    for h_idx in 0..hlen {
                        if snaps[i].board[h_idx].is_none() {
                            snapshot(&mut snaps, i, h_idx, true, true, me, winner, 0, None);
                        }
                    }
                    live[i] = false;
                    continue;
                }
                act.push(i);
            }
            if act.is_empty() {
                break;
            }
            let req: Vec<&Obs> = act.iter().map(|&i| obs[i].as_ref().unwrap()).collect();
            let t_fwd = tick();
            let lg = nets.forward_threads(&req, self.search_threads)?;
            if let (Some(p), Some(t)) = (self.prof.as_mut(), t_fwd) {
                p.fwd_s += t.elapsed().as_secs_f64();
                p.fwd_calls += 1;
                p.fwd_obs += req.len();
            }
            if lg.len() != act.len() {
                return Err(format!("forward: {} rows for {} observations", lg.len(), act.len()));
            }
            for (&i, row) in act.iter().zip(lg.iter()) {
                n[i] += 1;
                let o = obs[i].as_ref().unwrap();
                let mask: Vec<bool> = o.action_mask.iter().map(|&x| x != 0).collect();
                let z = masked(row, &mask)?;
                let (z_g, key) = guard.mask(i as i32, o, &z);
                let br = &branches[i];
                let a = if br.sample == 0 {
                    argmax_select(&z_g).unwrap_or(0)
                } else {
                    sample_action(&z_g, self.rollout_temp, &mut rngs[i])
                };
                guard.note(i as i32, key, a as i32);
                let mut new_obs = Obs::zeroed();
                let t_step = tick();
                let pending = envs[i].step(a as i32, &mut new_obs)?;
                if let (Some(p), Some(t)) = (self.prof.as_mut(), t_step) {
                    p.step_s += t.elapsed().as_secs_f64();
                    p.steps += 1;
                }
                obs[i] = if pending { Some(*new_obs) } else { None };
            }
        }
        // Any branch still live (should not happen) gets its current position for remaining horizons.
        for i in 0..b {
            if live[i] {
                if obs[i].is_some() {
                    let player = envs[i].player();
                    for h_idx in 0..hlen {
                        if snaps[i].board[h_idx].is_none() {
                            snapshot(&mut snaps, i, h_idx, false, false, me, 0, player, obs[i].as_ref());
                        }
                    }
                } else {
                    let winner = envs[i].winner();
                    for h_idx in 0..hlen {
                        if snaps[i].board[h_idx].is_none() {
                            snapshot(&mut snaps, i, h_idx, true, false, me, winner, 0, None);
                        }
                    }
                }
            }
        }
        // Batched learned-value scoring for non-terminal snapshots.
        let mut snap_value: Vec<Vec<Option<f64>>> = vec![vec![None; hlen]; b];
        if self.has_value {
            for i in 0..b {
                for h_idx in 0..hlen {
                    if snaps[i].vterm[h_idx].is_some() {
                        snap_value[i][h_idx] = snaps[i].vterm[h_idx];
                    }
                }
            }
            let need: Vec<(usize, usize)> = (0..b)
                .flat_map(|i| (0..hlen).map(move |h| (i, h)))
                .filter(|&(i, h_idx)| snaps[i].board[h_idx].is_some() && snap_value[i][h_idx].is_none())
                .collect();
            if !need.is_empty() {
                let vals = {
                    let req: Vec<&Obs> = need
                        .iter()
                        .map(|&(i, h_idx)| snaps[i].vobs[h_idx].as_ref().expect("value snapshot without obs"))
                        .collect();
                    let t_val = tick();
                    let v = nets.value(&req)?;
                    if let (Some(p), Some(t)) = (self.prof.as_mut(), t_val) {
                        p.value_s += t.elapsed().as_secs_f64();
                        p.value_calls += 1;
                        p.value_obs += req.len();
                    }
                    v
                };
                if vals.len() != need.len() {
                    return Err(format!("value: {} values for {} observations", vals.len(), need.len()));
                }
                for (&(i, h_idx), &x) in need.iter().zip(vals.iter()) {
                    let v = x as f64;
                    snap_value[i][h_idx] = Some(if snaps[i].actor[h_idx] == me { v } else { -v });
                }
            }
        }
        // Average over samples per candidate.
        let mut keys: Vec<String> = Vec::new();
        for h in &hs {
            keys.push(format!("board@h{h}"));
            if self.has_value {
                keys.push(format!("value@h{h}"));
            }
            keys.push(format!("done@h{h}"));
            keys.push(format!("cards@h{h}"));
        }
        let mut out: BTreeMap<String, Vec<f64>> = keys.iter().map(|k| (k.clone(), vec![0.0; n_cands])).collect();
        let mut counts = vec![0usize; n_cands];
        for (i, br) in branches.iter().enumerate() {
            let j = br.cand_idx;
            counts[j] += 1;
            for (h_idx, h) in hs.iter().enumerate() {
                *out.get_mut(&format!("board@h{h}")).unwrap().get_mut(j).unwrap() +=
                    snaps[i].board[h_idx].expect("board snapshot missing");
                *out.get_mut(&format!("done@h{h}")).unwrap().get_mut(j).unwrap() +=
                    snaps[i].done[h_idx].expect("done snapshot missing");
                *out.get_mut(&format!("cards@h{h}")).unwrap().get_mut(j).unwrap() +=
                    snaps[i].cards[h_idx].expect("cards snapshot missing");
                if self.has_value {
                    *out.get_mut(&format!("value@h{h}")).unwrap().get_mut(j).unwrap() +=
                        snap_value[i][h_idx].expect("value snapshot missing");
                }
            }
        }
        let sk = self.select_key();
        let mut last_samples: Vec<Vec<f64>> = vec![Vec::new(); n_cands];
        if out.contains_key(&sk) && (sk.starts_with("value@") || sk.starts_with("board@")) {
            let h: usize = sk.split("@h").nth(1).unwrap().parse().map_err(|e| format!("select key {sk}: {e}"))?;
            let h_idx = hs.iter().position(|&x| x == h).ok_or_else(|| format!("select horizon {h} not searched"))?;
            for (i, br) in branches.iter().enumerate() {
                let v = if sk.starts_with("value@") {
                    snap_value[i][h_idx].expect("value sample missing")
                } else {
                    snaps[i].board[h_idx].expect("board sample missing")
                };
                last_samples[br.cand_idx].push(v);
            }
        }
        let mut last_samples_all: BTreeMap<String, Vec<Vec<f64>>> =
            keys.iter().map(|k| (k.clone(), vec![Vec::new(); n_cands])).collect();
        for kk in &keys {
            let (name, h) = kk.split_once("@h").ok_or_else(|| format!("score key {kk}"))?;
            let h: usize = h.parse().map_err(|e| format!("score key {kk}: {e}"))?;
            let h_idx = hs.iter().position(|&x| x == h).ok_or_else(|| format!("horizon {h}"))?;
            for (i, br) in branches.iter().enumerate() {
                let v = match name {
                    "board" => snaps[i].board[h_idx].expect("board sample missing"),
                    "cards" => snaps[i].cards[h_idx].expect("cards sample missing"),
                    "done" => snaps[i].done[h_idx].expect("done sample missing"),
                    "value" => snap_value[i][h_idx].expect("value sample missing"),
                    _ => return Err(format!("unknown score key {kk}")),
                };
                last_samples_all.get_mut(kk).unwrap()[br.cand_idx].push(v);
            }
        }
        normalise_averages(&mut out, &counts);
        Ok(RolloutOut { scores: out, last_samples, last_samples_all })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs_with(cards: &[i32], global: &[i32]) -> Obs {
        let mut o = Obs::zeroed();
        o.cards[..cards.len()].copy_from_slice(cards);
        o.global[..global.len()].copy_from_slice(global);
        *o
    }

    #[test]
    fn position_score_symmetric_from_actor_view() {
        let mut g = [0i32; 24];
        g[0] = 7000; // actor lp
        g[1] = 6000;
        g[6 + 1] = 2; // actor hand
        g[6 + 2] = 1; // actor mzone
        g[6 + 3] = 1; // actor szone
        g[13 + 1] = 1; // opp hand
        g[13 + 2] = 2; // opp mzone
        g[13 + 3] = 0;
        let o = obs_with(&[], &g);
        // actor 0 sees: (1+1) - (2+0) + 0.5*(2-1) + 1000/2000 = 0 + 0.5 + 0.5 = 1
        assert_eq!(position_score(&o, 0, 0), 1.0);
        // actor 1's view of the same state is the negation for me=1
        let mut g1 = g;
        g1.swap(0, 1);
        g1[6 + 1] = 1;
        g1[6 + 2] = 2;
        g1[13 + 1] = 2;
        g1[13 + 2] = 1;
        let o1 = obs_with(&[], &g1);
        assert_eq!(position_score(&o1, 1, 1), 1.0);
    }

    #[test]
    fn cards_score_is_hand_plus_field() {
        let mut g = [0i32; 24];
        g[7] = 3;
        g[8] = 2;
        g[9] = 1;
        g[14] = 1;
        g[15] = 1;
        g[16] = 0;
        let o = obs_with(&[], &g);
        assert_eq!(cards_score(&o, 0, 0), 4.0);
        assert_eq!(cards_score(&o, 1, 0), -4.0);
    }

    #[test]
    fn candidates_stable_descending_and_window_order() {
        let mut o = Obs::zeroed();
        o.action_mask[0] = 1;
        o.action_mask[1] = 1;
        o.action_mask[2] = 1;
        o.actions[0] = KIND_PASS;
        o.actions[10] = KIND_ACTIVATE;
        o.actions[20] = KIND_ACTIVATE;
        let e = Engine::new();
        let s = WindowSearch::new(&e, 2, 1, 1, 1.0, 300, 0, None, false);
        let z = [1.0, 1.0, 2.0];
        assert_eq!(s.candidates(&z, &o, false, false), vec![2, 0]);
        let w = s.candidates(&z, &o, true, false);
        assert_eq!(w, vec![0, 2]);
        assert_eq!(s.candidates(&z, &o, false, true), vec![2, 0]);
    }


    #[test]
    fn averaging_divides_by_branch_count() {
        let mut out = BTreeMap::new();
        out.insert("value@h1".to_string(), vec![4.0, 6.0, 9.0, 0.0]);
        out.insert("done@h1".to_string(), vec![2.0, 3.0, 7.0, 0.0]);
        normalise_averages(&mut out, &[2, 3, 0, 0]);
        assert_eq!(out["value@h1"], vec![2.0, 2.0, 9.0, 0.0]);
        assert_eq!(out["done@h1"], vec![1.0, 1.0, 7.0, 0.0]);
    }

    #[test]
    fn sampling_walk_matches_python_fixtures() {
        let data = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/window_sampling.jsonl"));
        let mut n = 0usize;
        for line in data.lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let z: Vec<f64> = v["z_bits"].as_array().unwrap().iter()
                .map(|x| f64::from_bits(x.as_u64().unwrap())).collect();
            let temp = f64::from_bits(v["temp_bits"].as_u64().unwrap());
            let mut r = Random::from_str(v["seed"].as_str().unwrap());
            assert_eq!(sample_action(&z, temp, &mut r) as u64, v["a"].as_u64().unwrap(), "line {n}");
            n += 1;
        }
        assert!(n >= 100, "only {n} fixtures");
    }

    #[test]
    fn sample_action_single_legal_and_none_legal() {
        let mut r = Random::from_i128(1);
        let z = [f64::NEG_INFINITY, 3.0, f64::NEG_INFINITY];
        assert_eq!(sample_action(&z, 1.0, &mut r), 1);
        let z2 = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert_eq!(sample_action(&z2, 1.0, &mut r), 0);
    }
}
