//! `deck_odds` (config/ask/tools.v1.json): the exact chance that an opening hand holds at least (or exactly) `min`
//! cards from each group, by the multivariate hypergeometric distribution. No sampling, so the eval set can check
//! answers to 0.1 points.

use serde_json::{Value, json};

/// One group of cards for the odds: `copies` in the deck, at least (or exactly, with `exact`) `min` in hand.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub label: String,
    pub copies: u32,
    pub min: u32,
    pub exact: bool,
}

/// C(n, k) as f64; exact for every deck size we allow (C(60, 6) is about 5e7).
fn choose(n: u32, k: u32) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// P(every group's count in a `hand`-card draw from `deck_size` meets its condition). Groups must not share cards
/// and must fit in the deck (`deck_odds` checks both).
pub(crate) fn probability(deck_size: u32, hand: u32, groups: &[Group]) -> f64 {
    let held: u32 = groups.iter().map(|g| g.copies).sum();
    assert!(held <= deck_size, "groups hold {held} cards, more than the {deck_size}-card deck");
    let rest = deck_size - held;
    // Walk every count combination that meets the conditions; the remaining cards come from the rest of the deck.
    fn walk(groups: &[Group], left: u32, rest: u32, ways: f64) -> f64 {
        let Some((g, tail)) = groups.split_first() else {
            return ways * choose(rest, left);
        };
        let top = if g.exact { g.min } else { g.copies.min(left) };
        (g.min..=top.min(left)).map(|k| walk(tail, left - k, rest, ways * choose(g.copies, k))).sum()
    }
    walk(groups, hand, rest, 1.0) / choose(deck_size, hand)
}

/// An optional whole number in `lo..=hi`. Models often send `6.0`, so whole floats count; anything else (a string,
/// `-1`, `2.5`, out of range) is an error, never a silent default.
fn int(obj: &Value, key: &str, lo: u64, hi: u64, who: &str) -> Result<Option<u32>, String> {
    let v = match obj.get(key) {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    whole(v, lo, hi).map(Some).ok_or(format!("{who}`{key}` must be a whole number from {lo} to {hi}, got {v}"))
}

fn whole(v: &Value, lo: u64, hi: u64) -> Option<u32> {
    let n = v.as_u64().or_else(|| v.as_f64().filter(|f| f.fract() == 0.0 && (0.0..=hi as f64).contains(f)).map(|f| f as u64));
    n.filter(|n| (lo..=hi).contains(n)).map(|n| n as u32)
}

/// The tool: `args` as in tools.v1.json; `main` is the saved deck's Main Deck when `args.deck` was given (the caller
/// loads it). Errors are messages the model can act on.
pub fn deck_odds(args: &Value, main: Option<&[u32]>) -> Result<Value, String> {
    let hand = int(args, "hand", 5, 6, "")?.unwrap_or(5);
    let deck_size = match main {
        Some(m) => m.len() as u32,
        None => int(args, "deck_size", 40, 60, "")?.ok_or("give `deck` (a saved deck id) or `deck_size`")?,
    };
    if deck_size < hand {
        return Err(format!("the Main Deck has {deck_size} cards, fewer than a {hand}-card hand"));
    }
    let raw = args.get("groups").and_then(Value::as_array).filter(|g| (1..=4).contains(&g.len()));
    let raw = raw.ok_or("groups must be a list of 1 to 4 groups")?;
    let mut seen: Vec<u32> = Vec::new();
    let mut groups = Vec::new();
    for (i, g) in raw.iter().enumerate() {
        let label = match g.get("label") {
            None | Some(Value::Null) => format!("group {}", i + 1),
            Some(Value::String(s)) => s.clone(),
            Some(v) => return Err(format!("group {}: `label` must be text, got {v}", i + 1)),
        };
        let who = format!("{label}: ");
        let codes = match g.get("codes") {
            None | Some(Value::Null) => None,
            Some(Value::Array(c)) if !c.is_empty() => Some(c),
            Some(v) => return Err(format!("{who}`codes` must be a non-empty list of card codes, got {v}")),
        };
        if codes.is_some() && g.get("copies").is_some_and(|v| !v.is_null()) {
            return Err(format!("{who}give `codes` or `copies`, not both (the deck says how many copies)"));
        }
        let copies = match (main, codes) {
            (Some(m), Some(codes)) => {
                let mut n = 0;
                for c in codes {
                    let c = whole(c, 1, u32::MAX as u64).ok_or(format!("{who}codes must be card codes, got {c}"))?;
                    if seen.contains(&c) {
                        return Err(format!("{who}card {c} is in more than one group"));
                    }
                    seen.push(c);
                    let k = m.iter().filter(|&&x| x == c).count() as u32;
                    if k == 0 {
                        return Err(format!("{who}card {c} is not in the Main Deck"));
                    }
                    n += k;
                }
                n
            }
            // With a saved deck every group names its cards, so groups can never overlap unseen.
            (Some(_), None) => return Err(format!("{who}with a saved `deck`, give the group's `codes`")),
            (None, Some(_)) => return Err(format!("{who}codes need a saved `deck`; give `copies` for a hypothetical")),
            (None, None) => int(g, "copies", 1, 60, &who)?.ok_or(format!("{who}give `copies` (1 to 60)"))?,
        };
        let min = int(g, "min", 1, 60, &who)?.unwrap_or(1);
        let exact = match g.get("exact") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(b)) => *b,
            Some(v) => return Err(format!("{who}`exact` must be true or false, got {v}")),
        };
        if min > copies {
            return Err(format!("{who}min {min} is more than its {copies} copies"));
        }
        groups.push(Group { label, copies, min, exact });
    }
    if groups.iter().map(|g| g.copies).sum::<u32>() > deck_size {
        return Err(format!("the groups hold more cards than the {deck_size}-card deck"));
    }
    if groups.iter().map(|g| g.min).sum::<u32>() > hand {
        return Err(format!("the groups need more cards than a {hand}-card hand"));
    }
    let p = probability(deck_size, hand, &groups);
    let p_each: Vec<f64> = groups.iter().map(|g| probability(deck_size, hand, std::slice::from_ref(g))).collect();
    let pct = |x: f64| (x * 10000.0).round() / 100.0;
    Ok(json!({
        "p": pct(p),
        "hand": hand,
        "deck_size": deck_size,
        "groups": groups.iter().map(|g| json!({"label": g.label, "copies": g.copies, "min": g.min, "exact": g.exact}))
            .collect::<Vec<_>>(),
        "p_each": p_each.into_iter().map(pct).collect::<Vec<_>>(),
        "unit": "percent",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(copies: u32, min: u32, exact: bool) -> Group {
        Group { label: "x".into(), copies, min, exact }
    }

    #[test]
    fn matches_the_eval_set() {
        // config/ask/eval/v1.jsonl o01-o05 (computed with math.comb).
        let r = |p: f64| (p * 10000.0).round() / 100.0;
        assert_eq!(r(probability(40, 5, &[g(3, 1, false)])), 33.76);
        assert_eq!(r(probability(40, 6, &[g(3, 1, false)])), 39.43);
        assert_eq!(r(probability(40, 5, &[g(9, 1, false)])), 74.18);
        assert_eq!(r(probability(40, 5, &[g(3, 1, false), g(3, 1, false)])), 9.8);
        assert_eq!(r(probability(40, 5, &[g(3, 2, true)])), 3.54);
        // The review's independent cases (math.comb, NOTEBOOK 2026-11-14).
        assert_eq!(r(probability(45, 6, &[g(3, 1, false)])), 35.6);
        assert_eq!(r(probability(45, 6, &[g(3, 1, false), g(9, 1, false), g(6, 2, false)])), 2.7);
        assert_eq!(r(probability(45, 6, &[g(3, 2, true), g(9, 1, false)])), 2.62);
        assert_eq!(r(probability(40, 5, &[g(3, 1, true), g(3, 1, false)])), 8.97);
        assert_eq!(r(probability(60, 6, &[g(40, 1, false), g(10, 2, false), g(5, 1, true), g(5, 1, false)])), 2.17);
    }

    #[test]
    fn all_outcomes_sum_to_one() {
        let total: f64 = (0..=3).map(|k| probability(45, 6, &[g(3, k, true)])).sum();
        assert!((total - 1.0).abs() < 1e-12, "{total}");
        assert!((probability(40, 5, &[g(3, 0, false)]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn counts_copies_from_the_deck() {
        let mut main = vec![1u32; 34];
        main.extend([7, 7, 7, 8, 8, 8]);
        let args = json!({"deck": "d", "groups": [{"label": "A", "codes": [7]}, {"label": "B", "codes": [8]}]});
        let out = deck_odds(&args, Some(&main)).unwrap();
        assert_eq!(out["p"], 9.8);
        assert_eq!(out["p_each"], json!([33.76, 33.76]));
        assert_eq!(out["deck_size"], 40);
        let both = json!({"deck": "d", "groups": [{"codes": [7, 8]}]});
        assert_eq!(deck_odds(&both, Some(&main)).unwrap()["groups"][0]["copies"], 6);
        // Whole floats are what models often send.
        let float = json!({"deck_size": 40.0, "hand": 6.0, "groups": [{"copies": 3.0, "min": 1.0}]});
        assert_eq!(deck_odds(&float, None).unwrap()["p"], 39.43);
    }

    #[test]
    fn bad_arguments_are_errors() {
        let main = vec![1u32; 40];
        let bad = [
            (json!({"deck_size": 40, "hand": 7, "groups": [{"copies": 3}]}), None, "`hand` must"),
            (json!({"deck_size": 30, "groups": [{"copies": 3}]}), None, "`deck_size` must"),
            (json!({"groups": [{"copies": 3}]}), None, "give `deck`"),
            (json!({"deck_size": 40, "groups": []}), None, "1 to 4"),
            (json!({"deck_size": 40, "groups": [{"codes": [5]}]}), None, "need a saved"),
            (json!({"deck": "d", "groups": [{"codes": [5]}]}), Some(&main[..]), "not in the Main"),
            (json!({"deck": "d", "groups": [{"codes": [1]}, {"codes": [1]}]}), Some(&main[..]), "more than one group"),
            (json!({"deck": "d", "groups": [{"codes": [1], "copies": 1}]}), Some(&main[..]), "not both"),
            (json!({"deck_size": 40, "groups": [{"copies": 2, "min": 3}]}), None, "more than its"),
            (json!({"deck_size": 40, "groups": [{"copies": 3, "min": 3}, {"copies": 3, "min": 3}]}), None, "more cards than a"),
            (json!({"deck_size": 40, "groups": [{"copies": 30}, {"copies": 20}]}), None, "more cards than the"),
        ];
        for (args, main, want) in bad {
            let e = deck_odds(&args, main).unwrap_err();
            assert!(e.contains(want), "{args}: {e}");
        }
    }
}
