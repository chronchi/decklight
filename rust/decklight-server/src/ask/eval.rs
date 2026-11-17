//! Scoring for the Ask eval set (config/ask/README.md): a tool score (the expected calls were made, in order, and no
//! call failed before the limit) and an answer score (every `answer` check passes). Pure functions over one turn's
//! record, so they are tested offline; `bin/ask_eval.rs` runs the turns.

use std::collections::BTreeMap;

use serde_json::{Value, json};

/// One tool call as the eval saw it.
#[derive(Clone, Debug)]
pub struct Call {
    pub name: String,
    pub args: Value,
    pub ok: bool,
    pub result: Value,
    /// Made after the limit was reached (a refusal, not the model's mistake).
    pub after_limit: bool,
}

/// `expected` is a subset of `actual`: object keys given must match, arrays match element by element with equal
/// length, text matches case-insensitively as a substring, numbers by value (a number written as text counts).
/// A `null` expected value means the key is absent (or null): `{"id": null}` is the deck list, not one deck.
pub fn args_match(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => e.iter().all(|(k, v)| match a.get(k) {
            None => v.is_null(),
            Some(av) => args_match(v, av),
        }),
        (Value::Null, a) => a.is_null(),
        (Value::Object(e), _) => e.is_empty(),
        (Value::Array(e), Value::Array(a)) => e.len() == a.len() && e.iter().zip(a).all(|(x, y)| args_match(x, y)),
        (Value::String(e), Value::String(a)) => a.to_lowercase().contains(&e.to_lowercase()),
        (Value::Number(e), a) => {
            let a = a.as_f64().or_else(|| a.as_str().and_then(|s| s.trim().parse().ok()));
            a.is_some_and(|a| (a - e.as_f64().unwrap_or(f64::NAN)).abs() < 1e-9)
        }
        (e, a) => e == a,
    }
}

/// The tool score: each expected call matched by a later actual call than the previous match (other calls in
/// between are free), and no failed call before the limit.
pub fn tool_score(expected: &[Value], calls: &[Call]) -> (bool, String) {
    let mut at = 0;
    for (i, e) in expected.iter().enumerate() {
        let name = e["name"].as_str().unwrap_or("");
        let want = e.get("args").cloned().unwrap_or(json!({}));
        // A call refused after the limit did not run, so it cannot be the expected one.
        match calls[at..].iter().position(|c| !c.after_limit && c.name == name && args_match(&want, &c.args)) {
            Some(j) => at += j + 1,
            None => return (false, format!("expected call {} ({name} {want}) not made", i + 1)),
        }
    }
    if let Some(c) = calls.iter().find(|c| !c.ok && !c.after_limit) {
        return (false, format!("{} failed: {}", c.name, c.result));
    }
    (true, String::new())
}

/// Numbers written in the text ("33.76", "33,8 %", "1,800" → 1800).
pub fn numbers(text: &str) -> Vec<f64> {
    let b: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let mut s = String::new();
        if i > 0 && (b[i - 1] == '-' || b[i - 1] == '−') {
            s.push('-');
        }
        while i < b.len() {
            let c = b[i];
            if c.is_ascii_digit() {
                s.push(c);
            } else if (c == '.' || c == ',') && b.get(i + 1).is_some_and(|d| d.is_ascii_digit()) {
                // A comma before exactly three digits is a thousands separator; otherwise a decimal mark.
                let digits = b[i + 1..].iter().take_while(|d| d.is_ascii_digit()).count();
                if !(c == ',' && digits == 3) {
                    s.push('.');
                }
            } else {
                break;
            }
            i += 1;
        }
        if let Ok(n) = s.parse() {
            out.push(n);
        }
    }
    out
}

fn has_number(text: &str, value: f64, tol: f64) -> bool {
    numbers(text).iter().any(|n| (n - value).abs() <= tol + 1e-9)
}

/// A crude language check: more of the language's function words than of the other's (Spanish also counts its
/// accents and inverted marks). Languages without a word list are not checked.
fn looks_like(text: &str, lang: &str) -> bool {
    let words: Vec<String> = text.split(|c: char| !c.is_alphabetic()).map(str::to_lowercase).collect();
    let count = |list: &[&str]| words.iter().filter(|w| list.contains(&w.as_str())).count();
    let en = count(&["the", "is", "and", "of", "to", "this", "it", "you", "your", "when", "with", "yes"]);
    let es = count(&["el", "la", "los", "las", "de", "que", "es", "una", "por", "para", "con", "cuando", "tu", "del", "sí", "y"])
        + text.chars().filter(|c| "áéíóúñ¿¡".contains(*c)).count();
    match lang {
        "es" => es > en,
        "en" => en > es,
        _ => true,
    }
}

/// Card names in the answer that no tool returned this turn, no earlier turn of the chat showed (`prior`: the
/// conversation so far as text) and the question did not mention. Names shorter than 8 characters are skipped (too
/// many false hits on plain words: Jinzo, Raigeki are not checked).
pub fn invented<'a>(
    answer: &str,
    question: &str,
    prior: &str,
    calls: &[Call],
    names: impl Iterator<Item = &'a String>,
) -> Vec<String> {
    let mut seen: String = calls.iter().map(|c| c.result.to_string()).collect::<Vec<_>>().join("\n");
    seen.push_str(prior);
    let question = question.to_lowercase();
    let mut out: Vec<String> = names
        .filter(|n| n.chars().count() >= 8 && answer.contains(n.as_str()))
        // Results are JSON text, so a name with quotes (Maxx "C") is in there escaped.
        .filter(|n| !seen.contains(n.as_str()) && !seen.contains(serde_json::to_string(n).unwrap_or_default().trim_matches('"')))
        .filter(|n| !question.contains(&n.to_lowercase()))
        .cloned()
        .collect();
    out.sort();
    out.dedup();
    out
}

/// What `numbers_from_fixture` expects, from the fixture's own stats: `win_rate` (all games) and `groups.win_rate`
/// (each group's, for the question's expected `group_by`).
pub fn fixture_numbers(keys: &[Value], expected_tools: &[Value], stats: &dyn Fn(&Value) -> Result<Value, String>) -> Vec<f64> {
    let mut out = Vec::new();
    for k in keys.iter().filter_map(Value::as_str) {
        if k == "win_rate" {
            if let Ok(v) = stats(&json!({})) {
                out.extend(v["summary"]["win_rate"].as_f64());
            }
        } else if k == "groups.win_rate" {
            let args = expected_tools
                .iter()
                .find(|t| t["name"] == "my_stats")
                .map(|t| t.get("args").cloned().unwrap_or(json!({})));
            if let Some(Ok(v)) = args.map(|a| stats(&a)) {
                out.extend(v["summary"]["groups"].as_array().into_iter().flatten().filter_map(|g| g["win_rate"].as_f64()));
            }
        }
    }
    out
}

/// The answer checks of one question. Returns (all passed, each check with its result).
pub fn answer_score(
    spec: &Value,
    question: &str,
    prior: &str,
    answer: &str,
    calls: &[Call],
    names: &BTreeMap<u32, String>,
    fixture: &[f64],
) -> (bool, Vec<Value>) {
    let a = &spec["answer"];
    let lower = answer.to_lowercase();
    let mut checks = Vec::new();
    let mut check = |name: String, pass: bool, detail: String| checks.push(json!({"check": name, "pass": pass, "detail": detail}));
    if answer.trim().is_empty() {
        check("non_empty".into(), false, "empty answer".into());
    }
    for m in a["must"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        let pass = m.split('|').any(|alt| lower.contains(&alt.to_lowercase()));
        check(format!("must {m}"), pass, String::new());
    }
    for n in a["numbers"].as_array().into_iter().flatten() {
        let (v, tol) = (n["value"].as_f64().unwrap_or(f64::NAN), n["tol"].as_f64().unwrap_or(0.1));
        check(format!("number {v}±{tol}"), has_number(answer, v, tol), String::new());
    }
    if a.get("numbers_from_fixture").is_some() {
        if fixture.is_empty() {
            check("fixture numbers".into(), false, "the fixture gave no numbers".into());
        }
        for v in fixture {
            // Rounded to whole percents is fine ("71%" for 71.4).
            check(format!("fixture number {v}"), has_number(answer, *v, 0.5), String::new());
        }
    }
    if a["must_not_invent"].as_bool() == Some(true) {
        let bad = invented(answer, question, prior, calls, names.values());
        check("must_not_invent".into(), bad.is_empty(), bad.join(", "));
    }
    if let Some(lang) = a["language"].as_str() {
        check(format!("language {lang}"), looks_like(answer, lang), String::new());
    }
    let cap = a["max_tool_calls"].as_u64().unwrap_or(8) as usize;
    let made = calls.iter().filter(|c| !c.after_limit).count();
    check(format!("tool calls ≤ {cap}"), made <= cap, format!("{made} made"));
    let pass = checks.iter().all(|c| c["pass"] == true);
    (pass, checks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: Value, ok: bool) -> Call {
        Call { name: name.into(), args, ok, result: json!({}), after_limit: false }
    }

    #[test]
    fn args_are_a_subset_with_loose_text_and_numbers() {
        assert!(args_match(&json!({"text": "ash blo"}), &json!({"text": "Ash Blossom", "limit": 5})));
        assert!(args_match(&json!({"code": 14558127}), &json!({"code": "14558127"})));
        assert!(args_match(&json!({"hand": 5}), &json!({"hand": 5.0})));
        assert!(!args_match(&json!({"hand": 5}), &json!({"hand": 6})));
        assert!(args_match(&json!({"groups": [{"copies": 3}]}), &json!({"groups": [{"copies": 3, "min": 1}]})));
        assert!(!args_match(&json!({"groups": [{"copies": 3}]}), &json!({"groups": [{"copies": 3}, {"copies": 1}]})));
        assert!(args_match(&json!({}), &json!(null)));
        assert!(!args_match(&json!({"id": "x"}), &json!({})));
        assert!(args_match(&json!({"id": null}), &json!({})) && !args_match(&json!({"id": null}), &json!({"id": "x"})));
    }

    #[test]
    fn tool_score_wants_order_and_no_failures_before_the_limit() {
        let exp = vec![json!({"name": "search_cards", "args": {"text": "ash"}}), json!({"name": "card", "args": {"code": 1}})];
        let good = vec![call("search_cards", json!({"text": "Ash"}), true), call("deck", json!({}), true), call("card", json!({"code": 1}), true)];
        assert!(tool_score(&exp, &good).0);
        let reversed = vec![good[2].clone(), good[0].clone()];
        assert!(!tool_score(&exp, &reversed).0);
        let mut failed = good.clone();
        failed.push(call("card", json!({"code": 0}), false));
        assert!(!tool_score(&exp, &failed).0);
        failed.last_mut().unwrap().after_limit = true;
        assert!(tool_score(&exp, &failed).0, "a refusal after the limit is not the model's mistake");
        let mut refused = good.clone();
        refused[2].after_limit = true;
        assert!(!tool_score(&exp, &refused).0, "a refused call does not count as made");
    }

    #[test]
    fn numbers_read_decimals_commas_and_thousands() {
        assert_eq!(numbers("about 33.76% or 33,8 % and 1,800 ATK, Level 3."), vec![33.76, 33.8, 1800.0, 3.0]);
        assert!(has_number("≈ 9.8%", 9.8, 0.1));
        assert!(!has_number("10.1%", 9.8, 0.1));
        assert_eq!(numbers("a change of -9.8 points"), vec![-9.8]);
    }

    #[test]
    fn invented_names_are_those_no_tool_returned() {
        let names: BTreeMap<u32, String> =
            [(1, "Ash Blossom & Joyous Spring".to_string()), (2, "Effect Veiler".to_string()), (3, "Dark Magician".to_string())].into();
        let mut c = call("card", json!({}), true);
        c.result = json!({"name": "Ash Blossom & Joyous Spring", "rows": [{"name": "Dark Magician Girl"}]});
        let ans = "Ash Blossom & Joyous Spring is like Effect Veiler; Dark Magician too.";
        assert_eq!(invented(ans, "q", "", &[c.clone()], names.values()), vec!["Effect Veiler".to_string()]);
        assert!(invented(ans, "q", "earlier: Effect Veiler", &[c.clone()], names.values()).is_empty(), "shown in an earlier turn");
        assert!(invented(ans, "is effect veiler good?", "", &[c], names.values()).is_empty(), "the question named it");
    }

    #[test]
    fn answer_checks_cover_must_numbers_language_and_the_cap() {
        let spec = json!({"answer": {"must": ["legal|allowed"], "numbers": [{"value": 33.76, "tol": 0.1}], "language": "es"}});
        let names = BTreeMap::new();
        let (pass, _) = answer_score(&spec, "q", "", "Es legal: la probabilidad es de 33,8 % para el mazo.", &[], &names, &[]);
        assert!(pass);
        let (pass, checks) = answer_score(&spec, "q", "", "It is legal and the odds are 33.76%.", &[], &names, &[]);
        assert!(!pass, "{checks:?}");
        assert!(answer_score(&spec, "q", "", "Sí, es legal: 33,8 %.", &[], &names, &[]).0, "terse Spanish");
        let fx = json!({"answer": {"numbers_from_fixture": ["win_rate"]}});
        assert!(answer_score(&fx, "q", "", "Your win rate is 55%.", &[], &names, &[54.5]).0);
        assert!(!answer_score(&fx, "q", "", "Your win rate is 60%.", &[], &names, &[54.5]).0);
        assert!(!answer_score(&fx, "q", "", "Your win rate is 55%.", &[], &names, &[]).0, "no fixture numbers is a failure");
        let many: Vec<Call> = (0..9).map(|_| call("card", json!({}), true)).collect();
        assert!(!answer_score(&json!({"answer": {}}), "q", "", "ok", &many, &names, &[]).0);
    }
}
