//! Incremental duel statistics for the web client (port of `src/decklight/web/stats.py`, behaviour for behaviour).

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

const FIELD_LOCS: [&str; 4] = ["mzone", "szone", "field", "emz"];
const SS_HOWS: [&str; 3] = ["normal", "tribute", "flip"];
const WIPE_REASONS: [&str; 4] = ["destroyed", "banished", "sent", "returned"];
const NO_WIPE_REASONS: [&str; 3] = ["material", "summon", "spsummon"];

fn is_ss(how: &str) -> bool {
    !SS_HOWS.contains(&how)
}

fn s(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn i(v: &Value, key: &str) -> i64 {
    match v.get(key) {
        Some(Value::Number(n)) => n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
        _ => 0,
    }
}

fn arr<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

#[derive(Clone)]
struct Entry {
    code: i64,
    name: String,
    count: i64,
}

impl Entry {
    fn value(&self) -> Value {
        json!({"code": self.code, "name": self.name, "count": self.count})
    }
}

#[derive(Clone)]
struct Last {
    link: i64,
    card: Option<Value>,
}

#[derive(Clone)]
struct Attack {
    n: i64,
    card: Value,
}

#[derive(Clone)]
struct Moment {
    turn: i64,
    kind: String,
    text: String,
    sub: String,
    w0: Option<f64>,
    widx: usize,
    wipe_key: Option<(i64, String)>,
}

impl Moment {
    fn value(&self) -> Value {
        json!({"turn": self.turn, "kind": self.kind, "text": self.text, "sub": self.sub})
    }
}

pub struct DuelStats {
    pub viewer: i32,
    lp0: i64,
    turn: i64,
    lp: BTreeMap<&'static str, i64>,
    dealt: BTreeMap<&'static str, i64>,
    summons: BTreeMap<&'static str, i64>,
    summons_ss: BTreeMap<&'static str, i64>,
    activations: BTreeMap<&'static str, i64>,
    negated: BTreeMap<&'static str, i64>,
    were_negated: BTreeMap<&'static str, i64>,
    windows_total: i64,
    windows_used: i64,
    win_hist: Vec<(i64, f64)>,
    chart_labels: Vec<String>,
    chart_me: Vec<i64>,
    chart_opp: Vec<i64>,
    moments: Vec<Moment>,
    field_lost: BTreeMap<(i64, String), i64>,
    opp_seen: Vec<Entry>,
    you_used: Vec<Entry>,
    links: BTreeMap<i64, Value>,
    solving: i64,
    last: Option<Last>,
    last_attack: Option<Attack>,
    view: Option<Value>,
    in_resolve: bool,
    resolving_owner: Option<String>,
}

impl DuelStats {
    pub fn new(viewer: i32, lp0: i64) -> DuelStats {
        let mut st = DuelStats {
            viewer,
            lp0,
            turn: 0,
            lp: BTreeMap::new(),
            dealt: BTreeMap::new(),
            summons: BTreeMap::new(),
            summons_ss: BTreeMap::new(),
            activations: BTreeMap::new(),
            negated: BTreeMap::new(),
            were_negated: BTreeMap::new(),
            windows_total: 0,
            windows_used: 0,
            win_hist: Vec::new(),
            chart_labels: Vec::new(),
            chart_me: Vec::new(),
            chart_opp: Vec::new(),
            moments: Vec::new(),
            field_lost: BTreeMap::new(),
            opp_seen: Vec::new(),
            you_used: Vec::new(),
            links: BTreeMap::new(),
            solving: 0,
            last: None,
            last_attack: None,
            view: None,
            in_resolve: false,
            resolving_owner: None,
        };
        st.reset();
        st
    }

    pub fn reset(&mut self) {
        self.turn = 0;
        self.lp = BTreeMap::from([("me", self.lp0), ("opp", self.lp0)]);
        self.dealt = BTreeMap::from([("me", 0), ("opp", 0)]);
        self.summons = BTreeMap::from([("me", 0), ("opp", 0)]);
        self.summons_ss = BTreeMap::from([("me", 0), ("opp", 0)]);
        self.activations = BTreeMap::from([("me", 0), ("opp", 0)]);
        self.negated = BTreeMap::from([("me", 0), ("opp", 0)]);
        self.were_negated = BTreeMap::from([("me", 0), ("opp", 0)]);
        self.windows_total = 0;
        self.windows_used = 0;
        self.win_hist.clear();
        self.chart_labels.clear();
        self.chart_me.clear();
        self.chart_opp.clear();
        self.moments.clear();
        self.field_lost.clear();
        self.opp_seen.clear();
        self.you_used.clear();
        self.links.clear();
        self.solving = 0;
        self.last = None;
        self.last_attack = None;
        self.view = None;
        self.in_resolve = false;
        self.resolving_owner = None;
    }

    pub fn feed_cues(&mut self, cues: &[Value]) {
        for c in cues {
            match s(c, "t").as_str() {
                "turn" => {
                    self.turn = i(c, "turn");
                    self.chart_labels.push(format!("T{}", self.turn));
                    self.chart_me.push(*self.lp.get("me").unwrap_or(&0));
                    self.chart_opp.push(*self.lp.get("opp").unwrap_or(&0));
                }
                "damage" | "recover" | "pay" | "lp" => self.feed_lp(c),
                "summon" => self.feed_summon(c),
                "activate" => self.feed_activate(c),
                "chain_solving" | "chain_solved" | "chain_negated" | "chain_disabled" | "chain_end" => {
                    self.feed_chain(c)
                }
                "attack" | "battle" => {
                    if let Some(card) = c.get("attacker").filter(|v| v.is_object()) {
                        self.last_attack = Some(Attack { n: i(c, "n"), card: card.clone() });
                    }
                    self.note_seen(c);
                }
                "move" => self.feed_move(c),
                "draw" | "confirm" | "target" => self.note_seen(c),
                _ => {}
            }
        }
    }

    fn feed_lp(&mut self, c: &Value) {
        let t = s(c, "t");
        let side = if s(c, "player") == "me" { "me" } else { "opp" };
        match t.as_str() {
            "damage" => {
                let amount = i(c, "amount");
                let cur = *self.lp.get(side).unwrap_or(&0);
                self.lp.insert(side, c.get("lp_after").map(|_| i(c, "lp_after")).unwrap_or(cur));
                let dealer = if side == "me" { "opp" } else { "me" };
                *self.dealt.get_mut(dealer).unwrap() += amount;
                if amount >= 1000 {
                    self.damage_moment(c, amount);
                }
            }
            "recover" | "pay" => {
                let cur = *self.lp.get(side).unwrap_or(&0);
                self.lp.insert(side, c.get("lp_after").map(|_| i(c, "lp_after")).unwrap_or(cur));
            }
            "lp" => {
                let cur = *self.lp.get(side).unwrap_or(&0);
                self.lp.insert(side, c.get("lp").map(|_| i(c, "lp")).unwrap_or(cur));
            }
            _ => {}
        }
    }

    fn feed_summon(&mut self, c: &Value) {
        let card = c.get("card").cloned().unwrap_or(Value::Null);
        let owner = s(&card, "owner");
        if owner != "me" && owner != "opp" {
            return;
        }
        *self.summons.get_mut(owner.as_str()).unwrap() += 1;
        if is_ss(&s(c, "how")) {
            *self.summons_ss.get_mut(owner.as_str()).unwrap() += 1;
        }
        let code = i(&card, "code");
        if code != 0 {
            let name = self.card_name(&card, code);
            let bucket = if owner == "opp" { &mut self.opp_seen } else { &mut self.you_used };
            add_play(bucket, code, name);
        }
    }

    fn feed_activate(&mut self, c: &Value) {
        let card = c.get("card").cloned().unwrap_or(Value::Null);
        let owner = s(&card, "owner");
        if owner == "me" || owner == "opp" {
            *self.activations.get_mut(owner.as_str()).unwrap() += 1;
        }
        let link = i(c, "link");
        if link != 0 {
            self.links.insert(link, if card.is_object() { card.clone() } else { json!({}) });
        }
        let code = i(&card, "code");
        if code != 0 && card.is_object() {
            let name = self.card_name(&card, code);
            if owner == "opp" {
                add_play(&mut self.opp_seen, code, name);
            } else if owner == "me" {
                add_play(&mut self.you_used, code, name);
            }
        }
    }

    fn card_name(&self, card: &Value, code: i64) -> String {
        let name = s(card, "name");
        if name.is_empty() { format!("#{code}") } else { name }
    }

    fn feed_chain(&mut self, c: &Value) {
        match s(c, "t").as_str() {
            "chain_solving" => {
                self.solving = i(c, "link");
                let owner = self
                    .links
                    .get(&self.solving)
                    .and_then(|v| v.get("owner"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                self.resolving_owner = if owner == "me" || owner == "opp" { Some(owner) } else { None };
                self.in_resolve = true;
            }
            "chain_solved" => {
                let link = i(c, "link");
                self.last = Some(Last { link, card: self.links.get(&link).cloned() });
                self.solving = 0;
                self.in_resolve = false;
                self.resolving_owner = None;
            }
            "chain_negated" | "chain_disabled" => {
                let link = i(c, "link");
                let card = self.links.get(&link).cloned().unwrap_or(json!({}));
                let mut by: Option<Value> = None;
                if self.solving != 0 && self.solving != link {
                    by = self.links.get(&self.solving).cloned();
                } else if let Some(last) = &self.last
                    && last.link == link + 1
                {
                    by = last.card.clone();
                }
                if by.is_none()
                    && let Some(k) = self.links.keys().find(|k| **k > link).copied()
                {
                    by = self.links.get(&k).cloned();
                }
                if let Some(v) = &by
                    && !v.is_object()
                {
                    by = None;
                }
                self.negation_moment(link, &card, by.as_ref(), s(c, "t") == "chain_disabled");
                self.solving = 0;
                self.in_resolve = false;
                self.resolving_owner = None;
            }
            "chain_end" => {
                self.links.clear();
                self.solving = 0;
                self.last = None;
                self.in_resolve = false;
                self.resolving_owner = None;
            }
            _ => {}
        }
    }

    fn feed_move(&mut self, c: &Value) {
        let card = c.get("card").cloned().unwrap_or(json!({}));
        let frm = c.get("from").cloned().unwrap_or(json!({}));
        let reason = s(c, "reason");
        if frm.is_object() && FIELD_LOCS.contains(&s(&frm, "loc").as_str()) {
            let side = s(&frm, "side");
            if (side == "me" || side == "opp")
                && !NO_WIPE_REASONS.contains(&reason.as_str())
                && WIPE_REASONS.contains(&reason.as_str())
                && self.opponent_caused(&side, c, &reason)
            {
                let key = (if self.turn != 0 { self.turn } else { 1 }, side);
                let n = {
                    let e = self.field_lost.entry(key.clone()).or_insert(0);
                    *e += 1;
                    *e
                };
                if n >= 3 {
                    self.wipe(key, n);
                }
            }
        }
        if reason == "set" && card.is_object() && s(&card, "owner") == "me" {
            let code = i(&card, "code");
            if code != 0 {
                let name = self.card_name(&card, code);
                add_play(&mut self.you_used, code, name);
            }
        }
        self.note_seen(c);
    }

    fn opponent_caused(&self, side: &str, c: &Value, reason: &str) -> bool {
        let foe = if side == "me" { "opp" } else { "me" };
        if self.in_resolve {
            return self.resolving_owner.as_deref() == Some(foe);
        }
        if reason != "destroyed" {
            return false;
        }
        let Some(atk) = &self.last_attack else { return false };
        if s(&atk.card, "owner") != foe {
            return false;
        }
        i(c, "n") - atk.n <= 12
    }

    fn note_seen(&mut self, c: &Value) {
        self.walk_seen(c);
    }

    fn walk_seen(&mut self, cur: &Value) {
        match cur {
            Value::Object(map) => {
                if map.contains_key("code") && map.contains_key("owner") {
                    let code = i(cur, "code");
                    if code != 0 && s(cur, "owner") == "opp" {
                        let name = self.card_name(cur, code);
                        ensure_seen(&mut self.opp_seen, code, name);
                    }
                }
                for v in map.values() {
                    self.walk_seen(v);
                }
            }
            Value::Array(items) => {
                for v in items {
                    self.walk_seen(v);
                }
            }
            _ => {}
        }
    }

    pub fn note_chain_window(&mut self, had_activatable: bool, used: bool) {
        if !had_activatable {
            return;
        }
        self.windows_total += 1;
        if used {
            self.windows_used += 1;
        }
    }

    pub fn note_winchance(&mut self, turn: i64, p: f64) {
        self.win_hist.push((turn, p));
    }

    fn new_moment(&self, turn: i64, kind: &str, text: String, sub: String) -> Moment {
        Moment {
            turn,
            kind: kind.to_string(),
            text,
            sub,
            w0: self.win_hist.last().map(|e| e.1),
            widx: self.win_hist.len(),
            wipe_key: None,
        }
    }

    pub fn note_view(&mut self, view: Option<&Value>) {
        self.view = view.cloned();
    }

    fn negation_moment(&mut self, _link: i64, card: &Value, by: Option<&Value>, disabled: bool) {
        let name = s(card, "name");
        let owner = s(card, "owner");
        let text;
        if owner == "me" {
            text = if name.is_empty() { "Your card was negated".to_string() } else { format!("Your {name} was negated") };
            *self.were_negated.get_mut("me").unwrap() += 1;
            let by_owner = by.map(|b| s(b, "owner")).unwrap_or_default();
            if by_owner == "me" {
                *self.negated.get_mut("me").unwrap() += 1;
            } else {
                *self.negated.get_mut("opp").unwrap() += 1;
            }
        } else if owner == "opp" {
            text = if name.is_empty() { "Bot's card was negated".to_string() } else { format!("Bot's {name} was negated") };
            *self.were_negated.get_mut("opp").unwrap() += 1;
            let by_owner = by.map(|b| s(b, "owner")).unwrap_or_default();
            if by_owner == "me" {
                *self.negated.get_mut("me").unwrap() += 1;
            } else {
                *self.negated.get_mut("opp").unwrap() += 1;
            }
        } else {
            text = if name.is_empty() { "A card was negated".to_string() } else { format!("{name} was negated") };
        }
        let by_name = by.map(|b| s(b, "name")).unwrap_or_default();
        let sub = if !by_name.is_empty() {
            format!("by {by_name}")
        } else if disabled {
            "effect negated".to_string()
        } else {
            "activation negated".to_string()
        };
        let m = self.new_moment(if self.turn != 0 { self.turn } else { 1 }, "negation", text, sub);
        self.moments.push(m);
    }

    fn damage_moment(&mut self, c: &Value, amount: i64) {
        let mut atk: Option<Value> = None;
        if let Some(last) = &self.last_attack
            && i(c, "n") - last.n <= 6
        {
            atk = Some(last.card.clone());
        }
        let (text, sub) = match atk {
            Some(a) if i(&a, "code") != 0 => {
                let nm = {
                    let n = s(&a, "name");
                    if n.is_empty() { "a monster".to_string() } else { n }
                };
                let owner = s(&a, "owner");
                if owner == "me" {
                    (format!("Your {nm} attacked"), format!("{amount} damage"))
                } else {
                    (format!("Bot attacked with {nm}"), format!("{amount} damage"))
                }
            }
            _ => {
                if s(c, "player") == "me" {
                    (format!("You took {amount} damage"), String::new())
                } else {
                    (format!("Bot took {amount} damage"), String::new())
                }
            }
        };
        let m = self.new_moment(if self.turn != 0 { self.turn } else { 1 }, "damage", text, sub);
        self.moments.push(m);
    }

    fn wipe(&mut self, key: (i64, String), n: i64) {
        let (turn, side) = key.clone();
        let text = if side == "me" {
            format!("You lost {n} cards from the field")
        } else {
            format!("Bot lost {n} cards from the field")
        };
        if let Some(pos) = self.moments.iter().position(|m| m.wipe_key.as_ref() == Some(&key)) {
            self.moments[pos].text = text;
            let mut m = self.moments.remove(pos);
            m.text = if side == "me" {
                format!("You lost {n} cards from the field")
            } else {
                format!("Bot lost {n} cards from the field")
            };
            self.moments.push(m);
        } else {
            let mut m = self.new_moment(turn, "wipe", text, String::new());
            m.wipe_key = Some(key);
            self.moments.push(m);
        }
    }

    fn ledger_side(&self, who: &str) -> Map<String, Value> {
        let key = if who == "you" { "me" } else { "opp" };
        let (hand, field, gy, ban, lpv) = if self.view.as_ref().is_none_or(|v| v.get(who).is_none()) {
            (0, 0, 0, 0, 0)
        } else {
            let side = &self.view.as_ref().unwrap()[who];
            (
                arr(side, "hand").len() as i64,
                field_count(side),
                arr(side, "gy").len() as i64,
                arr(side, "banished").len() as i64,
                i(side, "lp"),
            )
        };
        let total = hand + field;
        let mut m = Map::new();
        m.insert("lp".into(), json!(lpv));
        m.insert("hand".into(), json!(hand));
        m.insert("field".into(), json!(field));
        m.insert("total".into(), json!(total));
        m.insert("gy".into(), json!(gy));
        m.insert("banished".into(), json!(ban));
        m.insert("summons".into(), json!(self.summons.get(key).unwrap_or(&0)));
        m.insert("summons_ss".into(), json!(self.summons_ss.get(key).unwrap_or(&0)));
        m.insert("activations".into(), json!(self.activations.get(key).unwrap_or(&0)));
        m.insert("negated".into(), json!(self.negated.get(key).unwrap_or(&0)));
        m.insert("were_negated".into(), json!(self.were_negated.get(key).unwrap_or(&0)));
        m.insert("damage".into(), json!(self.dealt.get(key).unwrap_or(&0)));
        m
    }

    pub fn stats_message(&self, winchance_on: bool) -> Value {
        let mut you = self.ledger_side("you");
        let mut bot = self.ledger_side("bot");
        let you_lp = you.get("lp").and_then(Value::as_i64).unwrap_or(0);
        let bot_lp = bot.get("lp").and_then(Value::as_i64).unwrap_or(0);
        you.insert("lp".into(), json!(you_lp));
        bot.insert("lp".into(), json!(bot_lp));
        let you_total = you.get("total").and_then(Value::as_i64).unwrap_or(0);
        let bot_total = bot.get("total").and_then(Value::as_i64).unwrap_or(0);
        let diff = you_total - bot_total;
        you.insert("advantage".into(), json!(diff));
        bot.insert("advantage".into(), json!(-diff));
        let mut labels = self.chart_labels.clone();
        let mut cyou = self.chart_me.clone();
        let mut cbot = self.chart_opp.clone();
        if labels.is_empty() {
            labels = vec!["now".into()];
            cyou = vec![you_lp];
            cbot = vec![bot_lp];
        } else {
            labels.push("now".into());
            cyou.push(you_lp);
            cbot.push(bot_lp);
        }
        let show_wc = winchance_on && !self.win_hist.is_empty();
        let mut moments = Vec::new();
        for m in self.moments.iter().rev().take(5) {
            let mut v = m.value();
            if show_wc
                && let Some(w0) = m.w0
                && m.widx < self.win_hist.len()
            {
                let w1 = self.win_hist[m.widx].1;
                let span = format!("you were {}%, then {}%", py_round(w0 * 100.0), py_round(w1 * 100.0));
                let sub = v.get("sub").and_then(Value::as_str).unwrap_or("").to_string();
                let new = if sub.is_empty() { span } else { format!("{sub} · {span}") };
                if let Some(o) = v.as_object_mut() {
                    o.insert("sub".into(), Value::String(new));
                }
            }
            moments.push(v);
        }
        let mut opp_seen: Vec<&Entry> = self.opp_seen.iter().collect();
        opp_seen.sort_by(|a, b| a.name.cmp(&b.name));
        let mut you_used: Vec<&Entry> = self.you_used.iter().collect();
        you_used.sort_by(|a, b| a.name.cmp(&b.name));
        let mut out = Map::new();
        out.insert("turn".into(), json!(self.turn));
        out.insert("lp_you".into(), json!(you_lp));
        out.insert("lp_bot".into(), json!(bot_lp));
        out.insert("dealt_you".into(), json!(self.dealt.get("me").unwrap_or(&0)));
        out.insert("dealt_bot".into(), json!(self.dealt.get("opp").unwrap_or(&0)));
        out.insert("adv_you".into(), json!(you_total));
        out.insert("adv_bot".into(), json!(bot_total));
        out.insert("adv_diff".into(), json!(diff));
        out.insert("windows_used".into(), json!(self.windows_used));
        out.insert("windows_total".into(), json!(self.windows_total));
        out.insert("neg_you".into(), json!(self.negated.get("me").unwrap_or(&0)));
        out.insert("neg_were_you".into(), json!(self.were_negated.get("me").unwrap_or(&0)));
        out.insert("neg_bot".into(), json!(self.negated.get("opp").unwrap_or(&0)));
        out.insert("neg_were_bot".into(), json!(self.were_negated.get("opp").unwrap_or(&0)));
        out.insert("chart_labels".into(), json!(labels));
        out.insert("chart_you".into(), json!(cyou));
        out.insert("chart_bot".into(), json!(cbot));
        out.insert("moments".into(), Value::Array(moments));
        out.insert("ledger".into(), json!({"you": Value::Object(you), "bot": Value::Object(bot)}));
        out.insert("opp_seen".into(), Value::Array(opp_seen.iter().map(|e| e.value()).collect()));
        out.insert("you_used".into(), Value::Array(you_used.iter().map(|e| e.value()).collect()));
        if !self.win_hist.is_empty() && winchance_on {
            let last = self.win_hist.last().unwrap();
            out.insert(
                "winchance".into(),
                json!({
                    "p": last.1,
                    "turn": last.0,
                    "history": self.win_hist.iter().map(|e| json!([e.0, e.1])).collect::<Vec<_>>(),
                }),
            );
        }
        Value::Object(out)
    }
}

fn field_count(side: &Value) -> i64 {
    let mut n = 0i64;
    for zone in ["mzone", "emz", "szone"] {
        n += arr(side, zone).iter().filter(|c| !c.is_null()).count() as i64;
    }
    if side.get("field").is_some_and(|v| !v.is_null()) {
        n += 1;
    }
    n
}

fn ensure_seen(bucket: &mut Vec<Entry>, code: i64, name: String) {
    if let Some(e) = bucket.iter_mut().find(|e| e.code == code) {
        if e.name.is_empty() && !name.is_empty() {
            e.name = name;
        }
    } else {
        bucket.push(Entry { code, name, count: 0 });
    }
}

fn add_play(bucket: &mut Vec<Entry>, code: i64, name: String) {
    if let Some(e) = bucket.iter_mut().find(|e| e.code == code) {
        e.count += 1;
        if e.name.is_empty() && !name.is_empty() {
            e.name = name;
        }
    } else {
        bucket.push(Entry { code, name, count: 1 });
    }
}

fn py_round(x: f64) -> i64 {
    let fl = x.floor();
    let d = x - fl;
    let f = fl as i64;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1
    } else if f % 2 == 0 {
        f
    } else {
        f + 1
    }
}

