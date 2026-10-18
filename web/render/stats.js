// Stats + Cards panels for the duel-log sheet (Log | Stats | Cards tabs).
// Pure functions of the server's `stats` object (web/stats.py); no game logic.
import {thumbHTML, esc} from "./card.js";

export const LOG_TABS = ["log", "stats", "cards"];
const TAB_LABEL = {log: "Log", stats: "Stats", cards: "Cards"};

export function tabStripHTML(active, prefix) {
  const p = prefix || "d";
  return '<div class="dtabs" role="tablist" aria-label="Duel log views">' + LOG_TABS.map((t) =>
    '<button role="tab" id="' + p + 'tab-' + t + '" aria-selected="' + (active === t) + '" aria-controls="' + p + 'panel-' + t + '"' +
    ' tabindex="' + (active === t ? "0" : "-1") + '" class="' + (active === t ? "on" : "") + '"' +
    ' data-logtab="' + t + '">' + TAB_LABEL[t] + "</button>").join("") + "</div>";
}

export function tabPanelOpen(prefix, which) {
  const p = prefix || "d";
  return '<div class="dbody" data-key="' + p + 'body" role="tabpanel" id="' + p + 'panel-' + which +
    '" aria-labelledby="' + p + 'tab-' + which + '" tabindex="0">';
}

function fmtDiff(d) {
  return (d > 0 ? "+" : d < 0 ? "−" : "") + Math.abs(d);
}

export function tilesHTML(st) {
  const lpYou = st.lp_you != null ? st.lp_you : 0;
  const lpBot = st.lp_bot != null ? st.lp_bot : 0;
  const dealtYou = st.dealt_you || 0, dealtBot = st.dealt_bot || 0;
  const adv = st.adv_diff || 0, nYou = st.adv_you || 0, nBot = st.adv_bot || 0;
  const wUsed = st.windows_used || 0, wTot = st.windows_total || 0;
  const negYou = st.neg_you || 0, negWere = st.neg_were_you || 0;
  return '<div class="dtiles">' +
    '<div class="dtile"><div class="dk">Life points</div>' +
    '<div class="dn lp"><span class="you">' + lpYou + '</span> <span class="vs">vs</span> <span class="bot">' + lpBot + "</span></div>" +
    '<div class="ds">you dealt ' + dealtYou + ", took " + dealtBot + "</div></div>" +
    '<div class="dtile"><div class="dk">Card advantage</div>' +
    '<div class="dn">' + esc(fmtDiff(adv)) + "</div>" +
    '<div class="ds">' + nYou + " cards vs the bot's " + nBot + "</div></div>" +
    '<div class="dtile"><div class="dk">Response windows</div>' +
    '<div class="dn">' + wUsed + ' <span class="of">of ' + wTot + " used</span></div>" +
    '<div class="ds">' + Math.max(0, wTot - wUsed) + " passed while you could respond</div></div>" +
    '<div class="dtile"><div class="dk">Negations</div>' +
    '<div class="dn">' + negYou + ' <span class="of">/ ' + negWere + "</span></div>" +
    '<div class="ds">you negated ' + negYou + ", were negated " + negWere + "</div></div>" +
    "</div>";
}

function chartPoints(vals, w, h, pad, lo, hi) {
  const n = vals.length;
  if (n === 1) return ["M " + (w / 2).toFixed(1) + "," + (h - pad).toFixed(1)];
  const span = hi - lo || 1;
  return vals.map((v, i) => {
    const x = (pad + (w - 2 * pad) * (i / (n - 1))).toFixed(1);
    const y = (pad + (h - 2 * pad) * (1 - (v - lo) / span)).toFixed(1);
    return (i ? "L" : "M") + " " + x + "," + y;
  });
}

export function chartHTML(st) {
  const labels = st.chart_labels || ["now"];
  const you = st.chart_you || [st.lp_you || 0];
  const bot = st.chart_bot || [st.lp_bot || 0];
  const W = 352, H = 120, PAD = 10;
  const all = [...you, ...bot];
  let lo = Math.min(...all, 0), hi = Math.max(...all, 8000);
  if (hi - lo < 1000) { hi += 500; lo = Math.max(0, lo - 500); }
  const py = chartPoints(you, W, H, PAD, lo, hi).join(" ");
  const pb = chartPoints(bot, W, H, PAD, lo, hi).join(" ");
  // The calibrated win-chance line (dashed kincha, own 0-100% scale): one point
  // per chart label, carrying the last sample at or before that turn forward.
  let winPath = "", winLegend = "", winCol = null, winSummary = "";
  const hist = st.winchance && Array.isArray(st.winchance.history) ? st.winchance.history : null;
  if (hist && hist.length) {
    const turnOf = (label) => {
      const m = /^T(\d+)$/.exec(label || "");
      return m ? parseInt(m[1], 10) : Infinity;   // "now" takes the latest sample
    };
    const pts = [];
    const col = [];
    labels.forEach((label, i) => {
      let p = null;
      for (const [t, v] of hist) if (t <= turnOf(label)) p = v;
      col.push(p);
      if (p == null) return;
      const n = labels.length;
      const x = (PAD + (W - 2 * PAD) * (n === 1 ? 0.5 : i / (n - 1))).toFixed(1);
      const y = (PAD + (H - 2 * PAD) * (1 - p)).toFixed(1);
      pts.push((pts.length ? "L" : "M") + " " + x + "," + y);
    });
    if (pts.length > 1) winPath = '<path d="' + pts.join(" ") + '" class="ln win" fill="none"/>';
    else if (pts.length === 1) winPath = '<path d="' + pts[0] + '" class="ln win" fill="none"/>';
    winLegend = '<span><i class="sw win" aria-hidden="true"></i>Win chance</span>';
    winCol = col;
    winSummary = " Win chance: " + col.map((p) => (p == null ? "—" : Math.round(p * 100) + "%")).join(", ") + ".";
  }
  const dot = (vals, cls) => {
    const n = vals.length, span = hi - lo || 1;
    return vals.map((v, i) => {
      const x = (PAD + (W - 2 * PAD) * (n === 1 ? 0.5 : i / (n - 1))).toFixed(1);
      const y = (PAD + (H - 2 * PAD) * (1 - (v - lo) / span)).toFixed(1);
      return cls === "you"
        ? '<circle cx="' + x + '" cy="' + y + '" r="3.5" class="mk you"/>'
        : '<rect x="' + (+x - 3.5).toFixed(1) + '" y="' + (+y - 3.5).toFixed(1) + '" width="7" height="7" class="mk bot"/>';
    }).join("");
  };
  const summary = "Life points by turn. You: " + you.join(", ") + ". Bot: " + bot.join(", ") + "." + winSummary;
  const ticks = labels.map((l) => "<span>" + esc(l) + "</span>").join("");
  const rows = labels.map((l, i) =>
    "<tr><th scope=\"row\">" + esc(l) + "</th><td>" + (you[i] != null ? you[i] : "—") +
    "</td><td>" + (bot[i] != null ? bot[i] : "—") + "</td>" +
    (winCol ? "<td>" + (winCol[i] != null ? Math.round(winCol[i] * 100) + "%" : "—") + "</td>" : "") +
    "</tr>").join("");
  return '<section class="dsec" aria-label="Life points chart"><div class="dsechead">Life points</div>' +
    '<div class="dchart"><svg viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none" role="img" aria-label="' + esc(summary) + '">' +
    '<g class="grid" aria-hidden="true"><line x1="0" y1="' + (H / 2) + '" x2="' + W + '" y2="' + (H / 2) + '"/></g>' +
    '<path d="' + py + '" class="ln you" fill="none"/>' +
    '<path d="' + pb + '" class="ln bot" fill="none"/>' +
    winPath +
    dot(you, "you") + dot(bot, "bot") +
    "</svg>" +
    '<div class="dticks" aria-hidden="true">' + ticks + "</div></div>" +
    '<div class="dlegend"><span><i class="sw you" aria-hidden="true"></i>Your LP (round marks)</span>' +
    '<span><i class="sw bot" aria-hidden="true"></i>Bot LP (square marks)</span>' + winLegend + "</div>" +
    '<table class="sr-only"><caption>Life points by turn</caption>' +
    "<thead><tr><th scope=\"col\">Turn</th><th scope=\"col\">Your LP</th><th scope=\"col\">Bot LP</th>" +
    (winCol ? "<th scope=\"col\">Win chance</th>" : "") + "</tr></thead>" +
    "<tbody>" + rows + "</tbody></table></section>";
}

export function momentsHTML(st) {
  const ms = st.moments || [];
  if (!ms.length) return '<section class="dsec" aria-label="Key moments"><div class="dsechead">Key moments</div>' +
    '<p class="emptytx">Nothing decisive yet.</p></section>';
  const rows = ms.map((m) =>
    '<div class="dmom"><span class="t">T' + esc(String(m.turn != null ? m.turn : "?")) + "</span>" +
    '<div class="d"><span>' + esc(m.text || "") + "</span>" +
    (m.sub ? '<small>' + esc(m.sub) + "</small>" : "") + "</div></div>").join("");
  return '<section class="dsec" aria-label="Key moments"><div class="dsechead">Key moments</div>' + rows + "</section>";
}

export function ledgerHTML(st) {
  const L = st.ledger || {}, Y = L.you || {}, B = L.bot || {};
  const adv = (v) => esc(fmtDiff(v || 0));
  const row = (label, fy, fb) =>
    "<tr><td>" + label + "</td><td class=\"y\">" + fy + "</td><td class=\"b\">" + fb + "</td></tr>";
  return '<section class="dsec" aria-label="Duel ledger"><div class="dsechead">Ledger</div>' +
    '<table class="dledger"><thead><tr><th scope="col"><span class="sr-only">Measure</span></th>' +
    '<th scope="col"><span class="dot you" aria-hidden="true"></span> You</th>' +
    '<th scope="col"><span class="sq bot" aria-hidden="true"></span> Bot</th></tr></thead><tbody>' +
    row("Life points", Y.lp != null ? Y.lp : "—", B.lp != null ? B.lp : "—") +
    row("Cards in hand", Y.hand != null ? Y.hand : "—", B.hand != null ? B.hand : "—") +
    row("Cards on field", Y.field != null ? Y.field : "—", B.field != null ? B.field : "—") +
    row("Card advantage", adv(Y.advantage), adv(B.advantage)) +
    row("GY · banished", esc((Y.gy != null ? Y.gy : "—") + " · " + (Y.banished != null ? Y.banished : "—")),
        esc((B.gy != null ? B.gy : "—") + " · " + (B.banished != null ? B.banished : "—"))) +
    row("Summons", esc((Y.summons != null ? Y.summons : "—") + " (" + (Y.summons_ss || 0) + " SS)"),
        esc((B.summons != null ? B.summons : "—") + " (" + (B.summons_ss || 0) + " SS)")) +
    row("Effects activated", Y.activations != null ? Y.activations : "—", B.activations != null ? B.activations : "—") +
    row("Negated / were negated",
        esc((Y.negated != null ? Y.negated : "—") + " / " + (Y.were_negated != null ? Y.were_negated : "—")),
        esc((B.negated != null ? B.negated : "—") + " / " + (B.were_negated != null ? B.were_negated : "—"))) +
    row("Damage dealt", Y.damage != null ? Y.damage : "—", B.damage != null ? B.damage : "—") +
    "</tbody></table></section>";
}

export function statsPanelHTML(st, mock) {
  if (!st) return '<p class="emptytx">No duel yet.</p>';
  const turn = st.turn ? '<div class="dstathd"><span class="tag">Turn ' + esc(String(st.turn)) + "</span></div>" : "";
  return turn + tilesHTML(st) + chartHTML(st) + momentsHTML(st) + ledgerHTML(st);
}

function cardCell(e, mock) {
  const code = e.code || 0, name = e.name || ("#" + code), n = e.count || 0;
  const card = {uid: "stat:" + code, code, name, faceup: true, pos: null,
    atk: 0, def: 0, level: 0, rank: 0, link: 0, counters: 0, overlays: []};
  const played = n > 0 ? ", played " + n + " time" + (n === 1 ? "" : "s") : "";
  return '<button class="dcell" data-statcode="' + code + '" aria-label="' + esc(name + played + ", card details") + '">' +
    '<span class="cbox" aria-hidden="true">' + thumbHTML(card, mock) +
    (n > 0 ? '<span class="cnt">×' + n + "</span>" : "") + "</span>" +
    '<span class="cname">' + esc(name) + (n > 0 ? esc(" ×" + n) : "") + "</span></button>";
}

export function cardsPanelHTML(st, mock) {
  if (!st) return '<p class="emptytx">No duel yet.</p>';
  const opp = st.opp_seen || [], mine = st.you_used || [];
  const sec = (title, sub, list) =>
    '<section class="dsec" aria-label="' + esc(title) + '"><div class="dsechead">' + esc(title) + "</div>" +
    '<div class="dsub">' + esc(sub) + "</div>" +
    (list.length ? '<div class="dcards">' + list.map((e) => cardCell(e, mock)).join("") + "</div>"
      : '<p class="emptytx">None yet.</p>') + "</section>";
  return sec("Opponent's cards seen", "Every bot card revealed this duel · ×N = times played.", opp) +
    sec("Your cards used", "Activated, summoned or set.", mine);
}
