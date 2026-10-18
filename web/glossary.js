// Rules glossary (docs/web-ux-plan.md step 5, item 6): plain-language OCG/TCG
// Master Rule terms behind the tappable card-text keywords and the cue rule
// notes. Each entry is {id, term, aliases, short, long}:
//   short  one plain sentence (<= 120 chars), used in cue notes;
//   long   2-4 sentences, used in the keyword popover.
// The array literal is valid JSON (quoted keys) so tests/unit/test_glossary.py
// can read this file as text and parse it without a JS runtime. Aliases cover
// plural/wording variants and are unique across entries (whole-word matches,
// longest alias first).
import {esc} from "./render/card.js";

export const GLOSSARY = [
  {
    "id": "chain",
    "term": "Chain",
    "aliases": [
      "chain",
      "chains",
      "chained",
      "chaining"
    ],
    "short": "A chain resolves the last activated effect first, then the one before it.",
    "long": "When several effects are activated in response to each other, they form a chain. Effects resolve in reverse order: the most recently activated link resolves first. A chain ends when every link has resolved or been negated."
  },
  {
    "id": "chain-link",
    "term": "Chain link",
    "aliases": [
      "chain link",
      "chain links",
      "chain link 1",
      "chain link 2",
      "chain link 3",
      "chain link 4"
    ],
    "short": "Each card or effect added to a chain is one numbered chain link.",
    "long": "Every effect activated in a chain becomes a link, numbered from the first activation. The last link is the newest activation and resolves first. Chain links only exist while the chain is unresolved."
  },
  {
    "id": "spell-speed",
    "term": "Spell Speed",
    "aliases": [
      "spell speed",
      "spell speed 1",
      "spell speed 2",
      "spell speed 3"
    ],
    "short": "Spell Speed decides which effects can respond to which.",
    "long": "Spell Speed 1: Normal, Continuous, Equip, Field and Ritual Spells, and Ignition, Trigger and Flip Effects. Spell Speed 2: Quick-Play Spells, Normal and Continuous Traps, and Quick Effects. Spell Speed 3: Counter Traps. A response must be Spell Speed 2 or higher, and never lower than the link it responds to."
  },
  {
    "id": "quick-effect",
    "term": "Quick Effect",
    "aliases": [
      "quick effect",
      "quick effects"
    ],
    "short": "(Quick Effect) means the effect is Spell Speed 2 and usable in either player's turn.",
    "long": "A Quick Effect is a monster effect that can be activated during either player's turn, including in response to another effect. It is Spell Speed 2, so it can be chained to almost anything except a Counter Trap."
  },
  {
    "id": "ignition-effect",
    "term": "Ignition Effect",
    "aliases": [
      "ignition effect",
      "ignition effects"
    ],
    "short": "An Ignition Effect is a Spell Speed 1 effect you start in your own Main Phase.",
    "long": "An Ignition Effect has no trigger: you choose when to activate it, in your own Main Phase with an open game state. The card text says where it works from (field, hand or GY). Because it is Spell Speed 1, it can never be used as a response, but the opponent can respond to it."
  },
  {
    "id": "trigger-effect",
    "term": "Trigger Effect",
    "aliases": [
      "trigger effect",
      "trigger effects",
      "triggered effect"
    ],
    "short": "A Trigger Effect activates when its event happens; mandatory ones must, optional ones may.",
    "long": "A Trigger Effect responds to an event, such as being Summoned or sent to the GY, and starts a chain at Spell Speed 1. When several trigger at once, they go on the chain in this order: the turn player's mandatory, the opponent's mandatory, the turn player's optional, then the opponent's optional."
  },
  {
    "id": "continuous-effect",
    "term": "Continuous Effect",
    "aliases": [
      "continuous effect",
      "continuous effects"
    ],
    "short": "A Continuous Effect applies while the card is face-up and does not start a chain.",
    "long": "A Continuous Effect stays applied while its card is face-up on the field; it never starts a chain or forms a link. It can change ATK, prevent actions or grant abilities as long as the card remains. If the card leaves the field, the effect stops."
  },
  {
    "id": "once-per-turn",
    "term": "Once per turn",
    "aliases": [
      "once per turn",
      "you can only use this effect of",
      "hard once per turn",
      "soft once per turn"
    ],
    "short": "Once per turn limits how often an effect can be used.",
    "long": "A soft once-per-turn limit applies to each copy of the card separately, so a second copy can use the effect again. A hard once-per-turn (You can only use this effect of ... once per turn) applies to the card name: only one copy of that name may use it, even if several are on the field."
  },
  {
    "id": "damage-step",
    "term": "Damage Step",
    "aliases": [
      "damage step",
      "damage calculation"
    ],
    "short": "The Damage Step is when battle damage is calculated during an attack.",
    "long": "The Damage Step is the part of a battle where ATK and DEF are compared and battle damage is dealt. Few effects may be activated here, mainly those that change ATK or DEF or negate the attack. Destruction by battle happens at the end of the Damage Step."
  },
  {
    "id": "battle-phase-steps",
    "term": "Battle Phase steps",
    "aliases": [
      "battle phase",
      "battle step",
      "start step",
      "end step"
    ],
    "short": "The Battle Phase runs Start Step, Battle Step, Damage Step, then End Step.",
    "long": "A Battle Phase has four steps: Start Step, Battle Step, Damage Step and End Step. Each attack is declared in the Battle Step and resolved in the Damage Step. When the turn player stops attacking, the phase moves to the End Step and then to Main Phase 2."
  },
  {
    "id": "banish",
    "term": "Banish",
    "aliases": [
      "banish",
      "banishes",
      "banished",
      "banishing"
    ],
    "short": "Banish sends a card to the banishment zone, not to the Graveyard.",
    "long": "A banished card is placed in the Banishment zone and is no longer in the Graveyard or on the field. Banishing is not destroying, so a card that cannot be destroyed can still be banished. Some cards return banished cards to the field or hand."
  },
  {
    "id": "send-to-gy",
    "term": "Send to the GY",
    "aliases": [
      "send to the gy",
      "sent to the gy",
      "send it to the gy",
      "send them to the gy"
    ],
    "short": "Sending to the GY puts a card in the Graveyard; it is not destruction.",
    "long": "Sending a card to the GY places it in the Graveyard. It is not the same as destroying: effects that prevent destruction do not stop a send. Many costs, such as discarding or milling, also send cards to the GY."
  },
  {
    "id": "destroy",
    "term": "Destroy",
    "aliases": [
      "destroy",
      "destroys",
      "destroyed",
      "destroying"
    ],
    "short": "Destroying sends a card from the field to the GY by battle or a card effect.",
    "long": "Destroy means a card on the field is removed by battle or a card effect, and it usually goes to the GY. Cards that protect from destruction stop it, but not banishing or sending. Destroyed Pendulum Monsters go face-up to the Extra Deck instead."
  },
  {
    "id": "negate-activation",
    "term": "Negate activation",
    "aliases": [
      "negate the activation",
      "negate activation",
      "negates the activation",
      "negate its activation"
    ],
    "short": "Negating the activation means the effect never resolves.",
    "long": "Negating an activation stops the card or effect from resolving at all. Costs already paid stay paid, and a once-per-turn effect still counts as used. A negated Spell or Trap card is sent to the GY; a monster whose effect activation is negated stays where it is."
  },
  {
    "id": "negate-effect",
    "term": "Negate effect",
    "aliases": [
      "negate the effects",
      "negate its effects",
      "negate that effect",
      "negate the effect",
      "negates the effects",
      "negate its effect"
    ],
    "short": "Negating the effect lets the activation stand but makes it do nothing.",
    "long": "Negating an effect, rather than the activation, means the card was still activated but its effect does nothing when it resolves. Costs and targets still happened. A monster whose effects are negated loses its effects while it stays face-up, often until the end of the turn."
  },
  {
    "id": "tribute",
    "term": "Tribute",
    "aliases": [
      "tribute",
      "tributes",
      "tributed"
    ],
    "short": "A Tribute is a monster you release from your field as a cost or summon.",
    "long": "Tributing sends a monster you control to the Graveyard as a cost. It is often used for a Tribute Summon or a Ritual Summon. Tributing is not destroying, so it works even through destruction protection."
  },
  {
    "id": "normal-summon",
    "term": "Normal Summon",
    "aliases": [
      "normal summon",
      "normal summons",
      "normal summoned",
      "tribute summon"
    ],
    "short": "A Normal Summon uses your once-per-turn summon; Level 5+ monsters need Tributes.",
    "long": "You may Normal Summon once per turn, and it does not start a chain. A Level 5 or 6 monster needs 1 Tribute and a Level 7 or higher monster needs 2, unless an effect says otherwise. Setting a monster counts as your Normal Summon for the turn."
  },
  {
    "id": "special-summon",
    "term": "Special Summon",
    "aliases": [
      "special summon",
      "special summons",
      "special summoned",
      "special summoning"
    ],
    "short": "A Special Summon is any summon other than your once-per-turn Normal Summon.",
    "long": "Special Summons come from card effects or from built-in procedures (Fusion, Synchro, Xyz, Link, Ritual) and do not use your Normal Summon. A built-in Summon does not start a chain, but the opponent may respond right after it. Link and Pendulum Monsters from the Extra Deck must go to the Extra Monster Zone or a zone a Link Monster points to; Fusion, Synchro and Xyz Monsters may use any Main Monster Zone."
  },
  {
    "id": "set",
    "term": "Set",
    "aliases": [
      "set",
      "sets",
      "set a card",
      "set the card"
    ],
    "short": "Setting places a monster or Spell/Trap face-down on the field without revealing it.",
    "long": "Setting a monster is a Normal Summon and places it face-down in Defense Position. Setting a Spell or Trap places it face-down, and it can usually be activated on a later turn. Flip effects and some cards care that a card was Set."
  },
  {
    "id": "flip",
    "term": "Flip",
    "aliases": [
      "flip",
      "flipped",
      "flip effect",
      "flip effects",
      "flip summon"
    ],
    "short": "Flip turns a face-down monster face-up; Flip Effects trigger when it is flipped.",
    "long": "Flipping a monster face-up can happen by a Flip Summon, by battle, or by a card effect, and its position may become Attack or Defense. A monster with a Flip Effect activates that effect after it is flipped. A card flipped after being attacked does not get its Flip Effect before damage is calculated."
  },
  {
    "id": "extra-deck",
    "term": "Extra Deck",
    "aliases": [
      "extra deck",
      "extra deck monster",
      "extra deck monsters"
    ],
    "short": "The Extra Deck holds Fusion, Synchro, Xyz and Link Monsters.",
    "long": "The Extra Deck holds up to 15 Fusion, Synchro, Xyz and Link Monsters. You Summon them only with the matching method, such as a Fusion Summon or a Link Summon. If an Extra Deck monster would return to the hand or Main Deck, it goes back to the Extra Deck."
  },
  {
    "id": "materials",
    "term": "Materials",
    "aliases": [
      "material",
      "materials",
      "xyz material",
      "xyz materials",
      "fusion material",
      "synchro material",
      "link material"
    ],
    "short": "Materials are the cards used to Summon a Fusion, Synchro, Xyz or Link Monster.",
    "long": "Materials are sent to the GY, or attached as Xyz Materials, to perform a Summon. The required kind and number are written in the monster's text. A Link Monster's material count also sets its Link Rating."
  },
  {
    "id": "hand-trap",
    "term": "Hand trap",
    "aliases": [
      "hand trap",
      "hand traps"
    ],
    "short": "A hand trap is a monster you can activate from your hand to interrupt the opponent.",
    "long": "Hand traps are monsters that can be activated from the hand during the opponent's turn, often as Spell Speed 2. Common examples negate searches, summons or effects. Because they start a chain, they must follow Spell Speed rules."
  },
  {
    "id": "field-spell",
    "term": "Field Spell",
    "aliases": [
      "field spell",
      "field spell card",
      "field spell cards"
    ],
    "short": "A Field Spell sits in the Field Zone and affects the whole field.",
    "long": "A Field Spell is placed in the Field Zone; each player may have one face-up. Its effect usually supports a theme or changes the rules. Playing a new one of your own sends your old one to the GY."
  },
  {
    "id": "quick-play-spell",
    "term": "Quick-Play Spell",
    "aliases": [
      "quick-play spell",
      "quick-play spells",
      "quick-play spell card"
    ],
    "short": "A Quick-Play Spell is Spell Speed 2: use it from hand on your turn, or Set it for either turn.",
    "long": "A Quick-Play Spell can be activated from the hand during your own turn as Spell Speed 2. If Set, it may be activated in either player's turn, but not in the turn it was Set. This makes it a common way to respond to the opponent."
  },
  {
    "id": "counter-trap",
    "term": "Counter Trap",
    "aliases": [
      "counter trap",
      "counter traps",
      "counter trap card"
    ],
    "short": "A Counter Trap is Spell Speed 3 and can respond to anything except another Counter Trap.",
    "long": "A Counter Trap is a Trap Card at Spell Speed 3, the highest. Only another Counter Trap may be chained to it. Counter Traps are the usual way to negate activations."
  },
  {
    "id": "direct-attack",
    "term": "Direct attack",
    "aliases": [
      "direct attack",
      "attacks directly",
      "attack directly"
    ],
    "short": "A direct attack hits the opponent's LP when no monster blocks it.",
    "long": "A direct attack is an attack on the opponent with no defending monster. It can only happen if the opponent controls no monsters or an effect allows it. The damage equals the attacker's ATK."
  },
  {
    "id": "piercing-damage",
    "term": "Piercing damage",
    "aliases": [
      "piercing",
      "piercing damage",
      "piercing battle damage"
    ],
    "short": "Piercing damage is dealt when you attack a Defense Position monster with lower DEF.",
    "long": "Piercing means that when the attacker's ATK exceeds the DEF of a Defense Position monster, the difference is dealt as battle damage. Without piercing, battle damage is 0 when attacking a Defense Position monster. Card effects can grant piercing."
  },
  {
    "id": "cost-vs-effect",
    "term": "Cost vs effect",
    "aliases": [
      "as a cost",
      "as the cost",
      "pay as a cost",
      "as a cost to activate"
    ],
    "short": "A cost is paid before an effect resolves and cannot be refunded by negating the effect.",
    "long": "In card text, what comes before the semicolon (;) is done when the effect is activated: costs and targets. A cost stays paid even if the activation or the effect is negated. Everything after the semicolon happens only when the effect resolves."
  },
  {
    "id": "if-vs-when",
    "term": "\"If\" vs \"when\"",
    "aliases": [
      "missing the timing",
      "miss the timing",
      "misses the timing"
    ],
    "short": "Optional when triggers can miss their timing; if triggers cannot.",
    "long": "When ... you can is an optional Trigger Effect that must be the last thing to happen to activate, so it can miss the timing. If ... you can does not check the last action the same way and is harder to miss. This is why some triggers work after a chain but others do not."
  },
  {
    "id": "target",
    "term": "Target",
    "aliases": [
      "target",
      "targets",
      "targeting",
      "targeted"
    ],
    "short": "Targeting chooses a specific card when the effect is activated.",
    "long": "Some effects choose, or target, a card when they are activated, before they resolve. If the target leaves the field or becomes illegal, the effect usually resolves without applying. Others choose at resolution and are not affected the same way."
  }
];

let _rules = null;

function aliasRegExp(alias) {
  const body = alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  return new RegExp("\\b" + body + "\\b", "i");
}

// Longest alias first, so "negate the activation" beats "negate".
function rules() {
  if (_rules) return _rules;
  const items = [];
  for (const g of GLOSSARY) {
    for (const a of g.aliases || []) items.push({id: g.id, term: g.term, alias: a});
  }
  items.sort((a, b) => b.alias.length - a.alias.length);
  _rules = items.map((it) => Object.assign({}, it, {rx: aliasRegExp(it.alias)}));
  return _rules;
}

// Wrap the first occurrence of each glossary alias in already-escaped text.
// `tag` defaults to "button" (card detail); option-effect text uses "span"
// because an option row is itself a <button> (nested buttons break the DOM).
// Matching on escaped text means card text can never inject HTML.
export function kwHTML(escapedText, tag) {
  if (!escapedText) return escapedText || "";
  tag = tag || "button";
  const taken = [];
  const hits = [];
  for (const r of rules()) {
    const m = r.rx.exec(escapedText);
    if (!m) continue;
    const s = m.index;
    const e = s + m[0].length;
    if (taken.some(([a, b]) => s < b && a < e)) continue;   // never inside another match
    taken.push([s, e]);
    hits.push({s, e, id: r.id, term: r.term, txt: m[0]});
  }
  hits.sort((a, b) => a.s - b.s);
  let out = "";
  let pos = 0;
  for (const h of hits) {
    out += escapedText.slice(pos, h.s) +
      "<" + tag + (tag === "button" ? ' type="button"' : "") + ' class="kw" data-act="kw" data-kw="' + h.id +
      '" aria-label="' + esc("Rule: " + h.term) + '">' + h.txt + "</" + tag + ">";
    pos = h.e;
  }
  return out + escapedText.slice(pos);
}

export function glossaryEntry(id) {
  return GLOSSARY.find((g) => g.id === id) || null;
}

export function ruleShort(id) {
  const g = glossaryEntry(id);
  return g ? g.short : null;
}
