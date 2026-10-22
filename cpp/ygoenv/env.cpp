// Port of src/decklight/engine/{obs,env}.py.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ygoenv.h"

namespace ygo {

namespace {

constexpr uint32_t STATUS_OBS_MASK = STATUS_DISABLED | STATUS_SET_TURN | STATUS_SUMMON_TURN |
                                     STATUS_FLIP_SUMMON_TURN | STATUS_SPSUMMON_TURN | STATUS_FORM_CHANGED |
                                     STATUS_ATTACK_CANCELED | STATUS_CHAINING | STATUS_FORBIDDEN;
constexpr uint32_t QUERY_LOCATIONS[] = {LOCATION_DECK,  LOCATION_HAND,    LOCATION_MZONE, LOCATION_SZONE,
                                        LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA};
constexpr uint32_t ROW_LOCS[] = {LOCATION_HAND,  LOCATION_MZONE,   LOCATION_SZONE,
                                 LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA};

int32_t i32(uint64_t v) { return static_cast<int32_t>(static_cast<uint32_t>(v & 0xFFFFFFFFu)); }

int loc_id(uint32_t location) {
  if (location & LOCATION_OVERLAY) return 8;
  switch (location) {
    case LOCATION_DECK: return 1;
    case LOCATION_HAND: return 2;
    case LOCATION_MZONE: return 3;
    case LOCATION_SZONE: return 4;
    case LOCATION_GRAVE: return 5;
    case LOCATION_REMOVED: return 6;
    case LOCATION_EXTRA: return 7;
    default: return 0;
  }
}

uint64_t key(uint32_t con, uint32_t loc, uint32_t seq, uint32_t ov) {
  return (uint64_t{con} << 56) | (uint64_t{loc & 0xFF} << 48) | (uint64_t{seq & 0xFFFF} << 32) | ov;
}

uint64_t ref_key(const CardRef& r) {
  if (r.location & LOCATION_OVERLAY) return key(r.controller, LOCATION_OVERLAY, r.sequence, r.position);
  return key(r.controller, r.location, r.sequence, 0);
}

bool revealing(Kind k) { return k == ACTIVATE || k == SUMMON || k == SPSUMMON || k == REPOS || k == ATTACK; }

}  // namespace

bool prof_on() {
  static bool on = std::getenv("YGOENV_PROFILE") != nullptr;
  return on;
}

uint64_t now_us() {
  using namespace std::chrono;
  return static_cast<uint64_t>(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
}

void Prof::dump(const char* what) const {
  std::fprintf(stderr,
               "[ygo-prof] %s steps=%llu boards=%llu autos=%llu keys=%llu fills=%llu declares=%llu "
               "lqueries=%llu "
               "advance=%llu decide=%llu options=%llu qloc=%llu qfield=%llu board_rest=%llu "
               "history=%llu keymatch=%llu fill=%llu respond=%llu\n",
               what, (unsigned long long)steps, (unsigned long long)boards, (unsigned long long)autos,
               (unsigned long long)keys, (unsigned long long)fills, (unsigned long long)declares,
               (unsigned long long)lqueries,
               (unsigned long long)advance_us, (unsigned long long)decide_us, (unsigned long long)options_us,
               (unsigned long long)qloc_us, (unsigned long long)qfield_us, (unsigned long long)board_rest_us,
               (unsigned long long)history_us, (unsigned long long)keymatch_us, (unsigned long long)fill_us,
               (unsigned long long)respond_us);
}

namespace {

struct Span {
  uint64_t& acc;
  uint64_t t0 = 0;
  bool on = false;
  Span(uint64_t& a, bool on_) : acc(a), on(on_) { if (on) t0 = now_us(); }
  ~Span() { if (on) acc += now_us() - t0; }
};

}  // namespace

bool Board::is_public(const CardState& c) {
  if (c.location == LOCATION_GRAVE || c.location == LOCATION_OVERLAY) return true;
  return c.is_public || ((c.position & POS_FACEUP) && c.location != LOCATION_HAND);
}

Board::Board(Duel& duel, int me, Prof* prof) : me(me) {
  bool on = prof && prof_on();
  uint64_t t_start = on ? now_us() : 0, t_loc = 0, t_field = 0;
  std::unordered_map<uint64_t, std::vector<std::optional<CardState>>> q;
  for (int p = 0; p < 2; ++p)
    for (uint32_t loc : QUERY_LOCATIONS) {
      auto cards = duel.query_location(static_cast<uint8_t>(p), loc);
      counts_[key(p, loc, 0, 0)] = static_cast<int>(std::count_if(cards.begin(), cards.end(), [](auto& c) { return c.has_value(); }));
      q[key(p, loc, 0, 0)] = std::move(cards);
    }
  if (on) t_loc = now_us();
  Bytes field = duel.query_field();
  if (on) t_field = now_us();
  // LPs and chain (layout: query.py lps / parse_chain)
  auto rd32 = [&](size_t p) { uint32_t v; std::memcpy(&v, field.data() + p, 4); return v; };
  size_t pos = 4;
  for (int p = 0; p < 2; ++p) {
    lp_[p] = rd32(pos);
    pos += 4;
    for (int z = 0; z < 15; ++z) pos += 1 + (field[pos] ? 5 : 0);
    pos += 24;
  }
  uint32_t n = rd32(pos);
  pos += 4;
  for (uint32_t i = 0; i < n; ++i) {
    ChainLink ch;
    ch.code = rd32(pos);
    ch.controller = field[pos + 4];
    ch.location = field[pos + 5];
    ch.sequence = rd32(pos + 6);
    ch.position = rd32(pos + 10);
    ch.trig_controller = field[pos + 14];
    ch.trig_location = field[pos + 15];
    ch.trig_sequence = rd32(pos + 16);
    std::memcpy(&ch.description, field.data() + pos + 20, 8);
    pos += 28;
    chain_.push_back(ch);
  }

  for (int p : {me, 1 - me}) {
    bool mine = p == me;
    for (uint32_t loc : ROW_LOCS)
      for (auto& c : q[key(p, loc, 0, 0)]) {
        if (!c) continue;
        if (loc == LOCATION_EXTRA && !mine && !(c->position & POS_FACEUP)) continue;
        rows.push_back({*c, mine || is_public(*c)});
      }
    for (auto& c : q[key(p, LOCATION_MZONE, 0, 0)]) {
      if (!c) continue;
      for (uint32_t i = 0; i < c->overlay.size(); ++i) {
        CardState o;
        o.controller = p;
        o.location = LOCATION_OVERLAY;
        o.sequence = c->sequence;
        o.code = c->overlay[i];
        o.is_public = true;
        o.overlay_index = i;
        rows.push_back({o, true});
      }
    }
  }
  std::vector<CardState> deck;
  for (auto& c : q[key(me, LOCATION_DECK, 0, 0)])
    if (c) deck.push_back(*c);
  std::sort(deck.begin(), deck.end(),
            [](auto& a, auto& b) { return std::tie(a.code, a.sequence) < std::tie(b.code, b.sequence); });
  for (auto& c : deck) rows.push_back({c, true});
  truncated = std::max(0, static_cast<int>(rows.size()) - MAX_CARDS);
  if (truncated) rows.resize(MAX_CARDS);
  for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
    auto& c = rows[i].card;
    index_[key(c.controller, c.location, c.sequence, c.overlay_index)] = i;
  }
  if (on) {
    uint64_t t_end = now_us();
    prof->qloc_us += t_loc - t_start;
    prof->qfield_us += t_field - t_loc;
    prof->board_rest_us += t_end - t_field;
    ++prof->boards;
  }
}

int Board::row_of(const CardRef* ref) const {
  if (!ref) return -1;
  auto it = index_.find(ref_key(*ref));
  return it == index_.end() ? -1 : it->second;
}

uint32_t Board::option_code(const Option& opt) const {
  if (!opt.has_card) return opt.code;
  int r = row_of(&opt.card);
  if (r >= 0) return rows[r].visible ? opt.code : 0;
  // a card offered from a hidden pile (Deck, face-down Extra Deck) is revealed to the chooser by the effect
  if (opt.card.controller == static_cast<uint32_t>(me) || opt.card.location == LOCATION_DECK ||
      opt.card.location == LOCATION_EXTRA)
    return opt.code;
  return 0;
}

std::set<uint32_t> Board::visible_codes() const {
  std::set<uint32_t> s;
  for (auto& r : rows)
    if (r.visible && r.card.code) s.insert(r.card.code);
  return s;
}

std::array<int32_t, ACTION_COLS> Board::option_row(const Option& opt, int msg_type, int step) const {
  const CardRef* ref = opt.has_card ? &opt.card : nullptr;
  return {static_cast<int32_t>(opt.kind),
          row_of(ref),
          i32(option_code(opt)),
          i32(opt.desc >> 20),
          static_cast<int32_t>(opt.desc & 0xFFFFF),
          i32(static_cast<uint64_t>(opt.arg)),
          msg_type,
          ref ? loc_id(ref->location) : 0,
          ref ? static_cast<int32_t>(ref->controller ^ me) : -1,
          step};
}

void Board::fill(Obs& obs, const Duel& duel, int msg_type, const std::vector<CardRef>& selected) const {
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& c = rows[i].card;
    int32_t* o = &obs.cards[i * CARD_COLS];
    uint32_t loc = c.location;
    int pos;
    if (loc == LOCATION_MZONE) {
      switch (c.position) {
        case POS_FACEUP_ATTACK: pos = 1; break;
        case POS_FACEDOWN_ATTACK: pos = 2; break;
        case POS_FACEUP_DEFENSE: pos = 3; break;
        case POS_FACEDOWN_DEFENSE: pos = 4; break;
        default: pos = 0;
      }
    } else if (loc == LOCATION_HAND || loc == LOCATION_DECK || loc == LOCATION_OVERLAY) {
      pos = 0;
    } else {
      pos = (c.position & POS_FACEUP) ? 5 : 6;
    }
    o[1] = loc_id(loc);
    o[2] = loc == LOCATION_DECK ? 0 : static_cast<int32_t>(c.sequence) + 1;
    o[3] = static_cast<int32_t>(c.controller ^ me);
    o[5] = pos;
    o[6] = is_public(c);
    obs.card_mask[i] = 1;
    if (!rows[i].visible) {
      // Privileged critic (RL v2 part B): the true code of every hidden row. The policy must
      // never read this; it only reaches the critic via critic_obs (card_known'/known_code').
      obs.priv_code[i] = i32(c.code);
      continue;
    }
    o[0] = i32(c.code);
    o[4] = static_cast<int32_t>((loc != LOCATION_OVERLAY ? c.owner : c.controller) ^ me);
    uint64_t race = (c.race & 0xFFFFFFFFu) | (((c.race >> 62) & 1) << 31);
    o[7] = i32(c.type);
    o[8] = i32(c.attribute);
    o[9] = i32(race);
    o[10] = i32(c.level);
    o[11] = i32(c.rank);
    o[12] = i32(c.link);
    o[13] = i32(c.link_marker);
    o[14] = c.attack;
    o[15] = c.defense;
    o[16] = c.base_attack;
    o[17] = c.base_defense;
    o[18] = i32(c.lscale | c.rscale << 8);
    o[19] = i32(c.status & STATUS_OBS_MASK);
  }
  for (auto& ref : selected) {
    int i = row_of(&ref);
    if (i >= 0) obs.cards[i * CARD_COLS + 20] = 1;
  }
  // Card-knowledge memory (obs v2): per card row, 1 = opponent's hidden card whose identity the
  // viewer remembers (with the remembered passcode), 2 = own hidden card known to the opponent.
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& c = rows[i].card;
    if (c.location != LOCATION_HAND || c.controller > 1) continue;
    const auto& hk = duel.hand_known_[c.controller];
    if (c.sequence >= hk.size() || !hk[c.sequence]) continue;
    if (static_cast<int>(c.controller) == me) {
      obs.card_known[i] = 2;  // my own hand is visible to me; the flag is about the opponent
    } else if (!rows[i].visible) {
      obs.card_known[i] = 1;
      obs.known_code[i] = static_cast<int32_t>(c.code);
    }
  }
  for (size_t i = 0; i < std::min<size_t>(chain_.size(), MAX_CHAIN); ++i) {
    const auto& ch = chain_[i];
    CardRef ref{ch.code, ch.controller, ch.location, ch.sequence, ch.position};
    int32_t* o = &obs.chain[i * CHAIN_COLS];
    o[0] = row_of(&ref);
    o[1] = i32(ch.code);
    o[2] = i32(ch.description >> 20);
    o[3] = static_cast<int32_t>(ch.description & 0xFFFFF);
    o[4] = static_cast<int32_t>(ch.trig_controller ^ me);
    obs.chain_mask[i] = 1;
  }
  int opp = 1 - me;
  auto& g = obs.global;
  g[0] = i32(lp_[me]);
  g[1] = i32(lp_[opp]);
  g[2] = duel.turn;
  g[3] = duel.phase;
  g[4] = duel.first_player == me;
  g[5] = duel.turn_player == me;
  int k = 6;
  for (int p : {me, opp})
    for (uint32_t loc : QUERY_LOCATIONS) g[k++] = counts_.at(key(p, loc, 0, 0));
  g[20] = static_cast<int32_t>(chain_.size());
  g[21] = msg_type;
  g[22] = truncated;
  g[23] = OBS_VERSION;
}

HistoryEntry history_entry(const Board& board, const Option& opt, int msg_type, int step, int turn, int phase) {
  HistoryEntry e{board.option_row(opt, msg_type, step), board.me, turn, phase, false};
  e.row[1] = -1;
  if (revealing(opt.kind)) {
    e.is_public = true;
  } else if (!opt.has_card) {
    e.is_public = !(opt.kind == POSITION && (opt.arg & POS_FACEDOWN));
  } else {
    int r = board.row_of(&opt.card);
    e.is_public = opt.card.controller != static_cast<uint32_t>(board.me) ||
                  (r >= 0 && Board::is_public(board.rows[r].card));
  }
  return e;
}

// ---------------------------------------------------------------- E5b light replay path
//
// The full Board costs 14 query_location calls per decision, but an intermediate replay
// step only needs, for the chosen option (plain replay) or every option (replay_keys key
// match), the equivalent of Board::option_code (visibility of that one card) and the
// history entry's is_public flag. Both derive from the single CardState of the option's
// card, so one location query per distinct (controller, location) per step suffices.
//
// Exactness (replay/bit-identity tests verify it): with the truncation guard below, every
// card the light path can observe maps 1:1 to a Board row with the same CardState:
// - a card owned by the viewer is always in rows (every own location incl. deck, plus own
//   overlay materials) with visible=true;
// - an opponent card in DECK is never in rows and an opponent card in EXTRA face-down is
//   skipped by the Board ctor; both fall into Board::option_code's hidden-pile fallback,
//   which reports opt.code unconditionally, so no query is needed there;
// - every other card present in its location query is in rows (ROW_LOCS loop, overlay
//   rows); a card absent from its query reproduces Board's r<0 fallback exactly.
// Truncation guard: rows hold distinct physical cards (each card lives in exactly one
// location; own-deck and overlay rows don't overlap location rows), so rows <= physical
// cards = main0+extra0+main1+extra1+setup + Duel-created cards (tokens). Replays with
// more physical cards than MAX_CARDS (all our decks are <= 150; that needs 10+ tokens in
// one game) keep the full Board: light_ok() is false and LightScope stays inactive.
// Own-deck choices stay exact even past the guard (Board fallback for the viewer reports
// the code either way), but the guard conservatively disables the whole path.

namespace {
uint64_t light_nphys(const DuelConfig& c) {
  return c.main0.size() + c.extra0.size() + c.main1.size() + c.extra1.size() + c.setup.size();
}
}  // namespace

struct LightScope {
  Env* e;
  explicit LightScope(Env* e_) : e(e_) {
    e->light_ = light_nphys(e->config_) <= static_cast<uint64_t>(MAX_CARDS);
  }
  ~LightScope() { e->light_ = false; }
  void stop() { e->light_ = false; }
};

const std::vector<std::optional<CardState>>& Env::light_loc(uint32_t con, uint32_t loc, LocCache& cache) {
  uint64_t k = (uint64_t{con} << 32) | loc;
  for (auto& [ck, cards] : cache)
    if (ck == k) return cards;
  uint64_t t0 = prof_on_ ? now_us() : 0;
  auto cards = duel_->query_location(static_cast<uint8_t>(con), loc);
  if (prof_on_) {
    prof_.qloc_us += now_us() - t0;
    ++prof_.lqueries;
  }
  cache.emplace_back(k, std::move(cards));
  return cache.back().second;
}

std::pair<uint32_t, bool> Env::light_opt(const Option& opt, LocCache& cache) {
  // {code column, entry publicity for a non-revealing kind}. Revealing kinds are always
  // public; the caller applies that (history_entry) or doesn't need it (key match).
  int me = dec_->player;
  if (!opt.has_card) return {opt.code, !(opt.kind == POSITION && (opt.arg & POS_FACEDOWN))};
  uint32_t con = opt.card.controller, loc = opt.card.location;
  bool mine = con == static_cast<uint32_t>(me);
  if (!mine && (loc == LOCATION_DECK || loc == LOCATION_EXTRA)) return {opt.code, true};
  if (loc & LOCATION_OVERLAY) {
    // Overlay materials live in the MZONE card's overlay list; the ref's position is the
    // material index (see ref_key). A bogus index reproduces the r<0 fallback exactly.
    bool found = false;
    for (auto& o : light_loc(con, LOCATION_MZONE, cache))
      if (o && o->sequence == opt.card.sequence) {
        found = opt.card.position < o->overlay.size();
        break;
      }
    if (!found) return {(mine || loc == LOCATION_DECK || loc == LOCATION_EXTRA) ? opt.code : 0, !mine};
    return {opt.code, true};
  }
  const CardState* c = nullptr;
  for (auto& o : light_loc(con, loc, cache))
    if (o && o->sequence == opt.card.sequence) {
      c = &*o;
      break;
    }
  if (!c) {
    // Same query the Board would run, same absence: exact r<0 fallback.
    return {(mine || loc == LOCATION_DECK || loc == LOCATION_EXTRA) ? opt.code : 0, !mine};
  }
  if (loc == LOCATION_EXTRA && !mine && !(c->position & POS_FACEUP)) return {opt.code, true};
  bool pub = Board::is_public(*c);
  return {(mine || pub) ? opt.code : 0, !mine || pub};
}

HistoryEntry Env::light_history(const Option& opt, int msg_type, int step, int turn, int phase,
                                 LocCache& cache) {
  auto [code, pub] = light_opt(opt, cache);
  HistoryEntry e{{static_cast<int32_t>(opt.kind),
                  -1,
                  static_cast<int32_t>(code),
                  i32(opt.desc >> 20),
                  static_cast<int32_t>(opt.desc & 0xFFFFF),
                  i32(static_cast<uint64_t>(opt.arg)),
                  msg_type,
                  opt.has_card ? loc_id(opt.card.location) : 0,
                  opt.has_card ? static_cast<int32_t>(opt.card.controller ^ dec_->player) : -1,
                  step},
                 dec_->player,
                 turn,
                 phase,
                 revealing(opt.kind) || pub};
  return e;
}

void write_history_row(int32_t* o, const HistoryEntry& e, int me, int turn) {
  std::copy(e.row.begin(), e.row.end(), o);
  if (o[8] >= 0) o[8] ^= e.actor ^ me;
  if (e.actor != me && !e.is_public) o[2] = o[3] = o[4] = 0;
  o[10] = e.actor ^ me;
  o[11] = turn - e.turn;
  o[12] = e.phase;
}

// ---------------------------------------------------------------- Env

Env::Env(Engine& engine, DuelConfig config, bool auto_single)
    : engine_(engine), config_(std::move(config)), auto_single_(auto_single) {}

bool Env::reset(Obs& obs) {
  board_.reset();
  dec_.reset();
  duel_ = std::make_unique<Duel>(engine_, config_);
  history_.clear();
  choices.clear();
  decisions = truncated = 0;
  next_decision();
  return settle(obs);
}

bool Env::step(int index, Obs& obs) {
  if (index < 0 || index >= num_options()) throw std::out_of_range("option index");
  choices.push_back(index);
  take(index, true);
  return settle(obs);
}

std::vector<std::array<int32_t, HLOG_COLS>> Env::history_log() const {
  std::vector<std::array<int32_t, HLOG_COLS>> out(history_.size());
  for (size_t i = 0; i < history_.size(); ++i) {
    const auto& e = history_[i];
    std::copy(e.row.begin(), e.row.end(), out[i].begin());
    out[i][10] = e.actor;
    out[i][11] = e.turn;
    out[i][12] = e.phase;
    out[i][13] = e.is_public;
  }
  return out;
}

void Env::next_decision() {
  const Bytes* msg;
  {
    Span s(prof_.advance_us, prof_on_);
    msg = duel_->advance();
  }
  if (!msg) {
    dec_.reset();
    return;
  }
  int p = (*msg)[1];
  Span s(prof_.decide_us, prof_on_);
  dec_ = make_decision(*msg, engine_, [this, p] {
    if (prof_on_) ++prof_.declares;
    return known_codes(p);
  });
}

std::set<uint32_t> Env::known_codes(int p) {
  std::set<uint32_t> codes;
  const auto& main = p == 0 ? config_.main0 : config_.main1;
  const auto& extra = p == 0 ? config_.extra0 : config_.extra1;
  codes.insert(main.begin(), main.end());
  codes.insert(extra.begin(), extra.end());
  auto vis = Board(*duel_, p).visible_codes();
  codes.insert(vis.begin(), vis.end());
  for (auto& e : history_)
    if (e.actor == p || e.is_public) codes.insert(static_cast<uint32_t>(e.row[2]));
  codes.erase(0);
  return codes;
}

void Env::take(int index, bool record) {
  const Option& opt = opts_[index];
  if (record) {
    Span s(prof_.history_us, prof_on_);
    if (light_) {
      LocCache cache;
      history_.push_back(light_history(opt, dec_->msg_type, dec_->step, duel_->turn, duel_->phase, cache));
    } else {
      history_.push_back(history_entry(*board_, opt, dec_->msg_type, dec_->step, duel_->turn, duel_->phase));
    }
  }
  ++decisions;
  Bytes resp;
  bool done = false;
  {
    Span s(prof_.respond_us, prof_on_);
    done = dec_->choose(index, resp);
    if (done) {
      duel_->respond(resp);
    }
  }
  if (done) next_decision();
}

bool Env::settle_fast() {
  while (dec_) {
    std::vector<Option> opts;
    {
      Span s(prof_.options_us, prof_on_);
      opts = dec_->options();
    }
    if (opts.empty()) throw std::runtime_error("no legal option");
    if (auto_single_ && opts.size() == 1) {
      if (prof_on_) ++prof_.autos;
      opts_ = std::move(opts);
      take(0, false);
      continue;
    }
    if (prof_on_) ++prof_.steps;
    truncated += opts.size() > MAX_OPTIONS;
    if (opts.size() > MAX_OPTIONS) opts.resize(MAX_OPTIONS);
    opts_ = std::move(opts);
    if (!light_) board_ = std::make_unique<Board>(*duel_, dec_->player, prof_on_ ? &prof_ : nullptr);
    return true;
  }
  return false;
}

void Env::fill_obs(Obs& obs) {
  Span s(prof_.fill_us, prof_on_);
  if (prof_on_) ++prof_.fills;
  obs.clear();
  board_->fill(obs, *duel_, dec_->msg_type, dec_->selected);
  for (size_t i = 0; i < opts_.size(); ++i) {
    auto row = board_->option_row(opts_[i], dec_->msg_type, dec_->step);
    std::copy(row.begin(), row.end(), &obs.actions[i * ACTION_COLS]);
    obs.action_mask[i] = 1;
  }
  int me = dec_->player;
  size_t n = std::min<size_t>(history_.size(), HISTORY);
  for (size_t i = 0; i < n; ++i) {
    const auto& e = history_[history_.size() - 1 - i];
    write_history_row(&obs.history[i * HISTORY_COLS], e, me, duel_->turn);
    obs.history_mask[i] = 1;
  }
  size_t m = 0;  // long window over salient entries only (most recent first)
  for (size_t i = history_.size(); i-- > 0 && m < HISTORY_MAX;) {
    const auto& e = history_[i];
    if (!is_salient(static_cast<Kind>(e.row[0]))) continue;
    write_history_row(&obs.hist_long[m * HISTORY_COLS], e, me, duel_->turn);
    obs.hist_long_mask[m++] = 1;
  }
}

bool Env::settle(Obs& obs) {
  if (!settle_fast()) return false;
  fill_obs(obs);
  return true;
}

bool Env::replay(const std::vector<int>& choices, Obs& obs) {
  prof_on_ = prof_on();
  prof_.clear();
  LightScope light(this);
  board_.reset();
  dec_.reset();
  duel_ = std::make_unique<Duel>(engine_, config_);
  history_.clear();
  this->choices.clear();
  decisions = truncated = 0;
  next_decision();
  for (size_t k = 0; k < choices.size(); ++k) {
    if (!settle_fast()) {
      if (prof_on_) prof_.dump("replay-early");
      return false;  // duel ended before choice k (step-loop would misbehave here)
    }
    int index = choices[k];
    if (index < 0 || index >= num_options()) throw std::out_of_range("option index");
    this->choices.push_back(index);
    take(index, true);
  }
  light.stop();  // the final decision needs the full Board for fill_obs
  bool ok = settle(obs);
  if (prof_on_) prof_.dump("replay");
  return ok;
}

Env::ReplayKeysResult Env::replay_keys(const std::vector<std::array<int32_t, 8>>& keys,
                                       const std::vector<int>& ranks, Obs& obs) {
  ReplayKeysResult r;
  prof_on_ = prof_on();
  prof_.clear();
  LightScope light(this);
  board_.reset();
  dec_.reset();
  duel_ = std::make_unique<Duel>(engine_, config_);
  history_.clear();
  choices.clear();
  decisions = truncated = 0;
  next_decision();
  for (size_t k = 0; k < keys.size(); ++k) {
    if (!settle_fast()) {
      r.reason = "replay_error";  // duel ended before key k was consumed
      return r;
    }
    // find_by_key_rank over option_row columns [0,2,3,4,5,6,7,8]
    const auto& key = keys[k];
    int rank = ranks[k];
    int last = -1, at_rank = -1;
    int seen = 0;
    {
      Span s(prof_.keymatch_us, prof_on_);
      if (prof_on_) prof_.keys += opts_.size();
      LocCache cache;  // one query per distinct (controller, location) in this decision
      for (size_t j = 0; j < opts_.size(); ++j) {
        const Option& o = opts_[j];
        int32_t code = 0;
        if (light_) {
          code = static_cast<int32_t>(light_opt(o, cache).first);
        } else {
          auto row = board_->option_row(o, dec_->msg_type, dec_->step);
          if (!(row[0] == key[0] && row[2] == key[1] && row[3] == key[2] && row[4] == key[3] &&
                row[5] == key[4] && row[6] == key[5] && row[7] == key[6] && row[8] == key[7]))
            continue;
          if (seen == rank) at_rank = static_cast<int>(j);
          last = static_cast<int>(j);
          ++seen;
          continue;
        }
        if (static_cast<int32_t>(o.kind) == key[0] && code == key[1] &&
            static_cast<int32_t>(o.desc >> 20) == key[2] &&
            static_cast<int32_t>(o.desc & 0xFFFFF) == key[3] &&
            static_cast<int32_t>(static_cast<uint64_t>(o.arg)) == key[4] && dec_->msg_type == key[5] &&
            (o.has_card ? loc_id(o.card.location) : 0) == key[6] &&
            (o.has_card ? static_cast<int32_t>(o.card.controller ^ dec_->player) : -1) == key[7]) {
          if (seen == rank) at_rank = static_cast<int>(j);
          last = static_cast<int>(j);
          ++seen;
        }
      }
    }
    int idx = -1;
    if (seen == 0) {
      r.reason = "no_key";
      return r;
    }
    idx = (rank < seen) ? at_rank : last;
    r.resolved.push_back(idx);
    choices.push_back(idx);
    try {
      take(idx, true);
    } catch (...) {
      // NB: choices keeps the entry, exactly like step() (push before take) on a throw.
      r.reason = "replay_error";
      return r;
    }
  }
  light.stop();  // the final decision needs the full Board for fill_obs
  if (!settle_fast()) return r;  // all keys consumed but the duel ended: ok=false, reason "" (caller obs-checks None)
  fill_obs(obs);
  r.ok = true;
  if (prof_on_) prof_.dump("replay_keys");
  return r;
}

}  // namespace ygo
