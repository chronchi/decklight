// Port of src/ygo_zero/engine/{obs,env}.py.
#include <algorithm>
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

bool Board::is_public(const CardState& c) {
  if (c.location == LOCATION_GRAVE || c.location == LOCATION_OVERLAY) return true;
  return c.is_public || ((c.position & POS_FACEUP) && c.location != LOCATION_HAND);
}

Board::Board(Duel& duel, int me) : me(me) {
  std::unordered_map<uint64_t, std::vector<std::optional<CardState>>> q;
  for (int p = 0; p < 2; ++p)
    for (uint32_t loc : QUERY_LOCATIONS) {
      auto cards = duel.query_location(static_cast<uint8_t>(p), loc);
      counts_[key(p, loc, 0, 0)] = static_cast<int>(std::count_if(cards.begin(), cards.end(), [](auto& c) { return c.has_value(); }));
      q[key(p, loc, 0, 0)] = std::move(cards);
    }
  Bytes field = duel.query_field();
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
    if (!rows[i].visible) continue;
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

void Env::next_decision() {
  const Bytes* msg = duel_->advance();
  if (!msg) {
    dec_.reset();
    return;
  }
  int p = (*msg)[1];
  dec_ = make_decision(*msg, engine_, [this, p] { return known_codes(p); });
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
  if (record) history_.push_back(history_entry(*board_, opt, dec_->msg_type, dec_->step, duel_->turn, duel_->phase));
  ++decisions;
  Bytes resp;
  if (dec_->choose(index, resp)) {
    duel_->respond(resp);
    next_decision();
  }
}

bool Env::settle(Obs& obs) {
  while (dec_) {
    auto opts = dec_->options();
    if (opts.empty()) throw std::runtime_error("no legal option");
    if (auto_single_ && opts.size() == 1) {
      opts_ = std::move(opts);
      take(0, false);
      continue;
    }
    truncated += opts.size() > MAX_OPTIONS;
    if (opts.size() > MAX_OPTIONS) opts.resize(MAX_OPTIONS);
    opts_ = std::move(opts);
    board_ = std::make_unique<Board>(*duel_, dec_->player);
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
      int32_t* o = &obs.history[i * HISTORY_COLS];
      std::copy(e.row.begin(), e.row.end(), o);
      if (o[8] >= 0) o[8] ^= e.actor ^ me;
      if (e.actor != me && !e.is_public) o[2] = o[3] = o[4] = 0;
      o[10] = e.actor ^ me;
      o[11] = duel_->turn - e.turn;
      o[12] = e.phase;
      obs.history_mask[i] = 1;
    }
    return true;
  }
  return false;
}

}  // namespace ygo
