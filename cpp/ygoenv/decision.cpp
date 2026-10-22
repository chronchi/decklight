// Port of src/decklight/engine/{messages,actions,announce}.py (plus responses.py encodings).
#include <algorithm>
#include <cstring>
#include <stack>
#include <stdexcept>

#include "ygoenv.h"

namespace ygo {

namespace {

struct Reader {
  const Bytes& b;
  size_t p = 1;  // skip the type byte
  template <typename T>
  T get() {
    T v;
    std::memcpy(&v, b.data() + p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  uint8_t u8() { return get<uint8_t>(); }
  uint16_t u16() { return get<uint16_t>(); }
  uint32_t u32() { return get<uint32_t>(); }
  uint64_t u64() { return get<uint64_t>(); }
  CardRef card_info() {  // code + loc_info
    CardRef c;
    c.code = u32(); c.controller = u8(); c.location = u8(); c.sequence = u32(); c.position = u32();
    return c;
  }
  CardRef card_seq32() {
    CardRef c;
    c.code = u32(); c.controller = u8(); c.location = u8(); c.sequence = u32();
    return c;
  }
  CardRef card_seq8() {
    CardRef c;
    c.code = u32(); c.controller = u8(); c.location = u8(); c.sequence = u8();
    return c;
  }
};

// --- response encodings (responses.py)
Bytes int32(int32_t v) {
  Bytes b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}
Bytes command(int action, int index = 0) { return int32((index << 16) | action); }
Bytes cards_resp(std::vector<int> idx) {
  std::sort(idx.begin(), idx.end());
  Bytes b = int32(2);
  uint32_t n = static_cast<uint32_t>(idx.size());
  b.resize(8);
  std::memcpy(b.data() + 4, &n, 4);
  for (int i : idx) b.push_back(static_cast<uint8_t>(i));
  return b;
}
const Bytes CANCEL_RESP = int32(-1);

constexpr int IDLE_SUMMON = 0, IDLE_SPSUMMON = 1, IDLE_REPOS = 2, IDLE_MSET = 3, IDLE_SSET = 4,
              IDLE_ACTIVATE = 5, IDLE_BATTLE = 6, IDLE_END = 7;
constexpr int BATTLE_ACTIVATE = 0, BATTLE_ATTACK = 1, BATTLE_MAIN2 = 2, BATTLE_END = 3;

Option card_opt(Kind k, const CardRef& c, uint64_t desc = 0, int64_t arg = 0) {
  Option o{};
  o.kind = k;
  o.has_card = true;
  o.card = c;
  o.code = c.code;
  o.desc = desc;
  o.arg = arg;
  return o;
}
Option plain(Kind k, uint64_t desc = 0, int64_t arg = 0, uint32_t code = 0) {
  Option o{};
  o.kind = k;
  o.desc = desc;
  o.arg = arg;
  o.code = code;
  return o;
}

// ---------------------------------------------------------------- single-step messages

class Fixed : public Decision {
 public:
  using Decision::Decision;
  std::vector<Option> opts;
  std::vector<Bytes> resps;
  void add(Option o, Bytes r) {
    opts.push_back(o);
    resps.push_back(std::move(r));
  }
  std::vector<Option> options() override { return opts; }

 protected:
  bool apply(const Option&, int index, Bytes& response) override {
    response = resps[index];
    return true;
  }
};

std::unique_ptr<Decision> idle(Reader& r, int t) {
  auto d = std::make_unique<Fixed>(t, r.u8());
  const std::pair<Kind, int> kinds[] = {
      {SUMMON, IDLE_SUMMON}, {SPSUMMON, IDLE_SPSUMMON}, {REPOS, IDLE_REPOS}, {MSET, IDLE_MSET}, {SSET, IDLE_SSET}};
  for (auto [kind, act] : kinds) {
    uint32_t n = r.u32();
    for (uint32_t i = 0; i < n; ++i) {
      CardRef c = kind == REPOS ? r.card_seq8() : r.card_seq32();
      d->add(card_opt(kind, c), command(act, i));
    }
  }
  uint32_t n = r.u32();
  for (uint32_t i = 0; i < n; ++i) {
    CardRef c = r.card_seq32();
    uint64_t desc = r.u64();
    uint8_t mode = r.u8();
    d->add(card_opt(ACTIVATE, c, desc, mode), command(IDLE_ACTIVATE, i));
  }
  bool bp = r.u8(), ep = r.u8();
  if (bp) d->add(plain(TO_BATTLE), command(IDLE_BATTLE));
  if (ep) d->add(plain(TO_END), command(IDLE_END));
  return d;
}

std::unique_ptr<Decision> battle(Reader& r, int t) {
  auto d = std::make_unique<Fixed>(t, r.u8());
  uint32_t n = r.u32();
  for (uint32_t i = 0; i < n; ++i) {
    CardRef c = r.card_seq32();
    uint64_t desc = r.u64();
    uint8_t mode = r.u8();
    d->add(card_opt(ACTIVATE, c, desc, mode), command(BATTLE_ACTIVATE, i));
  }
  n = r.u32();
  for (uint32_t i = 0; i < n; ++i) {
    CardRef c = r.card_seq8();
    bool direct = r.u8();
    d->add(card_opt(ATTACK, c, 0, direct), command(BATTLE_ATTACK, i));
  }
  bool m2 = r.u8(), ep = r.u8();
  if (m2) d->add(plain(TO_MAIN2), command(BATTLE_MAIN2));
  if (ep) d->add(plain(TO_END), command(BATTLE_END));
  return d;
}

std::unique_ptr<Decision> chain(Reader& r, int t) {
  auto d = std::make_unique<Fixed>(t, r.u8());
  r.u8();  // spe_count
  bool forced = r.u8();
  r.u32();
  r.u32();
  uint32_t n = r.u32();
  for (uint32_t i = 0; i < n; ++i) {
    CardRef c = r.card_info();
    uint64_t desc = r.u64();
    uint8_t mode = r.u8();
    d->add(card_opt(ACTIVATE, c, desc, mode), int32(static_cast<int32_t>(i)));
  }
  if (!forced || n == 0) d->add(plain(PASS), CANCEL_RESP);
  return d;
}

// ---------------------------------------------------------------- card-set picks

class PickCards : public Decision {
 public:
  PickCards(int t, int player) : Decision(t, player) {}
  std::vector<CardRef> refs;
  std::vector<uint32_t> params;
  bool cancelable = false;
  std::vector<int> picked;

  bool is_picked(int i) const { return std::find(picked.begin(), picked.end(), i) != picked.end(); }
  std::vector<int> offered() {
    std::vector<int> out;
    for (int i = 0; i < static_cast<int>(refs.size()); ++i)
      if (!is_picked(i) && addable(i)) out.push_back(i);
    return out;
  }
  std::vector<Option> options() override {
    std::vector<Option> opts;
    for (int i : offered()) opts.push_back(card_opt(SELECT, refs[i], 0, params[i]));
    if (can_finish()) opts.push_back(plain(FINISH));
    if (cancelable && picked.empty()) opts.push_back(plain(CANCEL));
    return opts;
  }

 protected:
  virtual bool addable(int i) = 0;
  virtual bool can_finish() = 0;
  virtual bool auto_send() = 0;
  bool apply(const Option& opt, int index, Bytes& response) override {
    if (opt.kind == CANCEL) {
      response = CANCEL_RESP;
      return true;
    }
    if (opt.kind == FINISH) {
      response = cards_resp(picked);
      return true;
    }
    picked.push_back(offered()[index]);
    selected.push_back(opt.card);
    if (!auto_send()) return false;
    response = cards_resp(picked);
    return true;
  }
};

class SelectCard : public PickCards {
 public:
  uint32_t min = 0, max = 0;
  using PickCards::PickCards;

 protected:
  bool addable(int) override { return true; }
  bool can_finish() override { return min <= picked.size() && picked.size() < max; }
  bool auto_send() override { return picked.size() >= max; }
};

class SelectTribute : public PickCards {
 public:
  uint32_t min = 0, max = 0;
  using PickCards::PickCards;

 protected:
  int64_t total() const {
    int64_t s = 0;
    for (int i : picked) s += params[i];
    return s;
  }
  bool addable(int i) override {
    if (picked.size() + 1 > max) return false;
    int64_t need = static_cast<int64_t>(min) - total() - params[i];
    if (need <= 0) return true;
    size_t slots = max - picked.size() - 1;
    std::vector<uint32_t> rest;
    for (int j = 0; j < static_cast<int>(refs.size()); ++j)
      if (j != i && !is_picked(j)) rest.push_back(params[j]);
    std::sort(rest.rbegin(), rest.rend());
    int64_t s = 0;
    for (size_t k = 0; k < std::min(slots, rest.size()); ++k) s += rest[k];
    return s >= need;
  }
  bool can_finish() override { return total() >= min && picked.size() < max; }
  bool auto_send() override { return picked.size() >= max; }
};

// select_sum_check1 from playerop.cpp, as in random_agent._sum_check_exact
bool sum_exact(const std::vector<uint32_t>& params, int64_t acc, size_t i = 0) {
  if (acc == 0 || i == params.size()) return false;
  int64_t o1 = params[i] & 0xFFFF, o2 = params[i] >> 16;
  if (i == params.size() - 1) return acc == o1 || acc == o2;
  return (acc > o1 && sum_exact(params, acc - o1, i + 1)) || (o2 > 0 && acc > o2 && sum_exact(params, acc - o2, i + 1));
}
int64_t min_param(uint32_t p) {
  int64_t o1 = p & 0xFFFF, o2 = p >> 16;
  return (o2 && o2 < o1) ? o2 : o1;
}
bool sum_at_least(const std::vector<uint32_t>& params, int64_t acc) {
  int64_t total = 0, hi = 0, lo = 0x7FFFFFFF;
  for (uint32_t p : params) {
    int64_t o1 = p & 0xFFFF, o2 = p >> 16;
    int64_t ms = min_param(p);
    total += ms;
    hi += std::max(o1, o2);
    lo = std::min(lo, ms);
  }
  return !(hi < acc || total - lo >= acc);
}

class SelectSum : public PickCards {
 public:
  static constexpr int NODE_BUDGET = 200000;
  bool at_least = false;
  int64_t acc = 0;
  uint32_t min = 0, max = 0;
  std::vector<uint32_t> must_params;
  std::vector<std::vector<int>> valid;  // sorted index sets
  using PickCards::PickCards;

  bool is_valid(const std::vector<int>& chosen) const {
    std::vector<uint32_t> ps = must_params;
    for (int i : chosen) ps.push_back(params[i]);
    if (at_least) return sum_at_least(ps, acc);
    return min <= chosen.size() && chosen.size() <= max && sum_exact(ps, acc);
  }
  void enumerate() {
    int64_t must_min = 0;
    for (uint32_t p : must_params) must_min += min_param(p);
    int nodes = 0;
    std::vector<int> chosen;
    std::function<void(size_t, int64_t)> dfs = [&](size_t k, int64_t min_sum) {
      if (++nodes > NODE_BUDGET) return;
      if (!chosen.empty() && is_valid(chosen)) valid.push_back(chosen);
      if (k == refs.size()) return;
      if (at_least) {
        if (min_sum >= acc) return;
      } else if (chosen.size() >= max || min_sum >= acc) {
        return;
      }
      for (size_t j = k; j < refs.size(); ++j) {
        chosen.push_back(static_cast<int>(j));
        dfs(j + 1, min_sum + min_param(params[j]));
        chosen.pop_back();
      }
    };
    dfs(0, must_min);
    if (is_valid({})) valid.push_back({});
  }

 protected:
  bool addable(int i) override {
    std::vector<int> cur = picked;
    cur.push_back(i);
    std::sort(cur.begin(), cur.end());
    for (auto& s : valid)
      if (std::includes(s.begin(), s.end(), cur.begin(), cur.end())) return true;
    return false;
  }
  bool can_finish() override {
    std::vector<int> cur = picked;
    std::sort(cur.begin(), cur.end());
    return std::find(valid.begin(), valid.end(), cur) != valid.end();
  }
  bool auto_send() override { return false; }
};

// ---------------------------------------------------------------- other multi-step messages

class UnselectCard : public Decision {
 public:
  using Decision::Decision;
  bool finishable = false, cancelable = false;
  std::vector<CardRef> sel, unsel;
  std::vector<Option> options() override {
    std::vector<Option> o;
    for (auto& c : sel) o.push_back(card_opt(SELECT, c));
    for (auto& c : unsel) o.push_back(card_opt(UNSELECT, c));
    if (finishable) o.push_back(plain(FINISH));
    else if (cancelable) o.push_back(plain(CANCEL));
    return o;
  }

 protected:
  bool apply(const Option& opt, int index, Bytes& response) override {
    if (opt.kind == FINISH || opt.kind == CANCEL) {
      response = CANCEL_RESP;
    } else {
      response = int32(1);
      Bytes i = int32(index);
      response.insert(response.end(), i.begin(), i.end());
    }
    return true;
  }
};

class Place : public Decision {
 public:
  Place(int t, int player, uint8_t count, uint32_t flag) : Decision(t, player), count(count), taken(flag) {}
  uint8_t count;
  uint32_t taken;
  Bytes zones;
  std::vector<Option> options() override {
    std::vector<Option> o;
    for (int z = 0; z < 32; ++z) {
      int s = z & 15;
      if (s == 7) continue;  // MZ has 7 zones (0-6), SZ 8 (8-15)
      if (!(taken >> z & 1)) o.push_back(plain(PLACE, 0, z));
    }
    return o;
  }

 protected:
  bool apply(const Option& opt, int, Bytes& response) override {
    int z = static_cast<int>(opt.arg);
    taken |= 1u << z;
    int rel = z >> 4, s = z & 15;
    zones.push_back(static_cast<uint8_t>(player ^ rel));
    zones.push_back(s >= 8 ? LOCATION_SZONE : LOCATION_MZONE);
    zones.push_back(static_cast<uint8_t>(s >= 8 ? s - 8 : s));
    if (zones.size() / 3 < count) return false;
    response = zones;
    return true;
  }
};

class Bits : public Decision {
 public:
  Bits(int t, int player, bool race, uint8_t count, uint64_t available)
      : Decision(t, player), race(race), count(count), available(available) {}
  bool race;
  uint8_t count;
  uint64_t available, value = 0;
  std::vector<Option> options() override {
    std::vector<Option> o;
    uint64_t avail = available & ~value;
    for (int b = 0; b < (race ? 64 : 32); ++b)
      if (avail >> b & 1) o.push_back(plain(race ? RACE : ATTRIBUTE, 0, race ? b : (int64_t{1} << b)));
    return o;
  }

 protected:
  bool apply(const Option& opt, int, Bytes& response) override {
    value |= race ? (uint64_t{1} << opt.arg) : static_cast<uint64_t>(opt.arg);
    if (step < count) return false;
    response.resize(race ? 8 : 4);
    if (race) std::memcpy(response.data(), &value, 8);
    else {
      uint32_t v = static_cast<uint32_t>(value);
      std::memcpy(response.data(), &v, 4);
    }
    return true;
  }
};

class Counter : public Decision {
 public:
  using Decision::Decision;
  uint16_t count = 0;
  std::vector<CardRef> cards;
  std::vector<int> have, left;
  std::vector<Option> options() override {
    std::vector<Option> o;
    for (size_t i = 0; i < cards.size(); ++i)
      if (left[i] > 0) o.push_back(card_opt(COUNTER, cards[i], 0, left[i]));
    return o;
  }

 protected:
  bool apply(const Option&, int index, Bytes& response) override {
    int k = -1;
    for (size_t j = 0; j < left.size(); ++j)
      if (left[j] > 0 && ++k == index) {
        --left[j];
        break;
      }
    if (step < count) return false;
    response.clear();
    for (size_t j = 0; j < cards.size(); ++j) {
      int16_t v = static_cast<int16_t>(have[j] - left[j]);
      response.push_back(static_cast<uint8_t>(v & 0xFF));
      response.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    }
    return true;
  }
};

class Sort : public Decision {
 public:
  using Decision::Decision;
  std::vector<CardRef> cards;
  std::vector<int> order;
  std::vector<int> left() const {
    std::vector<int> l;
    for (int i = 0; i < static_cast<int>(cards.size()); ++i)
      if (std::find(order.begin(), order.end(), i) == order.end()) l.push_back(i);
    return l;
  }
  std::vector<Option> options() override {
    std::vector<Option> o;
    for (int i : left()) o.push_back(card_opt(SORT, cards[i]));
    return o;
  }

 protected:
  bool apply(const Option& opt, int index, Bytes& response) override {
    order.push_back(left()[index]);
    selected.push_back(opt.card);
    auto l = left();
    if (l.size() > 1) return false;
    order.insert(order.end(), l.begin(), l.end());
    response.assign(order.size(), 0);
    for (size_t r = 0; r < order.size(); ++r) response[order[r]] = static_cast<uint8_t>(r);
    return true;
  }
};

std::vector<uint32_t> declare_options(const std::vector<uint64_t>& opcodes, Engine& engine,
                                      const std::set<uint32_t>& pool) {
  std::vector<uint32_t> out;
  for (uint32_t c : pool) {  // std::set iterates sorted
    auto it = engine.cards.find(c);
    if (it != engine.cards.end() && is_declarable(it->second, opcodes)) out.push_back(c);
  }
  if (!out.empty()) return out;
  for (uint32_t c : engine.sorted_codes)
    if (is_declarable(engine.cards.at(c), opcodes)) return {c};
  throw std::runtime_error("no declarable card in the card DB");
}

}  // namespace

bool Decision::choose(int index, Bytes& response) {
  auto opts = options();
  if (index < 0 || index >= static_cast<int>(opts.size())) throw std::out_of_range("option index");
  ++step;
  return apply(opts[index], index, response);
}

bool is_declarable(const CardData& cd, const std::vector<uint64_t>& opcodes) {
  std::stack<int64_t> st;
  bool alias = false, token = false;
  auto bin = [&](auto f) {
    if (st.size() >= 2) {
      int64_t rhs = st.top(); st.pop();
      int64_t lhs = st.top(); st.pop();
      st.push(f(lhs, rhs));
    }
  };
  auto un = [&](auto f) {
    if (!st.empty()) {
      int64_t v = st.top(); st.pop();
      st.push(f(v));
    }
  };
  for (uint64_t op : opcodes) {
    switch (op) {
      case OPCODE_ADD: bin([](int64_t a, int64_t b) { return a + b; }); break;
      case OPCODE_SUB: bin([](int64_t a, int64_t b) { return a - b; }); break;
      case OPCODE_MUL: bin([](int64_t a, int64_t b) { return a * b; }); break;
      case OPCODE_DIV: bin([](int64_t a, int64_t b) { return b == 0 ? int64_t{0} : a / b; }); break;
      case OPCODE_AND: bin([](int64_t a, int64_t b) { return int64_t{a && b}; }); break;
      case OPCODE_OR: bin([](int64_t a, int64_t b) { return int64_t{a || b}; }); break;
      case OPCODE_NEG: un([](int64_t v) { return -v; }); break;
      case OPCODE_NOT: un([](int64_t v) { return int64_t{!v}; }); break;
      case OPCODE_BAND: bin([](int64_t a, int64_t b) { return a & b; }); break;
      case OPCODE_BOR: bin([](int64_t a, int64_t b) { return a | b; }); break;
      case OPCODE_BXOR: bin([](int64_t a, int64_t b) { return a ^ b; }); break;
      case OPCODE_BNOT: un([](int64_t v) { return ~v; }); break;
      case OPCODE_LSHIFT: bin([](int64_t a, int64_t b) { return a << b; }); break;
      case OPCODE_RSHIFT: bin([](int64_t a, int64_t b) { return a >> b; }); break;
      case OPCODE_ISCODE: un([&](int64_t v) { return int64_t{cd.code == static_cast<uint32_t>(v)}; }); break;
      case OPCODE_ISTYPE: un([&](int64_t v) { return static_cast<int64_t>(cd.type & v); }); break;
      case OPCODE_ISRACE: un([&](int64_t v) { return static_cast<int64_t>(cd.race & v); }); break;
      case OPCODE_ISATTRIBUTE: un([&](int64_t v) { return static_cast<int64_t>(cd.attribute & v); }); break;
      case OPCODE_GETCODE: st.push(cd.code); break;
      case OPCODE_GETTYPE: st.push(cd.type); break;
      case OPCODE_GETRACE: st.push(static_cast<int64_t>(cd.race)); break;
      case OPCODE_GETATTRIBUTE: st.push(cd.attribute); break;
      case OPCODE_ISSETCARD:
        if (!st.empty()) {
          int32_t set_code = static_cast<int32_t>(st.top());
          st.pop();
          uint16_t settype = set_code & 0xfff, subtype = set_code & 0xf000;
          bool res = false;
          for (uint16_t sc : cd.setcodes)
            if (sc && (sc & 0xfff) == settype && (sc & 0xf000 & subtype) == subtype) res = true;
          st.push(res);
        }
        break;
      case OPCODE_ALLOW_ALIASES: alias = true; break;
      case OPCODE_ALLOW_TOKENS: token = true; break;
      default: st.push(static_cast<int64_t>(op)); break;
    }
  }
  if (st.size() != 1 || st.top() == 0) return false;
  if (cd.code == CARD_MARINE_DOLPHIN || cd.code == CARD_TWINKLE_MOSS) return true;
  bool is_token = (cd.type & (TYPE_MONSTER | TYPE_TOKEN)) == (TYPE_MONSTER | TYPE_TOKEN);
  return (alias || !cd.alias) && (token || !is_token);
}

std::unique_ptr<Decision> make_decision(const Bytes& msg, Engine& engine, const CodePool& pool) {
  int t = msg[0];
  Reader r{msg};
  switch (t) {
    case MSG_SELECT_IDLECMD: return idle(r, t);
    case MSG_SELECT_BATTLECMD: return battle(r, t);
    case MSG_SELECT_CHAIN: return chain(r, t);
    case MSG_SELECT_EFFECTYN: {
      auto d = std::make_unique<Fixed>(t, r.u8());
      CardRef c = r.card_info();
      uint64_t desc = r.u64();
      d->add(card_opt(YES, c, desc), int32(1));
      d->add(card_opt(NO, c, desc), int32(0));
      return d;
    }
    case MSG_SELECT_YESNO: {
      auto d = std::make_unique<Fixed>(t, r.u8());
      uint64_t desc = r.u64();
      d->add(plain(YES, desc), int32(1));
      d->add(plain(NO, desc), int32(0));
      return d;
    }
    case MSG_SELECT_OPTION: {
      auto d = std::make_unique<Fixed>(t, r.u8());
      int n = r.u8();
      for (int i = 0; i < n; ++i) d->add(plain(OPTION, r.u64(), i), int32(i));
      return d;
    }
    case MSG_SELECT_CARD: {
      int p = r.u8();
      auto d = std::make_unique<SelectCard>(t, p);
      d->cancelable = r.u8();
      d->min = r.u32();
      uint32_t mx = r.u32();
      uint32_t n = r.u32();
      for (uint32_t i = 0; i < n; ++i) d->refs.push_back(r.card_info());
      d->params.assign(n, 0);
      d->max = std::min(mx, n);
      return d;
    }
    case MSG_SELECT_TRIBUTE: {
      int p = r.u8();
      auto d = std::make_unique<SelectTribute>(t, p);
      d->cancelable = r.u8();
      d->min = r.u32();
      d->max = r.u32();
      uint32_t n = r.u32();
      for (uint32_t i = 0; i < n; ++i) {
        d->refs.push_back(r.card_seq32());
        d->params.push_back(r.u8());
      }
      return d;
    }
    case MSG_SELECT_SUM: {
      int p = r.u8();
      auto d = std::make_unique<SelectSum>(t, p);
      d->at_least = r.u8();
      d->acc = r.u32();
      d->min = r.u32();
      d->max = r.u32();
      uint32_t n = r.u32();
      for (uint32_t i = 0; i < n; ++i) {
        d->selected.push_back(r.card_info());
        d->must_params.push_back(r.u32());
      }
      n = r.u32();
      for (uint32_t i = 0; i < n; ++i) {
        d->refs.push_back(r.card_info());
        d->params.push_back(r.u32());
      }
      d->enumerate();
      return d;
    }
    case MSG_SELECT_UNSELECT_CARD: {
      int p = r.u8();
      auto d = std::make_unique<UnselectCard>(t, p);
      d->finishable = r.u8();
      d->cancelable = r.u8();
      r.u32();
      r.u32();
      uint32_t n = r.u32();
      for (uint32_t i = 0; i < n; ++i) d->sel.push_back(r.card_info());
      n = r.u32();
      for (uint32_t i = 0; i < n; ++i) d->unsel.push_back(r.card_info());
      return d;
    }
    case MSG_SELECT_PLACE:
    case MSG_SELECT_DISFIELD: {
      int p = r.u8();
      uint8_t count = r.u8();
      return std::make_unique<Place>(t, p, count, r.u32());
    }
    case MSG_SELECT_POSITION: {
      auto d = std::make_unique<Fixed>(t, r.u8());
      uint32_t code = r.u32();
      uint8_t positions = r.u8();
      for (int b : {1, 2, 4, 8})
        if (positions & b) d->add(plain(POSITION, 0, b, code), int32(b));
      return d;
    }
    case MSG_ANNOUNCE_NUMBER: {
      auto d = std::make_unique<Fixed>(t, r.u8());
      int n = r.u8();
      for (int i = 0; i < n; ++i) d->add(plain(NUMBER, 0, static_cast<int64_t>(r.u64())), int32(i));
      return d;
    }
    case MSG_ANNOUNCE_RACE: {
      int p = r.u8();
      uint8_t count = r.u8();
      return std::make_unique<Bits>(t, p, true, count, r.u64());
    }
    case MSG_ANNOUNCE_ATTRIB: {
      int p = r.u8();
      uint8_t count = r.u8();
      return std::make_unique<Bits>(t, p, false, count, r.u32());
    }
    case MSG_ANNOUNCE_CARD: {
      auto d = std::make_unique<Fixed>(t, r.u8());
      int n = r.u8();
      std::vector<uint64_t> ops(n);
      for (auto& op : ops) op = r.u64();
      for (uint32_t c : declare_options(ops, engine, pool()))
        d->add(plain(DECLARE, 0, 0, c), int32(static_cast<int32_t>(c)));
      return d;
    }
    case MSG_SORT_CARD:
    case MSG_SORT_CHAIN: {
      auto d = std::make_unique<Sort>(t, r.u8());
      uint32_t n = r.u32();
      for (uint32_t i = 0; i < n; ++i) {
        CardRef c;
        c.code = r.u32(); c.controller = r.u8(); c.location = r.u32(); c.sequence = r.u32();
        d->cards.push_back(c);
      }
      return d;
    }
    case MSG_SELECT_COUNTER: {
      auto d = std::make_unique<Counter>(t, r.u8());
      r.u16();  // counter type
      d->count = r.u16();
      uint32_t n = r.u32();
      for (uint32_t i = 0; i < n; ++i) {
        d->cards.push_back(r.card_seq8());
        d->have.push_back(r.u16());
      }
      d->left = d->have;
      return d;
    }
    case MSG_ROCK_PAPER_SCISSORS: {
      auto d = std::make_unique<Fixed>(t, r.u8());
      for (int v : {1, 2, 3}) d->add(plain(RPS, 0, v), int32(v));
      return d;
    }
  }
  throw std::runtime_error("unhandled select message " + std::to_string(t));
}

}  // namespace ygo
