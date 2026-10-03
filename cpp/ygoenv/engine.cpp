// Engine (card data + scripts) and Duel: port of src/ygo_zero/engine/{resources,duel,query}.py.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "lauxlib.h"
#include "lua.h"
#include "ygoenv.h"

namespace ygo {

// Precompile a script with the core's own Lua build (same liblua.a, so the bytecode format matches) and keep
// debug info, so errors still name the file and line. The core loads it with luaL_loadbuffer (mode "bt").
// Parsing ~700 KB of system/proc scripts was most of the cost of creating a duel (NOTEBOOK 2026-10-01).
static std::string compile_lua(const std::string& src, const std::string& name) {
  lua_State* L = luaL_newstate();
  std::string out;
  if (luaL_loadbuffer(L, src.data(), src.size(), name.c_str()) == LUA_OK) {
    lua_dump(
        L, [](lua_State*, const void* p, size_t n, void* ud) {
          static_cast<std::string*>(ud)->append(static_cast<const char*>(p), n);
          return 0;
        },
        &out, 0);
  }
  lua_close(L);
  return out.empty() ? src : out;  // on a syntax error keep the source: the core reports it as usual
}

void Engine::add_card(CardData cd) {
  if (cd.setcodes.empty() || cd.setcodes.back() != 0) cd.setcodes.push_back(0);
  cards[cd.code] = std::move(cd);
}

void Engine::finalize() {
  sorted_codes.clear();
  for (auto& [code, _] : cards) sorted_codes.push_back(code);
  std::sort(sorted_codes.begin(), sorted_codes.end());
}

const std::string* Engine::script(const std::string& name) {
  {
    std::shared_lock lock(mu_);
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second.get();
  }
  auto path = script_paths.find(name);
  if (path == script_paths.end()) return nullptr;
  std::ifstream f(path->second, std::ios::binary);
  if (!f) return nullptr;
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string code = precompile ? compile_lua(ss.str(), name) : ss.str();
  std::unique_lock lock(mu_);
  auto& slot = cache_[name];
  if (!slot) slot = std::make_unique<std::string>(std::move(code));
  return slot.get();
}

static std::array<uint64_t, 4> seed_words(uint64_t x) {
  std::array<uint64_t, 4> out;
  for (auto& w : out) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    w = z ^ (z >> 31);
  }
  return out;
}

void Duel::read_card(void* payload, uint32_t code, OCG_CardData* d) {
  auto& cards = static_cast<Engine*>(payload)->cards;
  auto it = cards.find(code);
  if (it == cards.end()) return;
  const CardData& cd = it->second;
  d->code = cd.code;
  d->alias = cd.alias;
  d->setcodes = const_cast<uint16_t*>(cd.setcodes.data());
  d->type = cd.type;
  d->level = cd.level;
  d->attribute = cd.attribute;
  d->race = cd.race;
  d->attack = cd.attack;
  d->defense = cd.defense;
  d->lscale = cd.lscale;
  d->rscale = cd.rscale;
  d->link_marker = cd.link_marker;
}

int Duel::read_script(void* payload, OCG_Duel duel, const char* name) {
  auto* self = static_cast<Duel*>(payload);
  const std::string* src = self->engine.script(name);
  if (!src) return 0;
  return OCG_LoadScript(duel, src->data(), static_cast<uint32_t>(src->size()), name) == 1;
}

void Duel::log(void* payload, const char* str, int type) {
  if (type == OCG_LOG_TYPE_ERROR) static_cast<Duel*>(payload)->errors.emplace_back(str);
}

bool Duel::load_script(const std::string& name) {
  const std::string* src = engine.script(name);
  return src && OCG_LoadScript(handle_, src->data(), static_cast<uint32_t>(src->size()), name.c_str()) == 1;
}

Duel::Duel(Engine& engine, const DuelConfig& config) : engine(engine), config(config) {
  OCG_DuelOptions opts{};
  auto seed = seed_words(config.seed);
  std::copy(seed.begin(), seed.end(), opts.seed);
  opts.flags = config.flags;
  for (auto* team : {&opts.team1, &opts.team2}) {
    team->startingLP = config.lp;
    team->startingDrawCount = config.hand;
    team->drawCountPerTurn = config.draw;
  }
  opts.cardReader = &Duel::read_card;
  opts.payload1 = &engine;
  opts.scriptReader = &Duel::read_script;
  opts.payload2 = this;
  opts.logHandler = &Duel::log;
  opts.payload3 = this;
  opts.cardReaderDone = [](void*, OCG_CardData*) {};
  opts.enableUnsafeLibraries = 0;
  int status = OCG_CreateDuel(&handle_, &opts);
  if (status != OCG_DUEL_CREATION_SUCCESS) throw std::runtime_error("OCG_CreateDuel failed: " + std::to_string(status));
  for (const char* name : {"constant.lua", "utility.lua"})
    if (!load_script(name)) throw std::runtime_error(std::string("cannot load ") + name);
  add_cards(0, LOCATION_DECK, config.main0);
  add_cards(0, LOCATION_EXTRA, config.extra0);
  add_cards(1, LOCATION_DECK, config.main1);
  add_cards(1, LOCATION_EXTRA, config.extra1);
  for (const auto& pl : config.setup) {
    uint32_t pos;
    switch (pl.location) {
      case LOCATION_HAND: pos = POS_FACEDOWN; break;
      case LOCATION_MZONE: pos = POS_FACEUP_ATTACK; break;
      case LOCATION_SZONE: pos = POS_FACEDOWN; break;
      case LOCATION_GRAVE: case LOCATION_REMOVED: pos = POS_FACEUP; break;
      default: throw std::invalid_argument("cannot place a card in location " + std::to_string(pl.location));
    }
    if (pl.position) pos = pl.position;
    OCG_NewCardInfo info{pl.team, 0, pl.code, pl.team, pl.location, pl.sequence, pos};
    OCG_DuelNewCard(handle_, &info);
  }
  OCG_StartDuel(handle_);
}

Duel::~Duel() {
  if (handle_) OCG_DestroyDuel(handle_);
}

void Duel::add_cards(uint8_t team, uint32_t location, const std::vector<uint32_t>& codes) {
  OCG_NewCardInfo info{};
  info.team = info.con = team;
  info.duelist = 0;
  info.loc = location;
  info.pos = POS_FACEDOWN_DEFENSE;
  for (uint32_t seq = 0; seq < codes.size(); ++seq) {
    info.code = codes[seq];
    info.seq = seq;
    OCG_DuelNewCard(handle_, &info);
  }
}

static bool is_select(uint8_t t) {
  switch (t) {
    case MSG_SELECT_BATTLECMD: case MSG_SELECT_IDLECMD: case MSG_SELECT_EFFECTYN: case MSG_SELECT_YESNO:
    case MSG_SELECT_OPTION: case MSG_SELECT_CARD: case MSG_SELECT_CHAIN: case MSG_SELECT_PLACE:
    case MSG_SELECT_POSITION: case MSG_SELECT_TRIBUTE: case MSG_SORT_CHAIN: case MSG_SELECT_COUNTER:
    case MSG_SELECT_SUM: case MSG_SELECT_DISFIELD: case MSG_SORT_CARD: case MSG_SELECT_UNSELECT_CARD:
    case MSG_ROCK_PAPER_SCISSORS: case MSG_ANNOUNCE_RACE: case MSG_ANNOUNCE_ATTRIB: case MSG_ANNOUNCE_CARD:
    case MSG_ANNOUNCE_NUMBER:
      return true;
    default:
      return false;
  }
}

const Bytes* Duel::advance() {
  // A few card interactions make the core spin without ever asking for input; fail instead of hanging the worker.
  uint64_t spins = 0;
  while (!has_pending_ && !ended) {
    if (++spins > 2000000) throw std::runtime_error("core made no progress (no selection after 2M process calls)");
    int status = OCG_DuelProcess(handle_);
    uint32_t length = 0;
    auto* buf = static_cast<const uint8_t*>(OCG_DuelGetMessage(handle_, &length));
    uint32_t pos = 0;
    while (buf && pos + 4 <= length) {
      uint32_t size;
      std::memcpy(&size, buf + pos, 4);
      pos += 4;
      if (pos + size > length) throw std::runtime_error("truncated message buffer");
      const uint8_t* m = buf + pos;
      pos += size;
      if (!size) continue;
      uint8_t t = m[0];
      if (t == MSG_RETRY) throw std::runtime_error("core rejected response at step " + std::to_string(responses.size()));
      if (t == MSG_NEW_TURN) {
        ++turn;
        turn_player = m[1] & 1;
        if (first_player < 0) first_player = turn_player;
      } else if (t == MSG_NEW_PHASE) {
        uint16_t ph;
        std::memcpy(&ph, m + 1, 2);
        phase = ph;
      } else if (t == MSG_WIN) {
        winner = m[1];
        win_reason = m[2];
        ended = true;
        break;
      } else if (is_select(t)) {
        pending_.assign(m, m + size);
        has_pending_ = true;
      }
    }
    if (status == OCG_DUEL_STATUS_END) ended = true;
    else if (status == OCG_DUEL_STATUS_AWAITING && !has_pending_ && !ended)
      throw std::runtime_error("core is awaiting a response but sent no select message");
  }
  return has_pending_ ? &pending_ : nullptr;
}

void Duel::respond(const Bytes& response) {
  if (!has_pending_) throw std::runtime_error("no pending selection");
  OCG_DuelSetResponse(handle_, response.data(), static_cast<uint32_t>(response.size()));
  responses.push_back(response);
  has_pending_ = false;
}

// ---------------------------------------------------------------- queries

constexpr uint32_t CARD_FLAGS = QUERY_CODE | QUERY_POSITION | QUERY_TYPE | QUERY_LEVEL | QUERY_RANK |
                                QUERY_ATTRIBUTE | QUERY_RACE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_BASE_ATTACK |
                                QUERY_BASE_DEFENSE | QUERY_OVERLAY_CARD | QUERY_OWNER | QUERY_STATUS |
                                QUERY_IS_PUBLIC | QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;

template <typename T>
static T rd(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

static size_t parse_card(const uint8_t* buf, size_t pos, CardState& c) {
  while (true) {
    auto size = rd<uint16_t>(buf + pos);
    auto flag = rd<uint32_t>(buf + pos + 2);
    const uint8_t* body = buf + pos + 6;
    pos += 2 + size;
    switch (flag) {
      case QUERY_END: return pos;
      case QUERY_RACE: c.race = rd<uint64_t>(body); break;
      case QUERY_OWNER: c.owner = body[0]; break;
      case QUERY_IS_PUBLIC: c.is_public = body[0] != 0; break;
      case QUERY_LINK: c.link = rd<uint32_t>(body); c.link_marker = rd<uint32_t>(body + 4); break;
      case QUERY_OVERLAY_CARD: {
        auto n = rd<uint32_t>(body);
        c.overlay.resize(n);
        for (uint32_t i = 0; i < n; ++i) c.overlay[i] = rd<uint32_t>(body + 4 + 4 * i);
        break;
      }
      case QUERY_CODE: c.code = rd<uint32_t>(body); break;
      case QUERY_POSITION: c.position = rd<uint32_t>(body); break;
      case QUERY_TYPE: c.type = rd<uint32_t>(body); break;
      case QUERY_LEVEL: c.level = rd<uint32_t>(body); break;
      case QUERY_RANK: c.rank = rd<uint32_t>(body); break;
      case QUERY_ATTRIBUTE: c.attribute = rd<uint32_t>(body); break;
      case QUERY_ATTACK: c.attack = rd<int32_t>(body); break;
      case QUERY_DEFENSE: c.defense = rd<int32_t>(body); break;
      case QUERY_BASE_ATTACK: c.base_attack = rd<int32_t>(body); break;
      case QUERY_BASE_DEFENSE: c.base_defense = rd<int32_t>(body); break;
      case QUERY_STATUS: c.status = rd<uint32_t>(body); break;
      case QUERY_LSCALE: c.lscale = rd<uint32_t>(body); break;
      case QUERY_RSCALE: c.rscale = rd<uint32_t>(body); break;
      default: break;
    }
  }
}

std::vector<std::optional<CardState>> Duel::query_location(uint8_t controller, uint32_t location) {
  OCG_QueryInfo info{CARD_FLAGS, controller, location, 0, 0};
  uint32_t length = 0;
  auto* buf = static_cast<const uint8_t*>(OCG_DuelQueryLocation(handle_, &length, &info));
  std::vector<std::optional<CardState>> out;
  size_t pos = 4;
  uint32_t seq = 0;
  while (pos < length) {
    if (rd<uint16_t>(buf + pos) == 0) {
      out.emplace_back();
      pos += 2;
    } else {
      CardState c;
      c.controller = controller;
      c.location = location;
      c.sequence = seq;
      pos = parse_card(buf, pos, c);
      out.emplace_back(std::move(c));
    }
    ++seq;
  }
  return out;
}

Bytes Duel::query_field() {
  uint32_t length = 0;
  auto* buf = static_cast<const uint8_t*>(OCG_DuelQueryField(handle_, &length));
  return Bytes(buf, buf + length);
}

}  // namespace ygo
