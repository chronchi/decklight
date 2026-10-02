// C++ port of the Python reference env (src/ygo_zero/engine/{duel,query,actions,obs,env}.py).
// The contract test (tests/engine/test_cpp_env.py) requires identical observations, step for step.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ocgapi.h"
#include "ocgapi_constants.h"

namespace ygo {

using Bytes = std::vector<uint8_t>;

constexpr uint32_t CARD_MARINE_DOLPHIN = 78734254;
constexpr uint32_t CARD_TWINKLE_MOSS = 13857930;

// ---------------------------------------------------------------- engine / duel

struct CardData {
  uint32_t code = 0, alias = 0;
  std::vector<uint16_t> setcodes;  // 0-terminated for the core
  uint32_t type = 0, level = 0, attribute = 0;
  uint64_t race = 0;
  int32_t attack = 0, defense = 0;
  uint32_t lscale = 0, rscale = 0, link_marker = 0;
};

class Engine {
 public:
  std::unordered_map<uint32_t, CardData> cards;
  std::vector<uint32_t> sorted_codes;  // for the ANNOUNCE_CARD fallback
  std::unordered_map<std::string, std::string> script_paths;

  void add_card(CardData cd);
  void finalize();  // call once after all add_card
  // Script source, loaded lazily and cached; nullptr if unknown. Thread-safe.
  const std::string* script(const std::string& name);

 private:
  std::shared_mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<std::string>> cache_;
};

struct DuelConfig {
  std::vector<uint32_t> main0, extra0, main1, extra1;
  uint64_t seed = 0;
  uint64_t flags = DUEL_MODE_MR5 | DUEL_TCG_SEGOC_NONPUBLIC;
  uint32_t lp = 8000, hand = 5, draw = 1;
};

struct CardState {
  uint32_t controller = 0, location = 0, sequence = 0;
  uint32_t code = 0, position = 0, type = 0, level = 0, rank = 0, attribute = 0;
  uint64_t race = 0;
  int32_t attack = 0, defense = 0, base_attack = 0, base_defense = 0;
  uint32_t owner = 0, status = 0;
  bool is_public = false;
  uint32_t lscale = 0, rscale = 0, link = 0, link_marker = 0;
  std::vector<uint32_t> overlay;
  uint32_t overlay_index = 0;
};

struct ChainLink {
  uint32_t code, controller, location, sequence, position;
  uint32_t trig_controller, trig_location, trig_sequence;
  uint64_t description;
};

class Duel {
 public:
  Duel(Engine& engine, const DuelConfig& config);
  ~Duel();
  Duel(const Duel&) = delete;
  Duel& operator=(const Duel&) = delete;

  // Run until a response is needed (returns the raw select message) or the duel ends (nullptr).
  const Bytes* advance();
  void respond(const Bytes& response);

  std::vector<std::optional<CardState>> query_location(uint8_t controller, uint32_t location);
  Bytes query_field();

  Engine& engine;
  DuelConfig config;
  std::vector<Bytes> responses;
  std::vector<std::string> errors;
  bool ended = false;
  int winner = -1, win_reason = 0;
  int turn = 0, turn_player = 0, first_player = -1, phase = 0;

 private:
  OCG_Duel handle_ = nullptr;
  Bytes pending_;
  bool has_pending_ = false;
  void add_cards(uint8_t team, uint32_t location, const std::vector<uint32_t>& codes);
  bool load_script(const std::string& name);
  static int read_script(void* payload, OCG_Duel duel, const char* name);
  static void read_card(void* payload, uint32_t code, OCG_CardData* data);
  static void log(void* payload, const char* str, int type);
};

// ---------------------------------------------------------------- decisions

enum Kind : int32_t {
  SUMMON = 1, SPSUMMON, REPOS, MSET, SSET, ACTIVATE, ATTACK, TO_BATTLE, TO_MAIN2, TO_END, PASS, YES, NO,
  OPTION, SELECT, UNSELECT, FINISH, CANCEL, PLACE, POSITION, NUMBER, ATTRIBUTE, RACE, DECLARE, SORT,
  COUNTER, RPS,
};

struct CardRef {
  uint32_t code = 0, controller = 0, location = 0, sequence = 0, position = 0;
  bool operator==(const CardRef&) const = default;
};

struct Option {
  Kind kind;
  bool has_card = false;
  CardRef card;
  uint32_t code = 0;
  uint64_t desc = 0;
  int64_t arg = 0;
};

class Decision {
 public:
  explicit Decision(int msg_type, int player) : msg_type(msg_type), player(player) {}
  virtual ~Decision() = default;
  virtual std::vector<Option> options() = 0;
  // Returns true and fills `response` once the selection is complete.
  bool choose(int index, Bytes& response);

  int msg_type, player;
  int step = 0;
  std::vector<CardRef> selected;

 protected:
  virtual bool apply(const Option& opt, int index, Bytes& response) = 0;
};

using CodePool = std::function<std::set<uint32_t>()>;
std::unique_ptr<Decision> make_decision(const Bytes& msg, Engine& engine, const CodePool& pool);
bool is_declarable(const CardData& cd, const std::vector<uint64_t>& opcodes);

// ---------------------------------------------------------------- observation

constexpr int OBS_VERSION = 1;
constexpr int MAX_CARDS = 160, CARD_COLS = 21;
constexpr int MAX_CHAIN = 16, CHAIN_COLS = 5;
constexpr int MAX_OPTIONS = 64, ACTION_COLS = 10;
constexpr int HISTORY = 32, HISTORY_COLS = 13;
constexpr int GLOBAL_COLS = 24;

// One observation; the bindings expose these as numpy views.
struct Obs {
  std::array<int32_t, MAX_CARDS * CARD_COLS> cards{};
  std::array<uint8_t, MAX_CARDS> card_mask{};
  std::array<int32_t, GLOBAL_COLS> global{};
  std::array<int32_t, MAX_CHAIN * CHAIN_COLS> chain{};
  std::array<uint8_t, MAX_CHAIN> chain_mask{};
  std::array<int32_t, MAX_OPTIONS * ACTION_COLS> actions{};
  std::array<uint8_t, MAX_OPTIONS> action_mask{};
  std::array<int32_t, HISTORY * HISTORY_COLS> history{};
  std::array<uint8_t, HISTORY> history_mask{};
  void clear() { *this = Obs{}; }
};

struct Row {
  CardState card;
  bool visible;
};

struct HistoryEntry {
  std::array<int32_t, ACTION_COLS> row;
  int actor, turn, phase;
  bool is_public;
};

class Board {
 public:
  Board(Duel& duel, int me);
  int row_of(const CardRef* ref) const;
  uint32_t option_code(const Option& opt) const;
  std::set<uint32_t> visible_codes() const;
  std::array<int32_t, ACTION_COLS> option_row(const Option& opt, int msg_type, int step) const;
  void fill(Obs& obs, const Duel& duel, int msg_type, const std::vector<CardRef>& selected) const;
  static bool is_public(const CardState& c);

  int me;
  std::vector<Row> rows;
  int truncated = 0;

 private:
  std::unordered_map<uint64_t, int> index_;
  std::unordered_map<uint64_t, int> counts_;
  uint32_t lp_[2];
  std::vector<ChainLink> chain_;
};

HistoryEntry history_entry(const Board& board, const Option& opt, int msg_type, int step, int turn, int phase);

// ---------------------------------------------------------------- env

class Env {
 public:
  Env(Engine& engine, DuelConfig config, bool auto_single = true);
  // Both return false once the duel is over (obs untouched then).
  bool reset(Obs& obs);
  bool step(int index, Obs& obs);

  bool done() const { return duel_->ended; }
  int winner() const { return duel_->winner; }
  int player() const { return dec_->player; }
  int num_options() const { return static_cast<int>(opts_.size()); }
  const Duel& duel() const { return *duel_; }

  std::vector<int> choices;
  int decisions = 0, truncated = 0;

 private:
  Engine& engine_;
  DuelConfig config_;
  bool auto_single_;
  std::unique_ptr<Duel> duel_;
  std::unique_ptr<Decision> dec_;
  std::unique_ptr<Board> board_;
  std::vector<Option> opts_;
  std::vector<HistoryEntry> history_;

  void next_decision();
  std::set<uint32_t> known_codes(int p);
  void take(int index, bool record);
  bool settle(Obs& obs);
};

}  // namespace ygo
