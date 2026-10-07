// C++ port of the Python reference env (src/ygo_zero/engine/{duel,query,actions,obs,env}.py).
// The contract test (tests/engine/test_cpp_env.py) requires identical observations, step for step.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
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
  bool precompile = true;  // cache scripts as Lua bytecode

  void add_card(CardData cd);
  void finalize();  // call once after all add_card
  // Script source, loaded lazily and cached; nullptr if unknown. Thread-safe.
  const std::string* script(const std::string& name);

 private:
  std::shared_mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<std::string>> cache_;
};

struct Placement {  // puzzle start, see duel.py Placement
  uint8_t team;
  uint32_t code, location, sequence, position;
};

struct DuelConfig {
  std::vector<uint32_t> main0, extra0, main1, extra1;
  uint64_t seed = 0;
  uint64_t flags = DUEL_MODE_MR5 | DUEL_TCG_SEGOC_NONPUBLIC;
  uint32_t lp = 8000, hand = 5, draw = 1;
  std::vector<Placement> setup;
};

struct CardState {
  uint32_t controller = 0, location = 0, sequence = 0;
  uint32_t code = 0, position = 0, type = 0, level = 0, rank = 0, attribute = 0;
  uint64_t race = 0;
  int32_t attack = 0, defense = 0, base_attack = 0, base_defense = 0;
  uint32_t owner = 0, status = 0, cardid = 0;
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

// One card anywhere on the board, for honest-search determinization: its identity (code),
// its creation id (cardid, deterministic given the creation order) and whether the opponent
// knows a HAND card's identity. Queried with CODE | POSITION | CARDID | IS_PUBLIC only.
struct DebugCard {
  uint32_t controller = 0, location = 0, sequence = 0;
  uint32_t position = 0, code = 0, cardid = 0;
  bool is_public = false;
  uint8_t known = 0;  // hand_known_: 1 = the other player knows this HAND card's identity
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
  std::vector<DebugCard> cards_debug();  // every card of both players (DECK/HAND/MZONE/SZONE/GRAVE/REMOVED/EXTRA)

  Engine& engine;
  DuelConfig config;
   std::vector<Bytes> responses;
   std::vector<std::string> errors;
   bool ended = false;
   int winner = -1, win_reason = 0;
   int turn = 0, turn_player = 0, first_player = -1, phase = 0;
   uint64_t select_hint = 0;  // HINT_SELECTMSG sent just before the pending select message (0 = none)
   // Card-knowledge memory (docs/obs_spec.md v2 annex): per hand seq, whether the other player
   // knows the card's identity (+ the true codes, for the shuffle realignment). Lockstep with
   // HandMemory in src/ygo_zero/engine/knowledge.py.
   std::vector<uint8_t> hand_known_[2];
   std::vector<uint32_t> hand_codes_[2];
   int knowledge_resyncs_ = 0;
   int knowledge_resyncs() const { return knowledge_resyncs_; }

  private:
   OCG_Duel handle_ = nullptr;
   uint64_t hint_ = 0;
   Bytes pending_;
   bool has_pending_ = false;
   void add_cards(uint8_t team, uint32_t location, const std::vector<uint32_t>& codes);
   void track_message(const uint8_t* m, uint32_t size);  // fold one raw message into hand_known_
   void guard_knowledge();  // resync guard: lengths vs queried hand sizes at every decision
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
constexpr int HISTORY_MAX = 512;  // long salient window (obs v2); v1 HISTORY stays the reference
constexpr int HLOG_COLS = 14;     // history_log: stored option row (10) + actor, turn, phase, public
constexpr int GLOBAL_COLS = 24;

// Option kinds with no information for future decisions (obs.SALIENT_SKIP in Python).
inline bool is_salient(Kind k) { return k != PASS && k != PLACE; }

// One observation; the bindings expose these as numpy views.
struct Obs {
  std::array<int32_t, MAX_CARDS * CARD_COLS> cards{};
  std::array<uint8_t, MAX_CARDS> card_mask{};
  std::array<uint8_t, MAX_CARDS> card_known{};  // obs v2: 1 = remembered opp hidden card, 2 = own known to opp
  std::array<int32_t, MAX_CARDS> known_code{};  // obs v2: remembered passcode for card_known == 1 rows
  std::array<int32_t, MAX_CARDS> priv_code{};  // privileged critic (RL v2 part B): true code of every
                                               // row hidden from the viewer (!visible, code column 0), else 0
  std::array<int32_t, GLOBAL_COLS> global{};
  std::array<int32_t, MAX_CHAIN * CHAIN_COLS> chain{};
  std::array<uint8_t, MAX_CHAIN> chain_mask{};
  std::array<int32_t, MAX_OPTIONS * ACTION_COLS> actions{};
  std::array<uint8_t, MAX_OPTIONS> action_mask{};
  std::array<int32_t, HISTORY * HISTORY_COLS> history{};
  std::array<uint8_t, HISTORY> history_mask{};
  std::array<int32_t, HISTORY_MAX * HISTORY_COLS> hist_long{};
  std::array<uint8_t, HISTORY_MAX> hist_long_mask{};
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
// One recorded entry as seen by `me` (shared by the v1 and the long salient window).
void write_history_row(int32_t* o, const HistoryEntry& e, int me, int turn);

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
    int knowledge_resyncs() const { return duel_->knowledge_resyncs(); }
    int history_len() const { return static_cast<int>(history_.size()); }
    std::vector<DebugCard> cards() { return duel_->cards_debug(); }
   // All recorded entries in absolute form: stored row (10) + actor, turn, phase, public.
   std::vector<std::array<int32_t, HLOG_COLS>> history_log() const;

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

// ---------------------------------------------------------------- batched env

namespace ygo {

struct Deck {
  std::vector<uint32_t> main, extra;
};

// A finished game, enough to replay it (Env with the same config + choices).
struct Episode {
  int deck0, deck1;
  uint64_t seed;
  int64_t index = -1;  // position in the schedule (scheduled mode), else -1
  int winner, turns;
  std::vector<int> choices;
};

// N envs stepped in parallel by a persistent thread pool. Finished envs restart at once with a fresh
// (deck0, deck1, seed) drawn from a per-env splitmix64 stream, so the whole run is reproducible from `seed`.
class VecEnv {
 public:
  VecEnv(Engine& engine, std::vector<Deck> decks, int num_envs, int num_threads, uint64_t seed, bool auto_single);
  ~VecEnv();
  void reset();
  void step(const int32_t* actions);  // actions[num_envs]
  // Async mode (envpool send/recv): envs step independently; a slow env never blocks the others.
  void async_reset();  // queue an episode start for every env (like reset(), but returns at once)
  void send(const int32_t* ids, const int32_t* actions, int n);  // queue env ids[k] with actions[k]
  std::vector<int32_t> recv(int min_ready, int max_ready);
  std::vector<Episode> pop_finished();
  // Scheduled mode (evaluation): play exactly these (deck0, deck1, seed) games, in order, then go idle.
  // Idle envs have player = -1 and an empty action mask, and ignore their action. Call before reset().
  void set_schedule(std::vector<Episode> games);
  int64_t remaining() const;  // scheduled games not finished yet
  // Cut an episode after n decisions (0 = never): it finishes with winner TRUNCATED_WINNER (RL rollouts).
  void set_max_decisions(int n);
  static constexpr int TRUNCATED_WINNER = 3;

  const int num_envs;
  std::vector<Obs> obs;          // obs[i]: the decision now pending in env i
  std::vector<int32_t> player;   // who acts in obs[i]
  std::vector<int32_t> winner;   // after step: winner of the episode that just ended in env i, else -1
  std::vector<uint8_t> done;     // after step: env i finished an episode (obs[i] is the next one's start)
  std::vector<int64_t> game;     // schedule index of the game now in env i (-1: idle or unscheduled)

 private:
  Engine& engine_;
  std::vector<Deck> decks_;
  bool auto_single_;
  std::vector<std::unique_ptr<Env>> envs_;
  std::vector<uint64_t> rng_;
  std::vector<Episode> current_, finished_;
  std::vector<Episode> schedule_;
  bool scheduled_ = false;
  size_t next_game_ = 0;
  int64_t unfinished_ = 0;
  std::vector<uint8_t> idle_;
  std::vector<std::exception_ptr> errors_;
  const int32_t* actions_ = nullptr;
  bool resetting_ = false;
  int max_decisions_ = 0;

  // pool
  std::vector<std::thread> workers_;
  mutable std::mutex mu_;
  std::condition_variable cv_work_, cv_done_, cv_ready_;
  uint64_t generation_ = 0;
  int pending_ = 0;
  std::atomic<int> next_{0};
  bool stop_ = false;
  // async mode: per-env HELD (caller owns obs[i]) / QUEUED / RUNNING / READY, FIFO queues
  enum class Slot : uint8_t { HELD, QUEUED, RUNNING, READY };
  std::vector<Slot> slot_;
  std::deque<int> queue_;  // QUEUED ids waiting for a worker
  std::deque<int> ready_;  // READY ids in completion order, not yet returned by recv
  std::vector<int32_t> pending_action_;  // per-env action copy for queued work
  std::vector<char> pending_reset_;      // per-env reset flag for queued work
  void run_all();
  void worker();
  void work(int i, int32_t action, bool reset_flag);
  void start_episode(int i);
  void require_all_held() const;
};

}  // namespace ygo
