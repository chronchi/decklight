#include <algorithm>
#include <stdexcept>
#include <utility>

#include "ygoenv.h"

namespace ygo {

static uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Seeded main-deck shuffle (Fisher-Yates, splitmix64 stream), mirrored by Python DuelConfig.main_of: the core never
// shuffles at duel start (EDOPro's host does it before adding the cards), so without this every game of a deck had
// the same opening hand (the last 5 cards of the .ydk) and draws.
static std::vector<uint32_t> shuffled_main(std::vector<uint32_t> main, uint64_t seed, uint64_t team) {
  uint64_t st = seed ^ (0x9E3779B97F4A7C15ull * (team + 1));
  for (size_t k = main.size(); k > 1; --k) {
    size_t j = static_cast<size_t>(splitmix(st) % k);
    std::swap(main[k - 1], main[j]);
  }
  return main;
}

VecEnv::VecEnv(Engine& engine, std::vector<Deck> decks, int num_envs, int num_threads, uint64_t seed, bool auto_single)
    : num_envs(num_envs), obs(num_envs), player(num_envs, 0), winner(num_envs, -1), done(num_envs, 0), game(num_envs, -1),
      dags(num_envs, 0), dags_last(num_envs, 0), ga(num_envs, 0), deck0(num_envs, -1), deck1(num_envs, -1),
      deal_seed(num_envs, 0),
      start(num_envs, -1),
      engine_(engine), decks_(std::move(decks)), auto_single_(auto_single), envs_(num_envs), rng_(num_envs),
      dags_rng_(num_envs),
      current_(num_envs), idle_(num_envs, 0), errors_(num_envs), slot_(num_envs, Slot::HELD),
      pending_action_(num_envs, 0), pending_reset_(num_envs, 0) {
  if (decks_.empty() || num_envs <= 0) throw std::invalid_argument("need decks and num_envs > 0");
  for (int i = 0; i < num_envs; ++i) {
    uint64_t s = seed ^ (0xD1B54A32D192ED03ull * (i + 1));
    rng_[i] = splitmix(s);
  }
  for (int t = 0; t < std::max(1, num_threads); ++t) workers_.emplace_back(&VecEnv::worker, this);
}

VecEnv::~VecEnv() {
  {
    std::lock_guard lock(mu_);
    stop_ = true;
  }
  cv_work_.notify_all();
  cv_ready_.notify_all();
  for (auto& w : workers_) w.join();
}

void VecEnv::set_schedule(std::vector<Episode> games) {
  int n = static_cast<int>(decks_.size());
  for (auto& g : games)
    if (g.deck0 < 0 || g.deck0 >= n || g.deck1 < 0 || g.deck1 >= n) throw std::out_of_range("deck index");
  std::lock_guard lock(mu_);
  schedule_ = std::move(games);
  for (size_t k = 0; k < schedule_.size(); ++k) schedule_[k].index = static_cast<int64_t>(k);
  scheduled_ = true;
  next_game_ = 0;
  unfinished_ = static_cast<int64_t>(schedule_.size());
}

int64_t VecEnv::remaining() const {
  std::lock_guard lock(mu_);
  return unfinished_;
}

void VecEnv::set_max_decisions(int n) { max_decisions_ = n; }

void VecEnv::set_start_weights(std::vector<double> weights) {
  if (weights.size() != starts_.size())
    throw std::invalid_argument("start weights length must match the starts list");
  double total = 0.0;
  for (double w : weights) {
    if (!(w >= 0.0)) throw std::invalid_argument("start weights must be >= 0");
    total += w;
  }
  if (!(total > 0.0)) throw std::invalid_argument("start weights sum must be > 0");
  start_cum_.resize(weights.size());
  double run = 0.0;
  for (size_t k = 0; k < weights.size(); ++k) {
    run += weights[k];
    start_cum_[k] = run;
  }
  start_total_ = total;
}

void VecEnv::set_ga_share(double p, int start, int count, bool mixed) {
  if (!(p >= 0.0 && p <= 1.0)) throw std::invalid_argument("ga share must be in [0, 1]");
  if (start < 0 || count < 0 || start + count > static_cast<int>(decks_.size()))
    throw std::out_of_range("gauntlet slice out of deck range");
  if (p > 0.0 && count == 0) throw std::invalid_argument("ga share > 0 needs a non-empty gauntlet slice");
  ga_p_ = p;
  ga_start_ = start;
  ga_count_ = count;
  ga_mixed_ = mixed;
}

void VecEnv::set_pool_count(int n) {
  if (n <= 0 || n > static_cast<int>(decks_.size()))
    throw std::out_of_range("pool count out of deck range");
  pool_count_ = n;
}

void VecEnv::set_starts(std::vector<DagsStart> starts, double p, uint64_t seed) {
  if (!(p >= 0.0 && p <= 1.0)) throw std::invalid_argument("dags p must be in [0, 1]");
  starts_ = std::move(starts);
  dags_p_ = p;
  start_cum_.clear();  // a new starts list drops the old curriculum weights
  start_total_ = 0.0;
  for (int i = 0; i < num_envs; ++i) {
    uint64_t s = seed ^ (0xC2B2AE3D27D4EB4Full * (i + 1));
    dags_rng_[i] = splitmix(s);
  }
}

// DAGS mid-game start (non-scheduled mode only): with probability dags_p_ pick a start uniformly
// (or proportionally to the curriculum weights set by set_start_weights), build the Env from its
// final deck orders, reset, then step through the prefix. The episode then
// begins at the first live decision. A duel that ends, or a choice out of range, during the prefix
// discards the start (counts dags_bad) and the caller falls back to a normal start. With p = 0
// (or no starts) this consumes no random numbers: behaviour stays bit-identical.
bool VecEnv::try_dags(int i, Episode& ep) {
  if (dags_p_ <= 0.0 || starts_.empty()) return false;
  double u = static_cast<double>(splitmix(dags_rng_[i]) >> 11) * (1.0 / 9007199254740992.0);
  if (u >= dags_p_) return false;
  size_t idx;
  if (!start_cum_.empty() && start_cum_.size() == starts_.size()) {
    double w = static_cast<double>(splitmix(dags_rng_[i]) >> 11) * (1.0 / 9007199254740992.0) * start_total_;
    idx = static_cast<size_t>(std::upper_bound(start_cum_.begin(), start_cum_.end(), w) - start_cum_.begin());
    if (idx >= starts_.size()) idx = starts_.size() - 1;
  } else {
    idx = splitmix(dags_rng_[i]) % starts_.size();
  }
  const DagsStart& s = starts_[idx];
  DuelConfig cfg;
  cfg.main0 = s.main0;
  cfg.extra0 = s.extra0;
  cfg.main1 = s.main1;
  cfg.extra1 = s.extra1;
  cfg.seed = s.seed;
  cfg.lp = s.lp;
  cfg.hand = s.hand;
  cfg.draw = s.draw;
  cfg.setup = s.setup;
  auto env = std::make_unique<Env>(engine_, std::move(cfg), auto_single_);
  bool bad = !env->reset(obs[i]);
  for (int c : s.prefix) {
    if (bad) break;
    if (c < 0 || c >= env->num_options()) {
      bad = true;
      break;
    }
    if (!env->step(c, obs[i])) bad = true;  // the duel ended during the prefix: no live decision
  }
  if (bad) {
    ++dags_bad_;
    envs_[i].reset();
    obs[i].clear();
    return false;
  }
  ep.deck0 = ep.deck1 = -1;
  ep.seed = s.seed;
  ep.index = -1;
  ep.dags = 1;
  envs_[i] = std::move(env);
  dags[i] = 1;
  ga[i] = 0;
  deck0[i] = deck1[i] = -1;
  deal_seed[i] = s.seed;
  start[i] = static_cast<int32_t>(idx);
  ++dags_used_;
  return true;
}

void VecEnv::start_episode(int i) {
  Episode& ep = current_[i];
  if (scheduled_) {
    {
      std::lock_guard lock(mu_);
      if (next_game_ >= schedule_.size()) {
        idle_[i] = 1;
        game[i] = -1;
        ga[i] = 0;
        deck0[i] = deck1[i] = -1;
        deal_seed[i] = 0;
        start[i] = -1;
        envs_[i].reset();
        obs[i].clear();
        return;
      }
      ep = schedule_[next_game_++];
    }
    game[i] = ep.index;
    ga[i] = 0;
    deck0[i] = static_cast<int32_t>(ep.deck0);
    deck1[i] = static_cast<int32_t>(ep.deck1);
    deal_seed[i] = ep.seed;
    start[i] = -1;
  } else if (try_dags(i, ep)) {
    return;
  } else if (ga_p_ > 0.0 && ga_count_ > 0 &&
             static_cast<double>(splitmix(rng_[i]) >> 11) * (1.0 / 9007199254740992.0) < ga_p_) {
    int pool_n = pool_count_ > 0 ? pool_count_ : static_cast<int>(decks_.size());
    if (!ga_mixed_) {
      // Gauntlet-vs-gauntlet episode (deck-aware league): both decks from the gauntlet slice.
      ep.deck0 = ga_start_ + static_cast<int>(splitmix(rng_[i]) % ga_count_);
      ep.deck1 = ga_start_ + static_cast<int>(splitmix(rng_[i]) % ga_count_);
      ep.seed = splitmix(rng_[i]);
      ep.dags = 0;
      ga[i] = 1;
      deck0[i] = static_cast<int32_t>(ep.deck0);
      deck1[i] = static_cast<int32_t>(ep.deck1);
      deal_seed[i] = ep.seed;
      start[i] = -1;
    } else {
      // Mixed episode (overfitting rule): one seat train-pool, one seat gauntlet.
      int ga_deck = ga_start_ + static_cast<int>(splitmix(rng_[i]) % ga_count_);
      int pool_deck = static_cast<int>(splitmix(rng_[i]) % pool_n);
      bool ga_on_0 = (splitmix(rng_[i]) % 2) == 0;
      ep.deck0 = ga_on_0 ? ga_deck : pool_deck;
      ep.deck1 = ga_on_0 ? pool_deck : ga_deck;
      ep.seed = splitmix(rng_[i]);
      ep.dags = 0;
      ga[i] = ga_on_0 ? 1 : 2;
      deck0[i] = static_cast<int32_t>(ep.deck0);
      deck1[i] = static_cast<int32_t>(ep.deck1);
      deal_seed[i] = ep.seed;
      start[i] = -1;
    }
  } else {
    // ga_p_ == 0 consumes no random numbers here: bit-identical to no gauntlet pool.
    int pool_n = pool_count_ > 0 ? pool_count_ : static_cast<int>(decks_.size());
    ep.deck0 = static_cast<int>(splitmix(rng_[i]) % pool_n);
    ep.deck1 = static_cast<int>(splitmix(rng_[i]) % pool_n);
    ep.seed = splitmix(rng_[i]);
    ep.dags = 0;
    ga[i] = 0;
    deck0[i] = static_cast<int32_t>(ep.deck0);
    deck1[i] = static_cast<int32_t>(ep.deck1);
    deal_seed[i] = ep.seed;
    start[i] = -1;
  }
  dags[i] = 0;
  const Deck &a = decks_[ep.deck0], &b = decks_[ep.deck1];
  DuelConfig cfg;
  cfg.main0 = shuffled_main(a.main, ep.seed, 0);  // same order as Python DuelConfig.main_of (shuffle=True)
  cfg.extra0 = a.extra;
  cfg.main1 = shuffled_main(b.main, ep.seed, 1);
  cfg.extra1 = b.extra;
  cfg.seed = ep.seed;
  envs_[i] = std::make_unique<Env>(engine_, std::move(cfg), auto_single_);
  if (!envs_[i]->reset(obs[i])) throw std::runtime_error("duel ended before the first decision");
}

void VecEnv::require_all_held() const {
  for (int i = 0; i < num_envs; ++i)
    if (slot_[i] != Slot::HELD) throw std::logic_error("VecEnv: async work outstanding (mixing sync and async)");
}

void VecEnv::work(int i, int32_t action, bool reset_flag) {
  try {
    done[i] = 0;
    winner[i] = -1;
    if (reset_flag) {
      idle_[i] = 0;
      start_episode(i);
    } else if (idle_[i]) {
      // scheduled games exhausted: ignore the action
    } else if (bool over = !envs_[i]->step(action, obs[i]);
               over || (max_decisions_ > 0 && envs_[i]->decisions >= max_decisions_)) {
      Env& env = *envs_[i];
      Episode& ep = current_[i];
      ep.winner = over ? env.winner() : TRUNCATED_WINNER;
      ep.turns = env.duel().turn;
      ep.choices = env.choices;
      done[i] = 1;
      winner[i] = ep.winner;
      dags_last[i] = static_cast<uint8_t>(ep.dags);
      {
        std::lock_guard lock(mu_);
        finished_.push_back(ep);
        --unfinished_;
      }
      start_episode(i);
    }
    player[i] = idle_[i] ? -1 : envs_[i]->player();
  } catch (...) {
    errors_[i] = std::current_exception();
  }
}

void VecEnv::worker() {
  uint64_t seen = 0;
  while (true) {
    int id = -1;
    int32_t act = 0;
    bool rst = false, is_async = false, is_sync = false;
    {
      std::unique_lock lock(mu_);
      cv_work_.wait(lock, [&] { return stop_ || generation_ != seen || !queue_.empty(); });
      if (stop_) return;
      if (!queue_.empty()) {
        id = queue_.front();
        queue_.pop_front();
        slot_[id] = Slot::RUNNING;
        act = pending_action_[id];
        rst = pending_reset_[id] != 0;
        is_async = true;
      } else if (generation_ != seen) {
        seen = generation_;
        is_sync = true;
      }
    }
    if (is_async) {
      work(id, act, rst);
      {
        std::lock_guard lock(mu_);
        slot_[id] = Slot::READY;
        ready_.push_back(id);
      }
      cv_ready_.notify_all();
    } else if (is_sync) {
      int i;
      bool rst = resetting_;
      const int32_t* acts = actions_;
      // NOTE: actions_ is null during reset(); do not touch it then.
      while ((i = next_.fetch_add(1)) < num_envs) work(i, rst ? 0 : acts[i], rst);
      std::lock_guard lock(mu_);
      if (--pending_ == 0) cv_done_.notify_one();
    }
  }
}

void VecEnv::run_all() {
  std::unique_lock lock(mu_);
  next_ = 0;
  pending_ = static_cast<int>(workers_.size());
  ++generation_;
  cv_work_.notify_all();
  cv_done_.wait(lock, [&] { return pending_ == 0; });
  lock.unlock();
  for (auto& e : errors_)
    if (e) {
      auto err = e;
      std::fill(errors_.begin(), errors_.end(), nullptr);
      std::rethrow_exception(err);
    }
}

void VecEnv::reset() {
  {
    std::lock_guard lock(mu_);
    require_all_held();
  }
  resetting_ = true;
  run_all();
  resetting_ = false;
}

void VecEnv::step(const int32_t* actions) {
  {
    std::lock_guard lock(mu_);
    require_all_held();
  }
  actions_ = actions;
  run_all();
}

void VecEnv::async_reset() {
  std::unique_lock lock(mu_);
  require_all_held();
  for (int i = 0; i < num_envs; ++i) {
    slot_[i] = Slot::QUEUED;
    pending_reset_[i] = 1;
    queue_.push_back(i);
  }
  lock.unlock();
  cv_work_.notify_all();
}

void VecEnv::send(const int32_t* ids, const int32_t* actions, int n) {
  std::unique_lock lock(mu_);
  for (int k = 0; k < n; ++k) {
    int id = ids[k];
    if (id < 0 || id >= num_envs) throw std::invalid_argument("VecEnv::send: id out of range");
    if (slot_[id] != Slot::HELD) throw std::invalid_argument("VecEnv::send: id not HELD");
    for (int j = 0; j < k; ++j)
      if (ids[j] == id) throw std::invalid_argument("VecEnv::send: duplicate id");
  }
  for (int k = 0; k < n; ++k) {
    int id = ids[k];
    slot_[id] = Slot::QUEUED;
    pending_action_[id] = actions[k];
    pending_reset_[id] = 0;
    queue_.push_back(id);
  }
  lock.unlock();
  // one wake per queued env: notify_all would wake every pool thread for each send
  if (n >= static_cast<int>(workers_.size())) cv_work_.notify_all();
  else
    for (int k = 0; k < n; ++k) cv_work_.notify_one();
}

std::vector<int32_t> VecEnv::recv(int min_ready, int max_ready) {
  std::unique_lock lock(mu_);
  int64_t outstanding = 0;
  for (auto s : slot_)
    if (s != Slot::HELD) ++outstanding;
  if (outstanding == 0 || max_ready <= 0) return {};
  int64_t target = std::min<int64_t>(min_ready, outstanding);
  if (target < 0) target = 0;
  cv_ready_.wait(lock, [&] { return stop_ || static_cast<int64_t>(ready_.size()) >= target; });
  if (stop_ && ready_.empty()) return {};
  size_t take = std::min<size_t>(static_cast<size_t>(max_ready), ready_.size());
  std::vector<int32_t> out;
  out.reserve(take);
  for (size_t k = 0; k < take; ++k) {
    int id = ready_.front();
    ready_.pop_front();
    slot_[id] = Slot::HELD;
    out.push_back(id);
  }
  std::exception_ptr first;
  for (int id : out)
    if (errors_[id]) {
      if (!first) first = errors_[id];
      errors_[id] = nullptr;
    }
  lock.unlock();
  if (first) std::rethrow_exception(first);
  return out;
}

std::vector<Episode> VecEnv::pop_finished() {
  std::lock_guard lock(mu_);
  return std::exchange(finished_, {});
}

}  // namespace ygo
