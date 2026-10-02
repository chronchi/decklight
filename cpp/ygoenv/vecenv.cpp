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

VecEnv::VecEnv(Engine& engine, std::vector<Deck> decks, int num_envs, int num_threads, uint64_t seed, bool auto_single)
    : num_envs(num_envs), obs(num_envs), player(num_envs, 0), winner(num_envs, -1), done(num_envs, 0),
      engine_(engine), decks_(std::move(decks)), auto_single_(auto_single), envs_(num_envs), rng_(num_envs),
      current_(num_envs), errors_(num_envs) {
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
  for (auto& w : workers_) w.join();
}

void VecEnv::start_episode(int i) {
  Episode& ep = current_[i];
  int n = static_cast<int>(decks_.size());
  ep.deck0 = static_cast<int>(splitmix(rng_[i]) % n);
  ep.deck1 = static_cast<int>(splitmix(rng_[i]) % n);
  ep.seed = splitmix(rng_[i]);
  const Deck &a = decks_[ep.deck0], &b = decks_[ep.deck1];
  DuelConfig cfg;
  cfg.main0 = a.main;
  cfg.extra0 = a.extra;
  cfg.main1 = b.main;
  cfg.extra1 = b.extra;
  cfg.seed = ep.seed;
  envs_[i] = std::make_unique<Env>(engine_, std::move(cfg), auto_single_);
  if (!envs_[i]->reset(obs[i])) throw std::runtime_error("duel ended before the first decision");
}

void VecEnv::work(int i) {
  try {
    done[i] = 0;
    winner[i] = -1;
    if (resetting_) {
      start_episode(i);
    } else if (!envs_[i]->step(actions_[i], obs[i])) {
      Env& env = *envs_[i];
      Episode& ep = current_[i];
      ep.winner = env.winner();
      ep.turns = env.duel().turn;
      ep.choices = env.choices;
      done[i] = 1;
      winner[i] = ep.winner;
      {
        std::lock_guard lock(mu_);
        finished_.push_back(ep);
      }
      start_episode(i);
    }
    player[i] = envs_[i]->player();
  } catch (...) {
    errors_[i] = std::current_exception();
  }
}

void VecEnv::worker() {
  uint64_t seen = 0;
  while (true) {
    {
      std::unique_lock lock(mu_);
      cv_work_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    int i;
    while ((i = next_.fetch_add(1)) < num_envs) work(i);
    std::lock_guard lock(mu_);
    if (--pending_ == 0) cv_done_.notify_one();
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
  resetting_ = true;
  run_all();
  resetting_ = false;
}

void VecEnv::step(const int32_t* actions) {
  actions_ = actions;
  run_all();
}

std::vector<Episode> VecEnv::pop_finished() {
  std::lock_guard lock(mu_);
  return std::exchange(finished_, {});
}

}  // namespace ygo
