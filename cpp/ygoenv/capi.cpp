// C ABI over the C++ env; see capi.h.
#include "capi.h"

#include <cstddef>
#include <cstring>
#include <exception>
#include <set>
#include <string>
#include <vector>

#include "ygoenv.h"

using ygo::Obs;

struct ygo_engine {
  ygo::Engine engine;
};

struct ygo_env {
  ygo::Env env;
  ygo_env(ygo::Engine& e, ygo::DuelConfig c, bool auto_single) : env(e, std::move(c), auto_single) {}
};

struct ygo_decision {
  std::unique_ptr<ygo::Decision> dec;
};

namespace {

void copy_err(const char* msg, char* err, size_t len) {
  if (!err || len == 0) return;
  std::strncpy(err, msg, len - 1);
  err[len - 1] = '\0';
}

std::vector<uint32_t> vec(const uint32_t* p, size_t n) { return n ? std::vector<uint32_t>(p, p + n) : std::vector<uint32_t>{}; }

}  // namespace

extern "C" {

uint32_t ygo_obs_version(void) { return ygo::OBS_VERSION; }
size_t ygo_obs_size(void) { return sizeof(Obs); }

size_t ygo_obs_offset(uint32_t field) {
  switch (field) {
    case 0: return offsetof(Obs, cards);
    case 1: return offsetof(Obs, card_mask);
    case 2: return offsetof(Obs, card_known);
    case 3: return offsetof(Obs, known_code);
    case 4: return offsetof(Obs, priv_code);
    case 5: return offsetof(Obs, global);
    case 6: return offsetof(Obs, chain);
    case 7: return offsetof(Obs, chain_mask);
    case 8: return offsetof(Obs, actions);
    case 9: return offsetof(Obs, action_mask);
    case 10: return offsetof(Obs, history);
    case 11: return offsetof(Obs, history_mask);
    case 12: return offsetof(Obs, hist_long);
    case 13: return offsetof(Obs, hist_long_mask);
    case 14: return offsetof(Obs, obs_history);
    case 15: return offsetof(Obs, obs_history_mask);
    default: return SIZE_MAX;
  }
}

ygo_engine* ygo_engine_new(void) { return new ygo_engine(); }
void ygo_engine_free(ygo_engine* e) { delete e; }
void ygo_engine_set_precompile(ygo_engine* e, int on) { e->engine.precompile = on != 0; }

void ygo_engine_add_card(ygo_engine* e, uint32_t code, uint32_t alias, const uint16_t* setcodes, size_t n_setcodes,
                         uint32_t type, uint32_t level, uint32_t attribute, uint64_t race, int32_t attack,
                         int32_t defense, uint32_t lscale, uint32_t rscale, uint32_t link_marker) {
  ygo::CardData cd;
  cd.code = code;
  cd.alias = alias;
  if (n_setcodes) cd.setcodes.assign(setcodes, setcodes + n_setcodes);
  cd.type = type;
  cd.level = level;
  cd.attribute = attribute;
  cd.race = race;
  cd.attack = attack;
  cd.defense = defense;
  cd.lscale = lscale;
  cd.rscale = rscale;
  cd.link_marker = link_marker;
  e->engine.add_card(std::move(cd));
}

void ygo_engine_set_script(ygo_engine* e, const char* name, const char* path) {
  e->engine.script_ranges.erase(name);
  e->engine.script_paths[name] = path;
}
void ygo_engine_set_script_range(ygo_engine* e, const char* name, const char* path, uint64_t offset, uint32_t len) {
  e->engine.script_paths.erase(name);
  e->engine.script_ranges[name] = ygo::ScriptRange{path, offset, len};
}
size_t ygo_engine_script(ygo_engine* e, const char* name, uint8_t* buf, size_t cap) {
  if (!e || !name) return 0;
  const std::string* s = e->engine.script(name);
  if (!s) return 0;
  if (buf && cap >= s->size()) std::memcpy(buf, s->data(), s->size());
  return s->size();
}
void ygo_engine_finalize(ygo_engine* e) { e->engine.finalize(); }
size_t ygo_engine_num_cards(const ygo_engine* e) { return e->engine.cards.size(); }

int ygo_engine_card_type(const ygo_engine* e, uint32_t code, uint32_t* type) {
  if (!e || !type) return -1;
  auto it = e->engine.cards.find(code);
  if (it == e->engine.cards.end()) return -1;
  *type = it->second.type;
  return 0;
}

ygo_env* ygo_env_new(ygo_engine* e, const uint32_t* main0, size_t n_main0, const uint32_t* extra0, size_t n_extra0,
                     const uint32_t* main1, size_t n_main1, const uint32_t* extra1, size_t n_extra1, uint64_t seed,
                     uint64_t flags, uint32_t lp, uint32_t hand, uint32_t draw, const uint64_t* seed_state,
                     size_t n_seed_state, const uint8_t* team_order, size_t n_team_order, int edopro_add, int auto_single,
                     char* err, size_t err_len) {
  try {
    ygo::DuelConfig c;
    c.main0 = vec(main0, n_main0);
    c.extra0 = vec(extra0, n_extra0);
    c.main1 = vec(main1, n_main1);
    c.extra1 = vec(extra1, n_extra1);
    c.seed = seed;
    c.flags = flags;
    c.lp = lp;
    c.hand = hand;
    c.draw = draw;
    if (seed_state && n_seed_state == 4) {
      std::memcpy(c.seed_state, seed_state, sizeof(c.seed_state));
      c.seed_state_set = true;
    }
    if (team_order && n_team_order == 2) {
      c.team_order[0] = team_order[0];
      c.team_order[1] = team_order[1];
    }
    c.edopro_add = edopro_add != 0;
    return new ygo_env(e->engine, std::move(c), auto_single != 0);
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return nullptr;
}

void ygo_env_free(ygo_env* env) { delete env; }

int ygo_env_reset(ygo_env* env, void* obs, char* err, size_t err_len) {
  try {
    return env->env.reset(*static_cast<Obs*>(obs)) ? 1 : 0;
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return -1;
}

int ygo_env_step(ygo_env* env, int32_t index, void* obs, char* err, size_t err_len) {
  try {
    return env->env.step(index, *static_cast<Obs*>(obs)) ? 1 : 0;
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return -1;
}

int32_t ygo_env_num_options(const ygo_env* env) { return env->env.num_options(); }
int32_t ygo_env_player(const ygo_env* env) { return env->env.player(); }
int32_t ygo_env_done(const ygo_env* env) { return env->env.done() ? 1 : 0; }
int32_t ygo_env_winner(const ygo_env* env) { return env->env.winner(); }
int32_t ygo_env_decisions(const ygo_env* env) { return env->env.decisions; }
int32_t ygo_env_truncated(const ygo_env* env) { return env->env.truncated; }

size_t ygo_env_num_errors(const ygo_env* env) {
  if (!env || !env->env.has_duel()) return 0;
  return env->env.duel().errors.size();
}

size_t ygo_env_error(const ygo_env* env, size_t i, char* buf, size_t len) {
  if (!env || !env->env.has_duel()) return 0;
  const auto& errs = env->env.duel().errors;
  if (i >= errs.size()) return 0;
  copy_err(errs[i].c_str(), buf, len);
  return errs[i].size();
}

void ygo_env_set_record_trace(ygo_env* env, int on) {
  if (!env) return;
  try {
    env->env.set_record_trace(on != 0);
  } catch (...) {
  }
}

int32_t ygo_env_msg_type(const ygo_env* env) {
  if (!env || !env->env.has_duel()) return -1;
  return env->env.msg_type();
}
int32_t ygo_env_dec_step(const ygo_env* env) {
  if (!env || !env->env.has_duel()) return -1;
  return env->env.dec_step();
}
int64_t ygo_env_dec_seq(const ygo_env* env) {
  if (!env || !env->env.has_duel()) return -1;
  return env->env.dec_seq();
}

size_t ygo_env_selected(const ygo_env* env, ygo_cardref* out, size_t cap) {
  if (!env || !env->env.has_duel()) return 0;
  const auto& sel = env->env.selected();
  size_t n = sel.size();
  size_t k = n < cap ? n : cap;
  for (size_t i = 0; i < k; ++i) {
    out[i].code = sel[i].code;
    out[i].controller = sel[i].controller;
    out[i].location = sel[i].location;
    out[i].sequence = sel[i].sequence;
    out[i].position = sel[i].position;
  }
  return n;
}

size_t ygo_env_options(const ygo_env* env, ygo_option* out, size_t cap) {
  if (!env || !env->env.has_duel()) return 0;
  const auto& opts = env->env.options();
  size_t n = opts.size();
  size_t k = n < cap ? n : cap;
  for (size_t i = 0; i < k; ++i) {
    const auto& o = opts[i];
    out[i].kind = static_cast<int32_t>(o.kind);
    out[i].has_card = o.has_card ? 1 : 0;
    if (o.has_card) {
      out[i].code = o.card.code;
      out[i].controller = o.card.controller;
      out[i].location = o.card.location;
      out[i].sequence = o.card.sequence;
      out[i].position = o.card.position;
    } else {
      out[i].code = out[i].controller = out[i].location = out[i].sequence = out[i].position = 0;
    }
    out[i].opt_code = o.code;
    out[i].desc = o.desc;
    out[i].arg = o.arg;
  }
  return n;
}

size_t ygo_env_pending_msg(const ygo_env* env, uint8_t* buf, size_t cap) {
  if (!env || !env->env.has_duel()) return 0;
  const ygo::Bytes* m = nullptr;
  try {
    m = env->env.pending_msg();
  } catch (...) {
    return 0;
  }
  if (!m) return 0;
  if (buf && cap >= m->size()) std::memcpy(buf, m->data(), m->size());
  return m->size();
}

size_t ygo_env_trace_len(const ygo_env* env) {
  if (!env || !env->env.has_duel()) return 0;
  return env->env.duel().trace.size();
}

size_t ygo_env_trace_msg(const ygo_env* env, size_t i, uint8_t* buf, size_t cap) {
  if (!env || !env->env.has_duel()) return 0;
  const auto& t = env->env.duel().trace;
  if (i >= t.size()) return 0;
  if (buf && cap >= t[i].size()) std::memcpy(buf, t[i].data(), t[i].size());
  return t[i].size();
}

int ygo_env_duel_state(const ygo_env* env, ygo_duel_state* out, char* err, size_t err_len) {
  if (!env || !env->env.has_duel()) {
    copy_err("no duel (reset not called)", err, err_len);
    return -1;
  }
  try {
    const auto& d = env->env.duel();
    out->turn = d.turn;
    out->turn_player = d.turn_player;
    out->first_player = d.first_player;
    out->phase = d.phase;
    out->winner = d.winner;
    out->win_reason = d.win_reason;
    out->select_hint = d.select_hint;
    return 0;
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return -1;
}

size_t ygo_env_query_field(const ygo_env* env, uint8_t* buf, size_t cap, char* err, size_t err_len) {
  if (!env || !env->env.has_duel()) {
    copy_err("no duel (reset not called)", err, err_len);
    return 0;
  }
  try {
    ygo::Bytes b = env->env.duel().query_field();
    if (buf && cap >= b.size()) std::memcpy(buf, b.data(), b.size());
    return b.size();
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return 0;
}

size_t ygo_env_query_location(const ygo_env* env, uint8_t controller, uint32_t location, uint32_t flags,
                              uint8_t* buf, size_t cap, char* err, size_t err_len) {
  if (!env || !env->env.has_duel()) {
    copy_err("no duel (reset not called)", err, err_len);
    return 0;
  }
  try {
    ygo::Bytes b = env->env.duel().query_location_raw(controller, location, flags);
    if (buf && cap >= b.size()) std::memcpy(buf, b.data(), b.size());
    return b.size();
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return 0;
}

size_t ygo_env_history_len(const ygo_env* env) {
  if (!env || !env->env.has_duel()) return 0;
  return static_cast<size_t>(env->env.history_len());
}

size_t ygo_env_history_log(const ygo_env* env, int32_t* out, size_t cap_rows) {
  if (!env || !env->env.has_duel()) return 0;
  auto log = env->env.history_log();
  size_t n = log.size();
  size_t k = n < cap_rows ? n : cap_rows;
  for (size_t i = 0; i < k; ++i) std::memcpy(out + i * ygo::HLOG_COLS, log[i].data(), ygo::HLOG_COLS * 4);
  return n;
}

size_t ygo_env_num_choices(const ygo_env* env) {
  if (!env || !env->env.has_duel()) return 0;
  return env->env.get_choices().size();
}

size_t ygo_env_choices(const ygo_env* env, int32_t* out, size_t cap) {
  if (!env || !env->env.has_duel()) return 0;
  const auto& c = env->env.get_choices();
  size_t n = c.size();
  size_t k = n < cap ? n : cap;
  for (size_t i = 0; i < k; ++i) out[i] = static_cast<int32_t>(c[i]);
  return n;
}

size_t ygo_env_cards(ygo_env* env, ygo_debug_card* out, size_t cap) {
  if (!env || !env->env.has_duel()) return 0;
  try {
    std::vector<ygo::DebugCard> rows = env->env.cards();
    size_t n = rows.size();
    size_t k = n < cap ? n : cap;
    for (size_t i = 0; i < k; ++i) {
      const auto& c = rows[i];
      out[i].controller = c.controller;
      out[i].location = c.location;
      out[i].sequence = c.sequence;
      out[i].position = c.position;
      out[i].code = c.code;
      out[i].cardid = c.cardid;
      out[i].is_public = c.is_public ? 1 : 0;
      out[i].known = c.known;
    }
    return n;
  } catch (...) {
    return 0;
  }
}

int ygo_env_observe_as(ygo_env* env, int32_t viewer, void* obs, char* err, size_t err_len) {
  if (!env) {
    copy_err("no env", err, err_len);
    return -1;
  }
  try {
    env->env.fill_obs_as(*static_cast<Obs*>(obs), viewer);
    return 0;
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return -1;
}

int ygo_env_replay_from_reset(ygo_env* env, const int32_t* choices, size_t n, void* obs, char* err, size_t err_len) {
  if (!env) {
    copy_err("no env", err, err_len);
    return -1;
  }
  try {
    std::vector<int> c;
    if (n) c.assign(choices, choices + n);
    return env->env.replay(c, *static_cast<Obs*>(obs)) ? 1 : 0;
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return -1;
}

int ygo_env_replay_keys(ygo_env* env, const int32_t* keys, const int32_t* ranks, size_t n, void* obs,
                        int32_t* resolved, size_t resolved_cap, size_t* n_resolved,
                        char* reason, size_t reason_len, char* err, size_t err_len) {
  if (n_resolved) *n_resolved = 0;
  if (reason && reason_len) reason[0] = '\0';
  if (!env) {
    copy_err("no env", err, err_len);
    return -1;
  }
  try {
    std::vector<std::array<int32_t, 8>> k;
    std::vector<int> r;
    k.reserve(n);
    r.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      std::array<int32_t, 8> a{};
      for (size_t j = 0; j < 8; ++j) a[j] = keys[i * 8 + j];
      k.push_back(a);
      r.push_back(ranks[i]);
    }
    ygo::Env::ReplayKeysResult res = env->env.replay_keys(k, r, *static_cast<Obs*>(obs));
    size_t m = res.resolved.size();
    if (n_resolved) *n_resolved = m;
    if (resolved) {
      size_t c = m < resolved_cap ? m : resolved_cap;
      for (size_t i = 0; i < c; ++i) resolved[i] = res.resolved[i];
    }
    if (!res.reason.empty()) {
      copy_err(res.reason.c_str(), reason, reason_len);
    }
    return res.ok ? 1 : 0;
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return -1;
}

size_t ygo_env_num_responses(const ygo_env* env) {
  if (!env || !env->env.has_duel()) return 0;
  return env->env.duel().responses.size();
}

size_t ygo_env_response(const ygo_env* env, size_t i, uint8_t* buf, size_t cap) {
  if (!env || !env->env.has_duel()) return 0;
  const auto& r = env->env.duel().responses;
  if (i >= r.size()) return 0;
  if (buf && cap >= r[i].size()) std::memcpy(buf, r[i].data(), r[i].size());
  return r[i].size();
}

ygo_decision* ygo_decision_new(ygo_engine* e, const uint8_t* msg, size_t len, const uint32_t* pool, size_t n_pool,
                               char* err, size_t err_len) {
  try {
    if (!e || !msg || len == 0) throw std::invalid_argument("decision needs an engine and a message");
    ygo::Bytes m(msg, msg + len);
    std::set<uint32_t> codes;
    if (pool) codes.insert(pool, pool + n_pool);
    auto d = ygo::make_decision(m, e->engine, [codes] { return codes; });
    if (!d) throw std::runtime_error("make_decision returned no decision");
    auto out = std::make_unique<ygo_decision>();
    out->dec = std::move(d);
    return out.release();
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return nullptr;
}

ygo_decision* ygo_decision_clone(const ygo_decision* d, char* err, size_t err_len) {
  try {
    if (!d || !d->dec) throw std::invalid_argument("no decision");
    auto out = std::make_unique<ygo_decision>();
    out->dec = d->dec->clone();
    return out.release();
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return nullptr;
}

ygo_decision* ygo_env_decision_clone(const ygo_env* env, char* err, size_t err_len) {
  try {
    if (!env || !env->env.has_decision()) return nullptr;
    auto d = env->env.clone_decision();
    if (!d) return nullptr;
    auto out = std::make_unique<ygo_decision>();
    out->dec = std::move(d);
    return out.release();
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return nullptr;
}

void ygo_decision_free(ygo_decision* d) { delete d; }
int32_t ygo_decision_msg_type(const ygo_decision* d) { return d && d->dec ? d->dec->msg_type : -1; }
int32_t ygo_decision_player(const ygo_decision* d) { return d && d->dec ? d->dec->player : -1; }
int32_t ygo_decision_step(const ygo_decision* d) { return d && d->dec ? d->dec->step : -1; }

size_t ygo_decision_options(const ygo_decision* d, ygo_option* out, size_t cap) {
  if (!d || !d->dec) return 0;
  std::vector<ygo::Option> opts;
  try {
    opts = d->dec->options();
  } catch (...) {
    return 0;  // never let a C++ exception cross the C boundary
  }
  size_t n = opts.size();
  size_t k = n < cap ? n : cap;
  for (size_t i = 0; i < k; ++i) {
    const auto& o = opts[i];
    out[i].kind = static_cast<int32_t>(o.kind);
    out[i].has_card = o.has_card ? 1 : 0;
    if (o.has_card) {
      out[i].code = o.card.code;
      out[i].controller = o.card.controller;
      out[i].location = o.card.location;
      out[i].sequence = o.card.sequence;
      out[i].position = o.card.position;
    } else {
      out[i].code = out[i].controller = out[i].location = out[i].sequence = out[i].position = 0;
    }
    out[i].opt_code = o.code;
    out[i].desc = o.desc;
    out[i].arg = o.arg;
  }
  return n;
}

int32_t ygo_decision_choose(ygo_decision* d, int32_t index, uint8_t* buf, size_t cap, size_t* out_len, char* err,
                            size_t err_len) {
  if (out_len) *out_len = 0;
  try {
    if (!d || !d->dec) throw std::invalid_argument("no decision");
    ygo::Bytes resp;
    bool done = d->dec->choose(index, resp);
    if (out_len) *out_len = resp.size();
    if (done && buf && cap >= resp.size()) std::memcpy(buf, resp.data(), resp.size());
    return done ? 1 : 0;
  } catch (const std::exception& ex) {
    copy_err(ex.what(), err, err_len);
  } catch (...) {
    copy_err("unknown C++ exception", err, err_len);
  }
  return -1;
}

}  // extern "C"
