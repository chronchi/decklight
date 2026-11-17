// C ABI over the C++ env (Engine, Env, Obs) for the Rust core (crate ygoenv-sys, ADR 0009).
// Same objects the pybind module wraps (bindings.cpp); no Python. Every call that can throw catches and
// returns an error code with the message copied into `err` (truncated to `err_len`, always NUL-terminated).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ygo_engine ygo_engine;
typedef struct ygo_env ygo_env;
typedef struct ygo_decision ygo_decision;

// Layout check for the Rust mirror of ygo::Obs: total size and the byte offset of field i (in declaration order:
// cards, card_mask, card_known, known_code, priv_code, global, chain, chain_mask, actions, action_mask, history,
// history_mask, hist_long, hist_long_mask); offset(i) for i >= 14 returns SIZE_MAX.
uint32_t ygo_obs_version(void);
size_t ygo_obs_size(void);
size_t ygo_obs_offset(uint32_t field);

ygo_engine* ygo_engine_new(void);
void ygo_engine_free(ygo_engine* e);
void ygo_engine_set_precompile(ygo_engine* e, int on);
// setcodes: n entries, without the terminating 0 (added here, as Engine::add_card does).
void ygo_engine_add_card(ygo_engine* e, uint32_t code, uint32_t alias, const uint16_t* setcodes, size_t n_setcodes,
                         uint32_t type, uint32_t level, uint32_t attribute, uint64_t race, int32_t attack,
                         int32_t defense, uint32_t lscale, uint32_t rscale, uint32_t link_marker);
// Script file name (e.g. "c12345.lua", "utility.lua") -> path on disk. Later calls override earlier ones.
void ygo_engine_set_script(ygo_engine* e, const char* name, const char* path);
// Same, but the bytes are [offset, offset + len) of `path` (a packed CardScripts.pack entry).
void ygo_engine_set_script_range(ygo_engine* e, const char* name, const char* path, uint64_t offset, uint32_t len);
// The script's bytes after `set_script`/`set_script_range` (precompiled when enabled), for smoke tests.
// Needed size; copies only if it fits. 0 = unknown script.
size_t ygo_engine_script(ygo_engine* e, const char* name, uint8_t* buf, size_t cap);
void ygo_engine_finalize(ygo_engine* e);
size_t ygo_engine_num_cards(const ygo_engine* e);
// 0 = found and *type set to the card's TYPE_* flags; -1 = the code is not in the card DB.
int ygo_engine_card_type(const ygo_engine* e, uint32_t code, uint32_t* type);

// Main decks are passed in the order they reach the core (already shuffled by the caller, like
// DuelConfig.main_of in Python). The engine must outlive the env. NULL on error.
// seed_state: 4 entries, used instead of seed_words(seed) when n_seed_state == 4. team_order: 2 entries,
// the order the teams' cards are added to the core (n_team_order == 0 -> {0, 1}). edopro_add != 0: every main-deck
// card at sequence 0, like EDOPro/Multirole (DuelConfig::edopro_add).
ygo_env* ygo_env_new(ygo_engine* e, const uint32_t* main0, size_t n_main0, const uint32_t* extra0, size_t n_extra0,
                     const uint32_t* main1, size_t n_main1, const uint32_t* extra1, size_t n_extra1, uint64_t seed,
                     uint64_t flags, uint32_t lp, uint32_t hand, uint32_t draw, const uint64_t* seed_state,
                     size_t n_seed_state, const uint8_t* team_order, size_t n_team_order, int edopro_add, int auto_single,
                     char* err, size_t err_len);
void ygo_env_free(ygo_env* env);

// reset/step: 1 = a decision is pending and `obs` (an ygo::Obs, ygo_obs_size() bytes) holds it; 0 = the duel ended
// (`obs` untouched); -1 = error (message in err).
int ygo_env_reset(ygo_env* env, void* obs, char* err, size_t err_len);
int ygo_env_step(ygo_env* env, int32_t index, void* obs, char* err, size_t err_len);

int32_t ygo_env_num_options(const ygo_env* env);
int32_t ygo_env_player(const ygo_env* env);
int32_t ygo_env_done(const ygo_env* env);
int32_t ygo_env_winner(const ygo_env* env);
int32_t ygo_env_decisions(const ygo_env* env);
int32_t ygo_env_truncated(const ygo_env* env);
// Lua/script errors the core reported so far (Duel::errors): count, and the i-th message copied into buf.
size_t ygo_env_num_errors(const ygo_env* env);
size_t ygo_env_error(const ygo_env* env, size_t i, char* buf, size_t len);

// ---- play-layer surface (A1a): decision-level parity with PyEnv(auto_single=False) ----
// All queries below are safe before `reset` (no duel): they return 0/-1 with `err` set, never crash.
// Buffer protocol: each `*_msg`/`query_*` returns the needed size; copies only if it fits.
void ygo_env_set_record_trace(ygo_env* env, int on);
int32_t ygo_env_msg_type(const ygo_env* env);
int32_t ygo_env_dec_step(const ygo_env* env);
// Monotonic decision id: changes when a new core selection message arrives (stable across its picks).
int64_t ygo_env_dec_seq(const ygo_env* env);

typedef struct ygo_cardref {
  uint32_t code, controller, location, sequence, position;
} ygo_cardref;
// Number of selected refs; copies up to `cap` entries.
size_t ygo_env_selected(const ygo_env* env, ygo_cardref* out, size_t cap);

typedef struct ygo_option {
  int32_t kind;
  int32_t has_card;
  uint32_t code, controller, location, sequence, position;
  uint32_t opt_code;
  uint64_t desc;
  int64_t arg;
} ygo_option;
// Number of options (== ygo_env_num_options); copies up to `cap` entries.
size_t ygo_env_options(const ygo_env* env, ygo_option* out, size_t cap);

// Needed size of the pending select message's raw bytes (0 = no decision); copies only if it fits.
size_t ygo_env_pending_msg(const ygo_env* env, uint8_t* buf, size_t cap);
size_t ygo_env_trace_len(const ygo_env* env);
// Needed size of trace message i (0 = out of range / no duel); copies only if it fits.
size_t ygo_env_trace_msg(const ygo_env* env, size_t i, uint8_t* buf, size_t cap);

typedef struct ygo_duel_state {
  int32_t turn, turn_player, first_player, phase, winner, win_reason;
  uint64_t select_hint;
} ygo_duel_state;
// 0 = ok, -1 = no duel (message in err).
int ygo_env_duel_state(const ygo_env* env, ygo_duel_state* out, char* err, size_t err_len);

// Needed size of the raw query buffers (0 + err if no duel); copy only if it fits.
size_t ygo_env_query_field(const ygo_env* env, uint8_t* buf, size_t cap, char* err, size_t err_len);
size_t ygo_env_query_location(const ygo_env* env, uint8_t controller, uint32_t location, uint32_t flags,
                              uint8_t* buf, size_t cap, char* err, size_t err_len);

size_t ygo_env_history_len(const ygo_env* env);
// Number of history rows; copies up to `cap_rows` rows of 14 int32.
size_t ygo_env_history_log(const ygo_env* env, int32_t* out, size_t cap_rows);
size_t ygo_env_num_choices(const ygo_env* env);
// Number of choices; copies up to `cap` entries.
size_t ygo_env_choices(const ygo_env* env, int32_t* out, size_t cap);

// ---- honest-search surface (A3a): the pybind `Env::cards`, `Env::replay`, `Env::replay_keys` ----
typedef struct ygo_debug_card {
  uint32_t controller, location, sequence, position, code, cardid;
  uint8_t is_public, known;
} ygo_debug_card;
// Every card of both players (DECK/HAND/MZONE/SZONE/GRAVE/REMOVED/EXTRA): number of rows; copies up to `cap`.
size_t ygo_env_cards(ygo_env* env, ygo_debug_card* out, size_t cap);

// reset + step every choice in one call. 1 = a decision is pending and `obs` holds it; 0 = the duel ended;
// -1 = error (message in err).
int ygo_env_replay_from_reset(ygo_env* env, const int32_t* choices, size_t n, void* obs, char* err, size_t err_len);

// The observation from `viewer`'s side (0/1) at the current position, whoever is to act (A4 hints: the value net
// never sees the other seat's hand). A pending decision keeps its options; after the end there are none.
// 0 = ok, -1 = error (no duel, bad viewer).
int ygo_env_observe_as(ygo_env* env, int32_t viewer, void* obs, char* err, size_t err_len);

// reset + semantic replay: keys is n rows of 8 int32 (option_row columns [0,2,3,4,5,6,7,8]), ranks is n entries.
// Returns 1 when every key resolved and `obs` holds the final decision; 0 when the duel ended or a key failed
// (`reason` is then "no_key" or "replay_error", or empty when the duel ended exactly at the end); -1 on a C++
// exception. `resolved` receives the indices resolved so far (up to `resolved_cap`; the count goes to `n_resolved`).
int ygo_env_replay_keys(ygo_env* env, const int32_t* keys, const int32_t* ranks, size_t n, void* obs,
                        int32_t* resolved, size_t resolved_cap, size_t* n_resolved,
                        char* reason, size_t reason_len, char* err, size_t err_len);

// Raw OCG response bytes of response i (match record v1): needed size; copies only if it fits.
size_t ygo_env_num_responses(const ygo_env* env);
size_t ygo_env_response(const ygo_env* env, size_t i, uint8_t* buf, size_t cap);

// ---- decisions without a duel (client mode): make_decision(msg, engine, pool) as an opaque handle ----
// `pool` is the ANNOUNCE_CARD pool (n_pool codes); NULL/0 uses the engine's card DB only.
ygo_decision* ygo_decision_new(ygo_engine* e, const uint8_t* msg, size_t len, const uint32_t* pool, size_t n_pool,
                               char* err, size_t err_len);
// Independent copy with its own pick state (the engine and message stay shared). NULL on error.
ygo_decision* ygo_decision_clone(const ygo_decision* d, char* err, size_t err_len);
// The env's pending decision, copied. NULL when no decision is pending (or on error).
ygo_decision* ygo_env_decision_clone(const ygo_env* env, char* err, size_t err_len);
void ygo_decision_free(ygo_decision* d);
int32_t ygo_decision_msg_type(const ygo_decision* d);
int32_t ygo_decision_player(const ygo_decision* d);
int32_t ygo_decision_step(const ygo_decision* d);
// Number of options; copies up to `cap` entries.
size_t ygo_decision_options(const ygo_decision* d, ygo_option* out, size_t cap);
// 1 = the selection is complete and its response is in buf (needed size in *out_len; copied if it fits);
// 0 = another pick is needed; -1 = error. `out_len` may be NULL.
int32_t ygo_decision_choose(ygo_decision* d, int32_t index, uint8_t* buf, size_t cap, size_t* out_len, char* err,
                            size_t err_len);

#ifdef __cplusplus
}
#endif
