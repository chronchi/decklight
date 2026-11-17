// LiteRT-LM speed check on the phone (NOTEBOOK 2026-11-15): load a .litertlm model on a backend, send each prompt
// file (a conversation message JSON) as a new conversation, print the reply head and the prefill/decode speeds.
//   litert_bench <model> <backend: gpu|cpu> <cache_dir> <message.json>...
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "engine.h"
#include "conversation.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static char* slurp(const char* p) {
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
  fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
  char* b = malloc(n + 1); fread(b, 1, n, f); b[n] = 0; fclose(f); return b;
}
#define CHECK(x) do { LiteRtLmStatusCode s_ = (x); if (s_ != kLiteRtLmStatusOk) { fprintf(stderr, "FAILED %s: %d\n", #x, (int)s_); return 1; } } while (0)

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: litert_bench <model> <backend> <cache_dir> <message.json>...\n"); return 2; }
  LiteRtLmEngineSettings* st = NULL;
  CHECK(litert_lm_engine_settings_create(argv[1], argv[2], NULL, NULL, &st));
  CHECK(litert_lm_engine_settings_set_max_num_tokens(st, 8192));
  CHECK(litert_lm_engine_settings_set_cache_dir(st, argv[3]));
  CHECK(litert_lm_engine_settings_enable_benchmark(st));
  double t0 = now();
  LiteRtLmEngine* eng = NULL;
  CHECK(litert_lm_engine_create(st, &eng));
  printf("engine %s ready in %.1f s\n", argv[2], now() - t0); fflush(stdout);
  for (int i = 4; i < argc; i++) {
    char* msg = slurp(argv[i]);
    LiteRtLmConversation* conv = NULL;
    CHECK(litert_lm_conversation_create(eng, NULL, &conv));
    double t1 = now();
    LiteRtLmJsonResponse* r = NULL;
    CHECK(litert_lm_conversation_send_message(conv, msg, NULL, NULL, &r));
    double dt = now() - t1;
    const char* out = NULL;
    litert_lm_json_response_get_string(r, &out);
    printf("%s: %.1f s; reply %.160s\n", argv[i], dt, out ? out : "(none)");
    LiteRtLmBenchmarkInfo* b = NULL;
    if (litert_lm_conversation_get_benchmark_info(conv, &b) == kLiteRtLmStatusOk) {
      double ttft = 0, pf = 0, dc = 0; size_t pn = 0, dn = 0;
      litert_lm_benchmark_info_get_time_to_first_token(b, &ttft);
      litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(b, 0, &pf);
      litert_lm_benchmark_info_get_decode_tokens_per_sec_at(b, 0, &dc);
      litert_lm_benchmark_info_get_prefill_token_count_at(b, 0, &pn);
      litert_lm_benchmark_info_get_decode_token_count_at(b, 0, &dn);
      printf("  ttft %.2f s; prefill %zu tok %.1f t/s; decode %zu tok %.1f t/s\n", ttft, pn, pf, dn, dc);
      litert_lm_benchmark_info_delete(b);
    }
    fflush(stdout);
    litert_lm_json_response_delete(r);
    litert_lm_conversation_delete(conv);
    free(msg);
  }
  litert_lm_engine_delete(eng);
  litert_lm_engine_settings_delete(st);
  return 0;
}
