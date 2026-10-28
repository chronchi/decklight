// OCG_DuelClone: one function more than the OCG C API (`third_party/ygopro-core/ocgapi.h`).
// Implemented by the Rust shim (`rust/ocgshim`); the C++ core has no such symbol,
// so this header is only used in the Rust build (`-DYGOENV_RUST=1`).
#pragma once

#include "ocgapi.h"

#ifdef __cplusplus
extern "C" {
#endif

// payload3: the clone's logHandler payload (the original's would log into the original duel).
OCG_Duel OCG_DuelClone(OCG_Duel duel, void* payload3);

#ifdef __cplusplus
}
#endif
