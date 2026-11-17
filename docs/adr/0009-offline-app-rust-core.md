# ADR 0009: Offline app — Rust core over the C++/Lua engine, Tauri 2 shell

Date: 2026-11-01. Status: accepted (user). Plan: `docs/plans/2026-11-01-offline-app.md`. Branch: `offline-app`.

## Context
The user wants decklight on an Android phone (S23 Ultra) without internet, with all cards, new cards every week,
and one codebase for the web client, the app and the laptop ("what's important is that you implement well and
thoroughly"). Today the product runs as a Python server (Starlette, PyTorch, ctypes and pybind engine bindings).

## Decision
1. **Engine: EDOPro's C++ ygopro-core with Lua card scripts**, everywhere the product plays. New cards are data
   (script + database row + embedding row), downloaded in card packs; no rebuild.
2. **Play-time logic in one Rust library** (`decklight-core`): engine FFI over our C++ env, message parsing,
   view/session, ONNX inference, product search, learning features. The laptop server (`decklight-server`) and the
   phone app use it; Python research calls it through pyo3.
3. **App: Tauri 2**, whose webview shows the unchanged web client served by the in-app server on 127.0.0.1.
4. The Python implementation is the oracle until each Rust piece passes its parity gate (bit-identical where
   deterministic, statistical with ≥ 1,500 games for search).

## Alternatives rejected (user's choices, 2026-11-01)
- tsoj's Rust engine port as the app's engine: every weekly card becomes code to translate, verify and rebuild. It
  stays an optional laptop backend for search/training.
- Both engines per game: two engines to keep in agreement.
- Python on the phone (Chaquopy/Briefcase): fragile native packaging. Termux: personal only.
- Capacitor or a native Kotlin UI: more glue, or a second interface.

## Consequences
- A large port, staged A0–A7 with parity gates; the Python stays the reference until A7.
- Training keeps the C++ env and PyTorch; the networks are exported to ONNX for play.
- A weekly card-pack pipeline (A5) becomes part of operations.
