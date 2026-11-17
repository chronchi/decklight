# S3 spike: Tauri 2 Android shell serving `web/` from an in-app loopback server

Status: **pass** (2026-11-01). The emulator lobby renders, the client's WebSocket
connects, ping round-trips, wrong-token closes with 4401. See `docs/briefs/` for
the task brief; evidence in `data/cache/s3/` (not in git).

## What it is

A Tauri 2 app (`com.decklight.s3spike`) that, on startup, spawns an **axum**
HTTP + WebSocket server on `127.0.0.1:0` (OS-picked port) inside the app process
and shows the unchanged web client (`web/`, embedded in the binary with
`rust-embed` + `debug-embed`, so debug builds also serve the real files) from
`http://127.0.0.1:<port>/?t=<token>`.

- `GET /ws?t=<token>`: WebSocket. Wrong/missing token → close 4401 (same code as
  `src/decklight/web/server.py` `ws_handler`). On connect it sends the bytes of
  `web/mock/01-lobby.json` verbatim — that file already IS the raw live lobby
  message (`{"t":"lobby","decks":[...]}`; the client tolerates the missing
  `models`/`model_default` keys by falling back to the base model).
- `{"t":"ping",...}` → `{"t":"pong",...same fields}`; `{"t":"resync"}` → lobby
  again (mirrors the live server's resync path); anything else is logged to
  logcat and ignored.
- Token: 12 random bytes, base64 URL-safe no-pad, generated per launch, printed
  to logcat (`S3 ready: ...`) so the test can use it.
- Window: built in `setup` with the final URL
  (`WebviewWindowBuilder` + `WebviewUrl::External`). A placeholder window +
  `navigate()` in `setup` does NOT work — the navigation races webview creation
  and is silently lost (first boot showed only the placeholder page).
- Android: `usesCleartextTraffic=false` for debug AND release; cleartext is
  allowed only for `127.0.0.1`/`localhost` via
  `gen/android/app/src/main/res/xml/network_security_config.xml`.
  INTERNET permission: **required** — without it the `127.0.0.1:0` bind fails
  with EPERM on API 36 (tested by removing it: `bind 127.0.0.1:0: Operation not
  permitted`, app aborts at startup).

## Exact build / install / test commands

All from `rust/spikes/tauri-shell/`, with throttles (`taskset -c 0-3`,
`cargo -j 3` via `CARGO_BUILD_JOBS=3`, gradle `--max-workers=3` in
`gen/android/gradle.properties`, JVM heap 2 GB):

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export ANDROID_HOME=$HOME/Android/Sdk NDK_HOME=$HOME/Android/Sdk/ndk/30.0.16248370
export JAVA_HOME=$HOME/.local/jdk/jdk-21.0.8+9   # Temurin 21; see gotcha 1
export CARGO_BUILD_JOBS=3

# first-time only (from this dir):
npx --yes @tauri-apps/cli@^2 android init   # generated src-tauri/gen/android

# debug APK for the emulator (x86_64):
taskset -c 0-3 npx --yes @tauri-apps/cli@^2 android build --debug --apk --target x86_64
# → src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk

# release APK for arm64 (unsigned; NOT installed anywhere):
taskset -c 0-3 npx --yes @tauri-apps/cli@^2 android build --apk --target aarch64
# → src-tauri/gen/android/app/build/outputs/apk/universal/release/app-universal-release-unsigned.apk

A=~/Android/Sdk/platform-tools/adb
$A install -r src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk
$A logcat -c
$A shell monkey -p com.decklight.s3spike.debug -c android.intent.category.LAUNCHER 1
$A logcat -d | grep -a s3_shell_lib        # port, token, ws connect, ping/pong
$A exec-out screencap -p > data/cache/s3/lobby.png

# ping/pong + 4401 from the laptop (needs the token from logcat):
$A forward tcp:<port> tcp:<port>
uv run --with websockets python3 - <<EOF
import asyncio, json, websockets
async def main():
    url = 'ws://127.0.0.1:<port>/ws?t=<token>'
    async with websockets.connect(url) as ws:
        print('lobby:', json.loads(await ws.recv())['t'])
        await ws.send(json.dumps({'t': 'ping', 'from': 'laptop', 'n': 1}))
        print('pong:', json.loads(await ws.recv()))
asyncio.run(main())
EOF
```

## Gotchas

1. **No JDK on the laptop** (only a JRE; `javac` missing). Gradle/AGP fails with
   `Toolchain installation ... does not provide the required capabilities:
   [JAVA_COMPILER]`. Fix used: Temurin 21 tarball unpacked to
   `~/.local/jdk/jdk-21.0.8+9`, `JAVA_HOME` pointed at it (outside git, fine).
2. **rust-embed 8 reads from disk in debug builds.** `WebAsset::get()` returned
   `None` on device (panic `web/mock/01-lobby.json embedded`) because the
   relative `../../../../web` path doesn't exist in the app's CWD. Fix:
   `features = ["debug-embed"]` — release builds always embed.
3. **axum 0.8 `CloseCode` is a plain `u16`.** `CloseCode::Library(4401)` and
   `4401.into()` both fail to compile; use `code: 4401u16`.
4. **`tauri android build` needs an npm shim.** The gradle rust plugin runs
   `npm run tauri android android-studio-script --target <arch>` with CWD
   `src-tauri/` and fails without `package.json`. Fix: `src-tauri/package.json`
   with `"tauri": "npx --yes @tauri-apps/cli@^2"` (no node_modules needed).
   Created with `npx @tauri-apps/cli init` (flags: `-A "S3 Shell" -W "decklight
   S3" -D ../dist`); identifier edited to `com.decklight.s3spike` afterwards.
5. **compileSdk/targetSdk 37 → 36.** The generated project wants API 37 but only
   `platforms;android-36` is installed (`sdkmanager` has no android-37 package
   in this channel); edited to 36 in `gen/android/app/build.gradle.kts`.
   (Also installed nothing new from sdkmanager for this reason.)
6. **Stale-bytes APK.** One incremental rebuild produced a 340 MB APK whose zip
   entries only sum to 182 MB (orphaned bytes from the previous build prepended;
   `unzip -t` still passes). Fix: `rm -rf .../build/outputs` and rebuild → sane
   174,255,698 bytes. Wipe `outputs/` before any size-critical build.
7. **INTERNET permission IS required (answered).** Removing
   `android.permission.INTERNET` makes the loopback bind fail with
   `Os { code: 1, kind: PermissionDenied, "Operation not permitted" }` on API
   36 and the app aborts. Keep the permission.

## Numbers (2026-11-01)

- debug x86_64 APK: **174,255,698 B** (~166 MiB; debug symbols kept per Tauri's
  `keepDebugSymbols`, unoptimized). Release arm64 APK (unsigned):
  **12,051,000 B** (~11.5 MiB).
- arm64 release build: 4 m 19 s wall (`taskset -c 0-3`, `-j 3`).
- Cold start on `decklight_test` emulator: process start → ws connect ≈ 2.7 s
  (`ActivityManager: Start proc` → `S3 ws: client connected`).

## If the client needs a change to work in the app

None needed for the lobby: the client already reads `t` from the query string
(`web/app.js` `TOKEN`) and connects to `/ws` on the same host (`web/ws.js`).
Caveat: the client sends `{"t":"puzzles"}` right after the lobby; the spike
only logs it. The full server (A2) answers it. No change to `web/` was made.
