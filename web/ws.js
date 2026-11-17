// Minimal WebSocket client for the decklight game server.
// Connects to /ws?t=<token>, resyncs after every reconnect, backs off.
// While the app is starting (the client is served before the bot is ready) the
// retries are quiet: 200 ms -> 1 s and no error UI for the first 30 s.
export function connectWS(token, hooks) {
  hooks = hooks || {};
  let ws = null;
  let backoffMs = 200;
  let closed = false;
  let connectedOnce = false;
  let timer = 0;
  const startedAt = Date.now();
  const STARTUP_MS = 30000;
  const starting = () => !connectedOnce && Date.now() - startedAt < STARTUP_MS;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = proto + "://" + location.host + "/ws?t=" + encodeURIComponent(token);

  function open() {
    if (closed) return;
    let sock;
    try {
      sock = new WebSocket(url);
    } catch (e) {
      schedule();
      return;
    }
    ws = sock;
    ws.onopen = () => {
      connectedOnce = true;
      backoffMs = 500;
      if (hooks.onOpen) hooks.onOpen();
      try {
        ws.send(JSON.stringify({t: "resync"}));
      } catch (e) { /* ignore */ }
    };
    ws.onmessage = (ev) => {
      let msg = null;
      try {
        msg = JSON.parse(ev.data);
      } catch (e) {
        return;
      }
      if (hooks.onMessage) hooks.onMessage(msg);
    };
    ws.onclose = (ev) => {
      if (closed) return;
      // 4400: another window took this session over. Reconnecting would take it back and the two would trade it
      // forever, so wait until the user asks to continue here (resume()).
      if (ev && ev.code === 4400) {
        ws = null;
        if (hooks.onTakenOver) hooks.onTakenOver();
        return;
      }
      // Starting up: the gate only serves the client until the server is ready, so a refused socket is expected.
      if (!starting() && hooks.onDrop) hooks.onDrop();
      schedule();
    };
    ws.onerror = () => {
      try {
        ws.close();
      } catch (e) { /* ignore */ }
    };
  }

  function schedule() {
    clearTimeout(timer);
    const quiet = starting();
    timer = setTimeout(open, quiet ? Math.min(backoffMs, 1000) : Math.max(backoffMs, 500));
    backoffMs = quiet ? Math.min(backoffMs * 2, 1000) : Math.min(backoffMs * 2, 10000);
  }

  open();
  return {
    send(obj) {
      if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
    },
    resume() {
      if (closed || ws) return;
      backoffMs = 500;
      open();
    },
    close() {
      closed = true;
      clearTimeout(timer);
      try {
        if (ws) ws.close();
      } catch (e) { /* ignore */ }
    },
  };
}
