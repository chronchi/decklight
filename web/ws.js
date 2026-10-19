// Minimal WebSocket client for the decklight game server.
// Connects to /ws?t=<token>, resyncs after every reconnect, backs off.
export function connectWS(token, hooks) {
  hooks = hooks || {};
  let ws = null;
  let backoffMs = 500;
  let closed = false;
  let timer = 0;
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
    ws.onclose = () => {
      if (closed) return;
      if (hooks.onDrop) hooks.onDrop();
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
    timer = setTimeout(open, backoffMs);
    backoffMs = Math.min(backoffMs * 2, 10000);
  }

  open();
  return {
    send(obj) {
      if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
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
