//! LAN discovery (the UDP answer EDOPro's LAN room list expects), a LAN listener for those answers, and the
//! Multirole room-list JSON (`http://host:7922/`).
//!
//! Ports and packet layout are `src/decklight/edopro/protocol.py` / `host.py` `_discovery_loop`:
//! a client broadcasts a `NETWORK_CLIENT_ID` (u16) to UDP 7920; the host answers to the sender's port 7921 with a
//! `HostPacket` (`<HHH2xI` + 40-byte name + 68-byte HostInfo = 120 bytes).

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::protocol::{
    CLIENT_VERSION, DISCOVERY_PORT, DISCOVERY_REPLY_PORT, HostInfo, NETWORK_CLIENT_ID, NETWORK_SERVER_ID,
    PRO_VERSION, ProtocolConfig, client_version_bytes, decode_name, encode_name,
};

/// A host's answer to a LAN discovery request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostPacket {
    pub server_id: u16,
    pub pro_version: u16,
    pub port: u16,
    pub name: String,
    pub info: HostInfo,
}

/// The `HostPacket` a host sends for its TCP `port` and room `name`.
pub fn build_host_packet(port: u16, name: &str, info: &HostInfo) -> Vec<u8> {
    let mut o = Vec::with_capacity(120);
    o.extend_from_slice(&(NETWORK_SERVER_ID as u16).to_le_bytes());
    o.extend_from_slice(&PRO_VERSION.to_le_bytes());
    o.extend_from_slice(&port.to_le_bytes());
    o.extend_from_slice(&[0, 0]);
    o.extend_from_slice(&0u32.to_le_bytes());
    o.extend_from_slice(&encode_name(name, 20));
    o.extend_from_slice(&info.pack());
    debug_assert_eq!(o.len(), 120);
    o
}

pub fn parse_host_packet(data: &[u8]) -> Result<HostPacket, String> {
    if data.len() < 120 {
        return Err(format!("host packet too short: {} (need 120)", data.len()));
    }
    let server_id = u16::from_le_bytes([data[0], data[1]]);
    let pro_version = u16::from_le_bytes([data[2], data[3]]);
    let port = u16::from_le_bytes([data[4], data[5]]);
    let name = decode_name(&data[12..52]);
    let info = HostInfo::unpack(&data[52..120])?;
    Ok(HostPacket { server_id, pro_version, port, name, info })
}

/// `host.py` `host_info(P.client_version_bytes(*P.CLIENT_VERSION))` for the discovery answer.
pub fn discovery_info() -> HostInfo {
    let v = client_version_bytes(CLIENT_VERSION.0, CLIENT_VERSION.1, CLIENT_VERSION.2, CLIENT_VERSION.3);
    crate::protocol::host_info(v)
}

/// Answer EDOPro LAN requests until `stop` is set. Runs on its own thread; returns it.
pub fn spawn_discovery(tcp_port: u16, name: String, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run_discovery(tcp_port, &name, &stop) {
            eprintln!("ALERT edopro_host: UDP discovery disabled: {e}");
        }
    })
}

pub fn run_discovery(tcp_port: u16, name: &str, stop: &AtomicBool) -> Result<(), String> {
    let sock = UdpSocket::bind(("0.0.0.0", DISCOVERY_PORT)).map_err(|e| format!("bind {DISCOVERY_PORT}: {e}"))?;
    let _ = sock.set_broadcast(true);
    // SO_REUSEADDR equivalent: std has no setter; rebinding after a quick restart is handled by the OS here.
    sock.set_read_timeout(Some(Duration::from_millis(500))).map_err(|e| e.to_string())?;
    let info = discovery_info();
    let mut buf = [0u8; 256];
    while !stop.load(Ordering::Relaxed) {
        match sock.recv_from(&mut buf) {
            Ok((n, addr)) => {
                if n >= 2 && u16::from_le_bytes([buf[0], buf[1]]) == NETWORK_CLIENT_ID as u16 {
                    let pkt = build_host_packet(tcp_port, name, &info);
                    let _ = sock.send_to(&pkt, (addr.ip(), DISCOVERY_REPLY_PORT));
                }
            }
            Err(e) => match e.kind() {
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => continue,
                _ => return Err(e.to_string()),
            },
        }
    }
    Ok(())
}

/// Broadcast a discovery request and collect every answer for `timeout`, with the address it came from (the room
/// is joined at that address and `HostPacket::port`).
pub fn discover_lan(timeout: Duration) -> Result<Vec<(IpAddr, HostPacket)>, String> {
    discover_lan_on((Ipv4Addr::BROADCAST, DISCOVERY_PORT), timeout)
}

pub fn discover_lan_on(target: (Ipv4Addr, u16), timeout: Duration) -> Result<Vec<(IpAddr, HostPacket)>, String> {
    let sock = UdpSocket::bind(("0.0.0.0", DISCOVERY_REPLY_PORT))
        .map_err(|e| format!("bind {DISCOVERY_REPLY_PORT}: {e}"))?;
    let _ = sock.set_broadcast(true);
    sock.set_read_timeout(Some(Duration::from_millis(100))).map_err(|e| e.to_string())?;
    sock.send_to(&(NETWORK_CLIENT_ID as u16).to_le_bytes(), SocketAddr::from(target))
        .map_err(|e| format!("send: {e}"))?;
    let mut out = Vec::new();
    let mut buf = [0u8; 512];
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match sock.recv_from(&mut buf) {
            Ok((n, from)) => {
                if let Ok(p) = parse_host_packet(&buf[..n]) {
                    let ip = from.ip();
                    if !out.iter().any(|(a, q): &(IpAddr, HostPacket)| *a == ip && q.port == p.port) {
                        out.push((ip, p));
                    }
                }
            }
            Err(e) => match e.kind() {
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => continue,
                _ => return Err(e.to_string()),
            },
        }
    }
    Ok(out)
}

/// One room of Multirole's lobby listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Room {
    pub roomid: u32,
    pub roomnotes: String,
    pub roomname: String,
    pub raw: Value,
}

/// Multirole's `GET /` listing: `{"rooms": [{"roomid": ..., "roomnotes": ..., ...}, ...]}`.
pub fn parse_room_list(text: &str) -> Result<Vec<Room>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("room list is not JSON: {e}"))?;
    let rooms = v.get("rooms").and_then(Value::as_array).ok_or("room list has no 'rooms' array")?;
    let mut out = Vec::with_capacity(rooms.len());
    for r in rooms {
        let roomid = r
            .get("roomid")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("room without a numeric roomid: {r}"))? as u32;
        out.push(Room {
            roomid,
            roomnotes: r.get("roomnotes").and_then(Value::as_str).unwrap_or("").to_string(),
            roomname: r.get("roomname").and_then(Value::as_str).unwrap_or("").to_string(),
            raw: r.clone(),
        });
    }
    Ok(out)
}

/// Minimal HTTP/1.0 GET (`http://host:port/path`) for the room listing; no extra dependency for one request.
pub fn http_get(host: &str, port: u16, path: &str) -> Result<String, String> {
    let mut s = TcpStream::connect((host, port)).map_err(|e| format!("connect {host}:{port}: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
    write!(s, "GET {path} HTTP/1.0\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n")
        .map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    match text.find("\r\n\r\n") {
        Some(i) => Ok(text[i + 4..].to_string()),
        None => Ok(text),
    }
}

/// `http://host:port/` -> rooms (the client's `--list-rooms`).
pub fn fetch_room_list(host: &str, port: u16) -> Result<Vec<Room>, String> {
    parse_room_list(&http_get(host, port, "/")?)
}

/// `host:port` (or a full `http://host:port/` URL) -> `(host, port)`.
pub fn parse_http_target(s: &str) -> Result<(String, u16), String> {
    let t = s.strip_prefix("http://").unwrap_or(s);
    let t = t.trim_end_matches('/');
    match t.rsplit_once(':') {
        Some((h, p)) => Ok((h.to_string(), p.parse().map_err(|_| format!("bad port in {s:?}"))?)),
        None => Err(format!("want host:port or http://host:port, got {s:?}")),
    }
}

/// A client's own wire versions for the discovery answer (what it announces).
pub fn default_protocol_version() -> [u8; 4] {
    let p = ProtocolConfig::default();
    client_version_bytes(p.client_version.0, p.client_version.1, p.client_version.2, p.client_version.3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_packet_round_trip() {
        let info = discovery_info();
        let pkt = build_host_packet(7911, "ygo-zero", &info);
        assert_eq!(pkt.len(), 120);
        let p = parse_host_packet(&pkt).unwrap();
        assert_eq!(p.server_id, NETWORK_SERVER_ID as u16);
        assert_eq!(p.pro_version, PRO_VERSION);
        assert_eq!(p.port, 7911);
        assert_eq!(p.name, "ygo-zero");
        assert_eq!(p.info, info);
    }

    #[test]
    fn short_host_packets_are_errors() {
        for n in 0..120 {
            assert!(parse_host_packet(&vec![0xFF; n]).is_err());
        }
    }

    #[test]
    fn room_list_json() {
        let text = r#"{"rooms":[{"roomid":7,"roomnotes":"games may be shared anonymously for research",
            "roomname":"r","host":"x"},{"roomid":9,"roomnotes":"","roomname":"q"}]}"#;
        let rooms = parse_room_list(text).unwrap();
        assert_eq!(rooms.len(), 2);
        assert_eq!(rooms[0].roomid, 7);
        assert_eq!(rooms[1].roomnotes, "");
        assert!(parse_room_list("{}").is_err());
        assert!(parse_room_list("nope").is_err());
    }

    #[test]
    fn http_target() {
        assert_eq!(parse_http_target("127.0.0.1:7922").unwrap(), ("127.0.0.1".into(), 7922));
        assert_eq!(parse_http_target("http://h:1/").unwrap(), ("h".into(), 1));
        assert!(parse_http_target("h").is_err());
    }
}
