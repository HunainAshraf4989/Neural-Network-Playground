// React hook owning the websocket to the backend. Holds the shared architecture
// from `state` broadcasts and exposes `send(message)` for the §12 client
// messages. The backend is the single source of truth, so this hook never
// mutates `arch` optimistically — it only renders what the server broadcasts
// back. `ack`/`error` replies are surfaced as a transient status line.
//
// Connection lifecycle (deploy S1): reconnects with exponential backoff +
// jitter (1.5 s base, 15 s cap — a fixed short loop would hammer a free-tier
// backend that takes ~30 s to cold-start). While disconnected, `status` says
// *why* the canvas is empty: "connecting" (backend reachable, ws not up yet)
// vs "waking" (the `/healthz` probe also fails — a sleeping HF Space being
// woken, or no backend at all).

import { useCallback, useEffect, useRef, useState } from "react";

import * as proto from "./protocol.js";

// Single backend base URL (http(s)://host[:port]); ws and healthz URLs are
// derived from it, so prod config is one env var: VITE_BACKEND_URL.
const BACKEND_URL = import.meta.env?.VITE_BACKEND_URL || "http://localhost:8765";
const RECONNECT_BASE_MS = 1500;
const RECONNECT_CAP_MS = 15000;

export function wsUrl(backendUrl) {
  return backendUrl.replace(/^http/, "ws").replace(/\/+$/, "") + "/ws";
}

export function healthzUrl(backendUrl) {
  return backendUrl.replace(/\/+$/, "") + "/healthz";
}

// Full backoff with jitter: cap the exponential curve, then randomize within
// [half, full] so a fleet of tabs doesn't reconnect in lockstep.
export function reconnectDelay(attempt, random = Math.random()) {
  const capped = Math.min(RECONNECT_CAP_MS, RECONNECT_BASE_MS * 2 ** attempt);
  return capped / 2 + random * (capped / 2);
}

export function useArchitectureSocket(backendUrl = BACKEND_URL) {
  const [arch, setArch] = useState({ nodes: [], edges: [], layout: {} });
  const [status, setStatus] = useState("connecting"); // connected|connecting|waking
  const [notice, setNotice] = useState(null); // {kind:"error"|"ack", message}
  const wsRef = useRef(null);
  const closedRef = useRef(false);
  const attemptRef = useRef(0);
  const pendingCodeRef = useRef(null); // resolver for an in-flight requestCode()

  useEffect(() => {
    closedRef.current = false;
    let reconnectTimer = null;

    // The ws just failed — ask /healthz whether anyone is home. Reachable
    // backend => plain "connecting"; unreachable => "waking" (sleeping Space
    // spinning up, or nothing there — either way: wait, don't panic).
    async function probeHealth() {
      try {
        const res = await fetch(healthzUrl(backendUrl), { cache: "no-store" });
        if (!closedRef.current) setStatus(res.ok ? "connecting" : "waking");
      } catch {
        if (!closedRef.current) setStatus("waking");
      }
    }

    function connect() {
      const ws = new WebSocket(wsUrl(backendUrl));
      wsRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0;
        setStatus("connected");
      };
      ws.onclose = () => {
        if (closedRef.current) return;
        setStatus("connecting");
        probeHealth();
        reconnectTimer = setTimeout(connect, reconnectDelay(attemptRef.current));
        attemptRef.current += 1;
      };
      ws.onmessage = (ev) => {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.type === "state") {
          setArch(msg.data);
        } else if (msg.type === "code") {
          // Reply to requestCode(): resolve the in-flight promise (if any).
          pendingCodeRef.current?.(msg);
          pendingCodeRef.current = null;
        } else if (msg.type === "error") {
          setNotice({ kind: "error", message: msg.message });
        } else if (msg.type === "ack") {
          setNotice(null);
        }
      };
    }

    connect();
    return () => {
      closedRef.current = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [backendUrl]);

  const send = useCallback((message) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    } else {
      setNotice({ kind: "error", message: "not connected to backend" });
    }
  }, []);

  // Ask the backend for the generated PyTorch source; resolves with the `code`
  // reply ({code} or {error}). Only one request is tracked at a time, which is
  // all the single "Export code" button needs.
  const requestCode = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.resolve({ error: "not connected to backend" });
    }
    return new Promise((resolve) => {
      pendingCodeRef.current = resolve;
      ws.send(JSON.stringify(proto.generateCode()));
    });
  }, []);

  return {
    arch,
    status,
    connected: status === "connected",
    notice,
    send,
    requestCode,
    clearNotice: () => setNotice(null),
  };
}
