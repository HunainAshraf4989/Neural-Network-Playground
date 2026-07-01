// React hook owning the websocket to the backend. Holds the shared architecture
// from `state` broadcasts and exposes `send(message)` for the §12 client
// messages. The backend is the single source of truth, so this hook never
// mutates `arch` optimistically — it only renders what the server broadcasts
// back. `ack`/`error` replies are surfaced as a transient status line.

import { useCallback, useEffect, useRef, useState } from "react";

import * as proto from "./protocol.js";

// Backend websocket URL. Defaults to the local backend; override with the
// VITE_WS_URL env var (e.g. to point the dev UI at a backend on another port).
const DEFAULT_URL = import.meta.env?.VITE_WS_URL || "ws://localhost:8765/ws";
const RECONNECT_MS = 1500;

export function useArchitectureSocket(url = DEFAULT_URL) {
  const [arch, setArch] = useState({ nodes: [], edges: [], layout: {} });
  const [connected, setConnected] = useState(false);
  const [notice, setNotice] = useState(null); // {kind:"error"|"ack", message}
  const wsRef = useRef(null);
  const closedRef = useRef(false);
  const pendingCodeRef = useRef(null); // resolver for an in-flight requestCode()

  useEffect(() => {
    closedRef.current = false;
    let reconnectTimer = null;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closedRef.current) reconnectTimer = setTimeout(connect, RECONNECT_MS);
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
  }, [url]);

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

  return { arch, connected, notice, send, requestCode, clearNotice: () => setNotice(null) };
}
