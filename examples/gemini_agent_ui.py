"""
Web UI demo for Gemini + VektoriAgent (single file).

Why this version:
- Single request per turn (no duplicate retrieval call)
- Proof of memory: explicit retrieved fact/episode/sentence list with scores
- Latency visibility: chat latency + evidence-render latency are shown in UI

Run:
    export GOOGLE_API_KEY="your-gemini-api-key"
    uv pip install -e ".[litellm,sentence-transformers]" google-generativeai aiosqlite
    python3 examples/gemini_agent_ui.py

Open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

# Allow direct execution from source checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vektori import AgentConfig, Vektori, VektoriAgent
from vektori.models.factory import create_chat_model

HOST = os.getenv("VEKTORI_UI_HOST", "127.0.0.1")
PORT = int(os.getenv("VEKTORI_UI_PORT", "8765"))
USER_ID = os.getenv("VEKTORI_DEMO_USER_ID", "demo-gemini-user")
AGENT_ID = os.getenv("VEKTORI_DEMO_AGENT_ID", "gemini-web-ui-agent")
SESSION_ID = os.getenv("VEKTORI_DEMO_SESSION_ID", f"gemini-web-ui-{uuid4()}")

# Faster default than full flash model; can override via env.
CHAT_MODEL = os.getenv("VEKTORI_CHAT_MODEL", "litellm:gemini/gemini-2.5-flash-lite")
EXTRACTION_MODEL = os.getenv("VEKTORI_EXTRACT_MODEL", "gemini:gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("VEKTORI_EMBED_MODEL", "sentence-transformers:all-MiniLM-L6-v2")


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Vektori Gemini Agent UI</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #121a33;
      --panel-2: #0f1730;
      --text: #e7ecff;
      --muted: #a8b3d6;
      --good: #7ee787;
      --warn: #ffd580;
      --border: #2a396b;
    }
    * { box-sizing: border-box; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    .wrap { max-width: 1320px; margin: 0 auto; padding: 18px; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    .sub { margin: 0 0 14px; color: var(--muted); }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }
    .panel {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 14px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      letter-spacing: .03em;
      text-transform: uppercase;
    }
    .chat-box { height: 58vh; overflow: auto; padding: 14px; }
    .msg { margin-bottom: 12px; padding: 10px 12px; border-radius: 10px; white-space: pre-wrap; }
    .user { background: #1c2b57; border: 1px solid #3356b0; }
    .assistant { background: #132840; border: 1px solid #23588f; }
    .meta { margin-top: 6px; font-size: 12px; color: var(--muted); }
    .input-row {
      display: flex; gap: 10px; padding: 12px; border-top: 1px solid var(--border);
      background: var(--panel-2);
    }
    input, button {
      border-radius: 10px; border: 1px solid var(--border); background: #0e1630; color: var(--text);
      padding: 10px 12px; font-size: 14px;
    }
    input { flex: 1; }
    button { cursor: pointer; background: #223b7a; }
    button:hover { background: #2a4a98; }
    .trace { padding: 12px; font-size: 14px; line-height: 1.45; }
    .k { color: var(--muted); }
    .v { color: var(--good); }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #24386f; }
    .warn { color: var(--warn); }
    .evidence { border-top: 1px solid var(--border); padding: 10px 12px; height: 24vh; overflow: auto; }
    .ev-item { margin-bottom: 8px; font-size: 13px; }
    .ev-title { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .03em; margin: 8px 0 4px; }
    .log { height: 16vh; overflow: auto; border-top: 1px solid var(--border); padding: 10px 12px; }
    .log-entry { margin-bottom: 8px; font-size: 12px; color: #d5e0ff; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .chat-box { height: 45vh; }
      .evidence { height: 22vh; }
      .log { height: 14vh; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Vektori Gemini Agent UI</h1>
    <p class="sub">Fast chat + explicit retrieval evidence so memory claims are verifiable.</p>
    <div class="grid">
      <section class="panel">
        <h2>Agent Chat</h2>
        <div id="chat" class="chat-box"></div>
        <div class="input-row">
          <input id="prompt" placeholder="Try: what do you remember about my communication style?" />
          <button id="send">Send</button>
          <button id="reset">Reset</button>
        </div>
      </section>

      <section class="panel">
        <h2>Memory Proof + Latency</h2>
        <div id="trace" class="trace">
          <div><span class="k">status:</span> <span id="status" class="v">ready</span></div>
          <div><span class="k">retrieval_reason:</span> <span id="reason">-</span></div>
          <div><span class="k">facts / episodes / sentences:</span> <span id="counts">-</span></div>
          <div><span class="k">chat_latency:</span> <span id="chatMs" class="badge">-</span></div>
          <div><span class="k">retrieval_latency:</span> <span id="retrievalMs" class="badge">-</span></div>
          <div><span class="k">model_latency:</span> <span id="modelMs" class="badge">-</span> <span id="modelCalls" class="k"></span></div>
          <div><span class="k">other_overhead:</span> <span id="overheadMs" class="badge">-</span></div>
          <div><span class="k">evidence_render_latency:</span> <span id="evidenceMs" class="badge">-</span></div>
          <div class="warn">Proof = retrieved evidence below (not just generated text).</div>
        </div>
        <div id="evidence" class="evidence"></div>
        <div id="log" class="log"></div>
      </section>
    </div>
  </div>

  <script>
    const chat = document.getElementById("chat");
    const prompt = document.getElementById("prompt");
    const sendBtn = document.getElementById("send");
    const resetBtn = document.getElementById("reset");
    const reasonEl = document.getElementById("reason");
    const countsEl = document.getElementById("counts");
    const chatMsEl = document.getElementById("chatMs");
    const retrievalMsEl = document.getElementById("retrievalMs");
    const modelMsEl = document.getElementById("modelMs");
    const modelCallsEl = document.getElementById("modelCalls");
    const overheadMsEl = document.getElementById("overheadMs");
    const evidenceMsEl = document.getElementById("evidenceMs");
    const statusEl = document.getElementById("status");
    const evidenceEl = document.getElementById("evidence");
    const logEl = document.getElementById("log");

    function addMsg(role, text, meta) {
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.textContent = text;
      if (meta) {
        const m = document.createElement("div");
        m.className = "meta";
        m.textContent = meta;
        div.appendChild(m);
      }
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    function addLog(line) {
      const div = document.createElement("div");
      div.className = "log-entry";
      div.textContent = line;
      logEl.prepend(div);
    }

    function esc(text) {
      return String(text || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function renderEvidenceList(label, items) {
      if (!items || !items.length) return `<div class="ev-title">${label}</div><div class="ev-item">-</div>`;
      const rows = items.map((it) => {
        const score = it.score !== null && it.score !== undefined ? ` score=${Number(it.score).toFixed(3)}` : "";
        const sid = it.session_id ? ` session=${it.session_id}` : "";
        return `<div class="ev-item">• ${esc(it.text)}<div class="meta">${esc(score + sid).trim() || "-"}</div></div>`;
      }).join("");
      return `<div class="ev-title">${label}</div>${rows}`;
    }

    function updateTrace(payload) {
      const d = payload.retrieval_debug || {};
      const c = (d.counts || {});
      reasonEl.textContent = d.reason || "-";
      countsEl.textContent = `${c.facts || 0} / ${c.episodes || 0} / ${c.sentences || 0}`;
      chatMsEl.textContent = payload.chat_latency_ms ? `${payload.chat_latency_ms}ms` : "-";
      retrievalMsEl.textContent = payload.retrieval_latency_ms ? `${payload.retrieval_latency_ms}ms` : "-";
      modelMsEl.textContent = payload.model_latency_ms ? `${payload.model_latency_ms}ms` : "-";
      modelCallsEl.textContent = payload.model_calls ? `(${payload.model_calls} call${payload.model_calls === 1 ? "" : "s"})` : "";
      overheadMsEl.textContent = payload.other_overhead_ms ? `${payload.other_overhead_ms}ms` : "-";
      statusEl.textContent = payload.status || "ok";
    }

    function updateEvidence(payload) {
      evidenceMsEl.textContent = payload.evidence_latency_ms ? `${payload.evidence_latency_ms}ms` : "-";
      evidenceEl.innerHTML =
        renderEvidenceList("facts", payload.facts || []) +
        renderEvidenceList("episodes", payload.episodes || []) +
        renderEvidenceList("sentences", payload.sentences || []);
    }

    async function send() {
      const text = prompt.value.trim();
      if (!text) return;
      prompt.value = "";
      addMsg("user", text);
      statusEl.textContent = "answering...";
      evidenceMsEl.textContent = "-";
      evidenceEl.innerHTML = "<div class='ev-item'>loading evidence...</div>";
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "request failed");
        addMsg("assistant", data.content || "(empty)", `chat ${data.chat_latency_ms || "-"}ms`);
        updateTrace(data);
        updateEvidence(data);
        addLog(
          `chat=${data.chat_latency_ms}ms retrieval=${data.retrieval_latency_ms || "-"}ms ` +
          `model=${data.model_latency_ms || "-"}ms calls=${data.model_calls || 0} ` +
          `reason=${data.retrieval_debug.reason} ` +
          `facts=${data.retrieval_debug.counts.facts} episodes=${data.retrieval_debug.counts.episodes} ` +
          `sentences=${data.retrieval_debug.counts.sentences}`
        );
        statusEl.textContent = "ready";
      } catch (err) {
        addMsg("assistant", `Error: ${err.message}`);
        statusEl.textContent = "request failed";
      }
    }

    async function reset() {
      const res = await fetch("/api/reset", { method: "POST" });
      const data = await res.json();
      chat.innerHTML = "";
      logEl.innerHTML = "";
      reasonEl.textContent = "-";
      countsEl.textContent = "-";
      chatMsEl.textContent = "-";
      retrievalMsEl.textContent = "-";
      modelMsEl.textContent = "-";
      modelCallsEl.textContent = "";
      overheadMsEl.textContent = "-";
      evidenceMsEl.textContent = "-";
      evidenceEl.innerHTML = "<div class='ev-item'>-</div>";
      statusEl.textContent = data.status || "reset";
      addLog("conversation reset (stored memory unchanged)");
    }

    sendBtn.addEventListener("click", send);
    resetBtn.addEventListener("click", reset);
    prompt.addEventListener("keydown", (e) => {
      if (e.key === "Enter") send();
    });
  </script>
</body>
</html>
"""


def _to_evidence_item(item: dict, prefer_score: bool = True) -> dict:
    score = None
    if prefer_score and "score" in item:
        score = item.get("score")
    elif "distance" in item:
        # distance lower is better; expose as score-like signal for readability
        try:
            score = 1 - float(item.get("distance", 1.0))
        except Exception:
            score = None
    return {
        "text": item.get("text", ""),
        "session_id": item.get("session_id"),
        "score": score,
    }


class Runtime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.memory: Vektori | None = None
        self.agent: VektoriAgent | None = None
        self._last_retrieval_ms: int | None = None
        self._model_ms_accum: int = 0
        self._model_calls: int = 0
        self.run(self._init())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    async def _seed_if_needed(self) -> None:
        assert self.memory is not None
        existing = await self.memory.search(
            query="support teams handling many follow-ups",
            user_id=USER_ID,
            agent_id=AGENT_ID,
            depth="l0",
            top_k=1,
        )
        if existing.get("facts"):
            return

        await self.memory.add(
            messages=[
                {"role": "user", "content": "I prefer concise responses with bullet points."},
                {"role": "assistant", "content": "Understood. I will keep responses concise."},
                {
                    "role": "user",
                    "content": "Our demo target is support teams handling many follow-ups.",
                },
            ],
            session_id="seed-web-ui-001",
            user_id=USER_ID,
            agent_id=AGENT_ID,
        )
        await self.memory.add(
            messages=[
                {"role": "user", "content": "Avoid long narratives unless I explicitly ask."},
                {"role": "assistant", "content": "Got it. I will stay brief and action-oriented."},
            ],
            session_id="seed-web-ui-002",
            user_id=USER_ID,
            agent_id=AGENT_ID,
        )

    async def _init(self) -> None:
        self.memory = Vektori(
            embedding_model=EMBEDDING_MODEL,
            extraction_model=EXTRACTION_MODEL,
            async_extraction=False,
        )
        chat_model = create_chat_model(CHAT_MODEL)
        self.agent = VektoriAgent(
            memory=self.memory,
            model=chat_model,
            user_id=USER_ID,
            agent_id=AGENT_ID,
            session_id=SESSION_ID,
            config=AgentConfig(
                retrieve_on_every_turn=True,
                background_add=True,  # faster turn latency (writes async in background)
                retrieval_depth="l1",
                retrieval_top_k=3,
                reserve_response_tokens=220,
                max_context_tokens=6000,
            ),
        )
        self._instrument_timings()
        await self._seed_if_needed()

    def _instrument_timings(self) -> None:
        assert self.memory is not None
        assert self.agent is not None

        original_search = self.memory.search

        async def timed_search(*args, **kwargs):
            started = time.perf_counter()
            out = await original_search(*args, **kwargs)
            self._last_retrieval_ms = int((time.perf_counter() - started) * 1000)
            return out

        self.memory.search = timed_search  # type: ignore[assignment]

        original_complete = self.agent.model.complete

        async def timed_complete(*args, **kwargs):
            started = time.perf_counter()
            out = await original_complete(*args, **kwargs)
            self._model_ms_accum += int((time.perf_counter() - started) * 1000)
            self._model_calls += 1
            return out

        self.agent.model.complete = timed_complete  # type: ignore[assignment]

    async def chat(self, message: str) -> dict:
        assert self.agent is not None
        self._last_retrieval_ms = None
        self._model_ms_accum = 0
        self._model_calls = 0
        started = time.perf_counter()
        result = await self.agent.chat(message)
        chat_elapsed_ms = int((time.perf_counter() - started) * 1000)
        evidence_started = time.perf_counter()
        facts = [_to_evidence_item(x) for x in result.memories_used.get("facts", [])][:3]
        episodes = [_to_evidence_item(x) for x in result.memories_used.get("episodes", [])][:3]
        sentences = [
            _to_evidence_item(x, prefer_score=False) for x in result.memories_used.get("sentences", [])
        ][:3]
        evidence_elapsed_ms = int((time.perf_counter() - evidence_started) * 1000)
        accounted = (self._last_retrieval_ms or 0) + self._model_ms_accum
        other_overhead_ms = max(0, chat_elapsed_ms - accounted)
        return {
            "content": result.content,
            "retrieval_debug": result.retrieval_debug,
            "chat_latency_ms": chat_elapsed_ms,
            "retrieval_latency_ms": self._last_retrieval_ms,
            "model_latency_ms": self._model_ms_accum,
            "model_calls": self._model_calls,
            "other_overhead_ms": other_overhead_ms,
            "facts": facts,
            "episodes": episodes,
            "sentences": sentences,
            "evidence_latency_ms": evidence_elapsed_ms,
            "status": "ok",
        }

    async def reset(self) -> dict:
        assert self.agent is not None
        self.agent.reset_window()
        return {"status": "conversation reset"}

    async def close(self) -> None:
        if self.agent is not None:
            await self.agent.close()
        if self.memory is not None:
            await self.memory.close()


runtime: Runtime | None = None


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._html(HTML)
            return
        if self.path == "/api/health":
            self._json({"status": "ok"})
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        global runtime
        if runtime is None:
            self._json({"error": "runtime not initialized"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        try:
            if self.path == "/api/chat":
                data = self._read_json()
                message = str(data.get("message", "")).strip()
                if not message:
                    self._json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
                    return
                self._json(runtime.run(runtime.chat(message)))
                return

            if self.path == "/api/reset":
                self._json(runtime.run(runtime.reset()))
                return

            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as e:
            self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    global runtime
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set. Export it and re-run.")
    runtime = Runtime()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Vektori Gemini Web UI running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if runtime is not None:
            runtime.run(runtime.close())
            runtime.loop.call_soon_threadsafe(runtime.loop.stop)


if __name__ == "__main__":
    main()
