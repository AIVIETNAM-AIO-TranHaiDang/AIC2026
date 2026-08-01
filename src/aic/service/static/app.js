// Operator console logic: keyboard-first, one global state object.
// Talks only to the /api endpoints; rendering is plain DOM.

"use strict";

const state = {
  results: [],
  selected: -1,
  session: null, // {session_id, constraints, version}
  resultVersion: null,
  lastQuery: "",
  positives: new Set(),
  negatives: new Set(),
  escalations: [], // [{name, timeout_s}] from /api/escalations
};

// Escalation hotkeys; only names the server actually offers are bound.
const ESCALATION_KEYS = { r: "rerank", i: "imagine", v: "vlm_verify" };

const $ = (id) => document.getElementById(id);

function setStatus(text) {
  $("status").textContent = text;
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `${path} failed (${response.status})`);
  }
  return response.json();
}

// Shared NDJSON streaming reader (used by both the KIS search and the VQA ask
// handlers). POSTs `body` to `url` and calls `onEvent({event, ...})` for each
// line as it arrives. Returns true when the stream ran, or false if the route
// is absent (404) so the caller falls back to the one-shot route. Any other
// HTTP/transport error throws. The server registers /stream routes only when
// service.streaming is on, so this degrades cleanly when it is off.
async function streamNdjson(url, body, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (response.status === 404) return false;
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `${url} failed (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const flush = (final) => {
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) onEvent(JSON.parse(line));
    }
    if (final) {
      const tail = buffer.trim();
      if (tail) onEvent(JSON.parse(tail));
    }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    flush(false);
  }
  buffer += decoder.decode();
  flush(true);
  return true;
}

// Tri-state cache: undefined = not yet probed, true/false = server answer. Once
// a /stream route 404s we stop probing and use the one-shot route.
let streamingEnabled;

function fmtTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function renderResults() {
  const container = $("results");
  container.replaceChildren();
  state.results.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = "card";
    if (i === state.selected) card.classList.add("selected");
    if (state.positives.has(r.shot_key)) card.classList.add("fed-positive");
    if (state.negatives.has(r.shot_key)) card.classList.add("fed-negative");

    const rank = document.createElement("div");
    rank.className = "rank";
    rank.textContent = `#${i + 1} · ${r.shot_key} · ${fmtTime(r.timestamp_ms)}`;
    card.appendChild(rank);

    if (r.frames.length) {
      const strip = document.createElement("div");
      strip.className = "strip";
      for (const src of r.frames.slice(0, 4)) {
        const img = document.createElement("img");
        img.src = src;
        img.loading = "lazy";
        img.alt = r.shot_key;
        strip.appendChild(img);
      }
      card.appendChild(strip);
    }

    const caption = document.createElement("div");
    caption.className = "caption";
    caption.textContent = r.caption || "(no caption)";
    card.appendChild(caption);

    const snippetParts = [];
    if (r.ocr_lines.length) snippetParts.push(`OCR: ${r.ocr_lines.join(" | ")}`);
    if (r.asr_text) snippetParts.push(`ASR: ${r.asr_text}`);
    if (snippetParts.length) {
      const snippet = document.createElement("div");
      snippet.className = "snippet";
      snippet.textContent = snippetParts.join("\n");
      card.appendChild(snippet);
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent =
      `${r.video_id} · shot ${r.shot_id} · ` +
      `${fmtTime(r.t_start_ms)}–${fmtTime(r.t_end_ms)}` +
      (r.scene ? ` · ${r.scene}` : "");
    card.appendChild(meta);

    const timeline = document.createElement("div");
    timeline.className = "timeline";
    const pos = document.createElement("div");
    pos.className = "pos";
    const span = Math.max(r.t_end_ms - r.t_start_ms, 1);
    pos.style.left = `${(100 * (r.timestamp_ms - r.t_start_ms)) / span}%`;
    timeline.appendChild(pos);
    card.appendChild(timeline);

    card.addEventListener("click", () => select(i));
    container.appendChild(card);
  });
}

function renderSession() {
  const panel = $("session-info");
  const aside = $("constraints");
  if (!state.session) {
    panel.hidden = true;
    aside.hidden = true;
    return;
  }
  panel.hidden = false;
  $("session-version").textContent = `v${state.session.version}`;
  aside.hidden = state.session.constraints.length === 0;
  const list = $("constraint-list");
  list.replaceChildren();
  for (const c of state.session.constraints) {
    const item = document.createElement("li");
    item.textContent = c;
    list.appendChild(item);
  }
}

function renderAdvisor(advisor) {
  const el = $("advisor");
  if (!advisor) {
    el.hidden = true;
    return;
  }
  const pct = Math.round(advisor.p_hit * 100);
  el.textContent = advisor.hint
    ? `Advisor: ${advisor.hint}`
    : `Advisor: ${pct}% likely in top ${advisor.target_k}.`;
  el.classList.toggle("warn", Boolean(advisor.hint));
  el.hidden = false;
}

function renderEscalations() {
  const el = $("escalations");
  el.replaceChildren();
  if (!state.escalations.length) {
    el.hidden = true;
    return;
  }
  for (const esc of state.escalations) {
    const button = document.createElement("button");
    const key = Object.keys(ESCALATION_KEYS).find(
      (k) => ESCALATION_KEYS[k] === esc.name,
    );
    button.type = "button";
    button.textContent = key ? `${esc.name} (${key})` : esc.name;
    button.addEventListener("click", () => runEscalation(esc.name));
    el.appendChild(button);
  }
  el.hidden = false;
}

function renderPlanner(planner) {
  const el = $("planner");
  if (!planner) {
    el.hidden = true;
    return;
  }
  const counts = Object.entries(planner.counts)
    .map(([value, n]) => `${value}: ${n}`)
    .join(", ");
  el.textContent = `Ask about "${planner.attribute}" (${counts})`;
  el.hidden = false;
}

function applyResponse(data) {
  state.results = data.results;
  state.selected = data.results.length ? 0 : -1;
  if (data.result_version !== null) state.resultVersion = data.result_version;
  if (data.session) state.session = data.session;
  renderResults();
  renderSession();
  renderPlanner(data.planner);
  renderAdvisor(data.advisor ?? null);
}

// Read-only preview of the fused ranking (streaming): show the candidates but
// touch no session/version state — the authoritative `results` event replaces
// them and carries the submittable result_version.
function applyPreview(results) {
  state.results = results;
  state.selected = results.length ? 0 : -1;
  renderResults();
}

function select(i) {
  if (!state.results.length) return;
  state.selected = Math.max(0, Math.min(i, state.results.length - 1));
  renderResults();
  const card = $("results").children[state.selected];
  if (card) card.scrollIntoView({ block: "nearest" });
}

async function runSearch() {
  const query = $("query").value.trim();
  if (!query) return;
  state.lastQuery = query;
  state.positives.clear();
  state.negatives.clear();
  setStatus("Searching…");
  const body = { query, session_id: state.session?.session_id ?? null };
  const finalStatus = (data) =>
    setStatus(
      `${data.results.length} results (spec: ${data.spec ? data.spec.source : "?"}).`
    );
  try {
    if (streamingEnabled !== false) {
      const ran = await streamNdjson("/api/search/stream", body, (evt) => {
        if (evt.event === "results_preview") {
          applyPreview(evt.results);
          setStatus(`${evt.results.length} results (fused — refining…).`);
        } else if (evt.event === "results") {
          applyResponse(evt);
          finalStatus(evt);
        }
      });
      if (ran) {
        streamingEnabled = true;
        return;
      }
      streamingEnabled = false; // route absent — this server has no /stream
    }
    const data = await api("/api/search", body);
    applyResponse(data);
    finalStatus(data);
  } catch (err) {
    setStatus(err.message);
  }
}

async function sendFeedback(shotKey, positive) {
  if (positive) {
    state.positives.add(shotKey);
    state.negatives.delete(shotKey);
  } else {
    state.negatives.add(shotKey);
    state.positives.delete(shotKey);
  }
  setStatus("Re-ranking with feedback…");
  try {
    const data = await api("/api/feedback", {
      query: state.lastQuery,
      positives: [...state.positives],
      negatives: [...state.negatives],
      session_id: state.session?.session_id ?? null,
    });
    applyResponse(data);
    setStatus(`Feedback applied (${data.results.length} results).`);
  } catch (err) {
    setStatus(err.message);
  }
}

async function findSimilar(shotKey) {
  setStatus("Finding similar shots…");
  try {
    const data = await api("/api/similar", { shot_key: shotKey });
    applyResponse(data);
    setStatus(`${data.results.length} shots similar to ${shotKey}.`);
  } catch (err) {
    setStatus(err.message);
  }
}

async function submitSelected() {
  const r = state.results[state.selected];
  if (!r) return;
  try {
    const data = await api("/api/submit", {
      video_id: r.video_id,
      timestamp_ms: r.timestamp_ms,
      session_id: state.session?.session_id ?? null,
      result_version: state.resultVersion,
    });
    $("submit-payload").textContent = data.payload;
    $("submit-dialog").showModal();
    if (navigator.clipboard) {
      navigator.clipboard.writeText(data.payload).catch(() => {});
    }
    setStatus(`Submission ready (${data.format}).`);
  } catch (err) {
    setStatus(err.message);
  }
}

async function runEscalation(name) {
  if (!state.lastQuery) {
    setStatus("Search first, then escalate.");
    return;
  }
  setStatus(`Escalating (${name})…`);
  try {
    const data = await api("/api/escalate", {
      name,
      query: state.lastQuery,
      session_id: state.session?.session_id ?? null,
      result_version: state.resultVersion,
    });
    applyResponse(data);
    const note = data.escalation?.note;
    setStatus(
      note
        ? `${name}: ${note}`
        : `${name} applied (${data.results.length} results).`,
    );
  } catch (err) {
    setStatus(err.message);
  }
}

async function loadEscalations() {
  try {
    const response = await fetch("/api/escalations");
    if (!response.ok) return;
    const data = await response.json();
    state.escalations = data.escalations ?? [];
  } catch {
    state.escalations = [];
  }
  renderEscalations();
}

async function newSession() {
  try {
    state.session = await api("/api/session", {});
    state.resultVersion = null;
    renderSession();
    setStatus("KIS-C session started.");
  } catch (err) {
    setStatus(err.message);
  }
}

async function addReveal() {
  const text = $("reveal-input").value.trim();
  if (!text || !state.session) return;
  try {
    state.session = await api(
      `/api/session/${state.session.session_id}/reveal`,
      { text },
    );
    $("reveal-input").value = "";
    renderSession();
    await runSearch(); // a reveal narrows immediately
  } catch (err) {
    setStatus(err.message);
  }
}

document.addEventListener("keydown", (event) => {
  const inInput = ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);
  if (event.key === "Escape") {
    $("submit-dialog").close();
    document.activeElement.blur();
    return;
  }
  if (inInput) return;
  const selected = state.results[state.selected];
  if (event.key >= "1" && event.key <= "9") {
    select(Number(event.key) - 1);
  } else if (event.key === "ArrowRight") {
    select(state.selected + 1);
  } else if (event.key === "ArrowLeft") {
    select(state.selected - 1);
  } else if (event.key === "f" && selected) {
    sendFeedback(selected.shot_key, true);
  } else if (event.key === "x" && selected) {
    sendFeedback(selected.shot_key, false);
  } else if (event.key === "m" && selected) {
    findSimilar(selected.shot_key);
  } else if (event.key === "s" && selected) {
    submitSelected();
  } else if (event.key in ESCALATION_KEYS) {
    const name = ESCALATION_KEYS[event.key];
    if (state.escalations.some((esc) => esc.name === name)) {
      runEscalation(name);
    }
  } else if (event.key === "/") {
    event.preventDefault();
    $("query").focus();
  }
});

loadEscalations();

$("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch();
  $("query").blur();
});
$("session-new").addEventListener("click", newSession);
$("reveal-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addReveal();
  }
});
$("submit-close").addEventListener("click", () => $("submit-dialog").close());
