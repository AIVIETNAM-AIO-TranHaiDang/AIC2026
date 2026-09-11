// Operator console logic: keyboard-first, one global state object.
// Talks only to the /api endpoints; rendering is plain DOM.

"use strict";

const DEFAULT_RESULT_PAGE_SIZE = 20;

const state = {
  results: [],
  visibleResults: DEFAULT_RESULT_PAGE_SIZE,
  selected: -1,
  session: null, // {session_id, constraints, version}
  resultVersion: null,
  lastQuery: "",
  positives: new Set(),
  negatives: new Set(),
  submission: loadSubmission(), // ordered [{video_id, frame_id, ...}], max 100
  selectedFrames: new Map(), // shot_key -> keyframe_id
  escalations: [], // [{name, timeout_s}] from /api/escalations
};

// Escalation hotkeys; only names the server actually offers are bound.
const ESCALATION_KEYS = { r: "rerank", i: "imagine", v: "vlm_verify" };
let draggedSubmissionIndex = null;

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

function loadSubmission() {
  try {
    const entries = JSON.parse(localStorage.getItem("aic-kis-submission") || "[]");
    if (!Array.isArray(entries)) return [];
    return entries.filter((entry) =>
      typeof entry?.video_id === "string" &&
      Number.isInteger(entry.frame_id) && entry.frame_id >= 0,
    ).slice(0, 100);
  } catch {
    return [];
  }
}

function persistSubmission() {
  localStorage.setItem("aic-kis-submission", JSON.stringify(state.submission));
}

function defaultFrame(result) {
  if (!result.frames?.length) return null;
  return result.frames.reduce((best, frame) =>
    Math.abs(frame.timestamp_ms - result.timestamp_ms) <
    Math.abs(best.timestamp_ms - result.timestamp_ms) ? frame : best,
  );
}

function selectedFrame(result) {
  const selectedId = state.selectedFrames.get(result.shot_key);
  return result.frames?.find((frame) => frame.keyframe_id === selectedId) ||
    defaultFrame(result);
}

function submissionKey(entry) {
  return `${entry.video_id}:${entry.frame_id}`;
}

function renderSubmission() {
  const list = $("submission-list");
  const count = $("submission-count");
  const hasEntries = state.submission.length > 0;
  list.replaceChildren();
  count.textContent = `${state.submission.length}/100`;
  $("submission-export-csv").disabled = !hasEntries;
  $("submission-export-json").disabled = !hasEntries;
  $("submission-clear").disabled = !hasEntries;

  state.submission.forEach((entry, index) => {
    const item = document.createElement("li");
    item.draggable = true;
    item.title = "Kéo thả để đổi thứ tự";
    item.addEventListener("dragstart", (event) => {
      draggedSubmissionIndex = index;
      item.classList.add("dragging");
      event.dataTransfer.effectAllowed = "move";
    });
    item.addEventListener("dragend", () => {
      draggedSubmissionIndex = null;
      item.classList.remove("dragging");
      [...list.children].forEach((child) => child.classList.remove("drag-over"));
    });
    item.addEventListener("dragover", (event) => {
      if (draggedSubmissionIndex === null || draggedSubmissionIndex === index) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      item.classList.add("drag-over");
    });
    item.addEventListener("dragleave", () => item.classList.remove("drag-over"));
    item.addEventListener("drop", (event) => {
      event.preventDefault();
      item.classList.remove("drag-over");
      moveSubmissionTo(draggedSubmissionIndex, index);
    });
    const label = document.createElement("span");
    label.textContent = `${entry.video_id},${entry.frame_id}`;
    label.title = `${entry.keyframe_id} · ${fmtTime(entry.timestamp_ms)}`;
    const actions = document.createElement("span");
    actions.className = "submission-item-actions";
    actions.append(
      (() => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "×";
        button.title = "Xóa khỏi danh sách nộp";
        button.addEventListener("click", () => removeSubmission(index));
        return button;
      })(),
    );
    item.append(label, actions);
    list.appendChild(item);
  });
}

function addToSubmission(result) {
  const frame = selectedFrame(result);
  if (!frame) {
    setStatus("Shot này không có keyframe có thể nộp.");
    return;
  }
  const entry = {
    video_id: result.video_id,
    frame_id: frame.frame_id,
    keyframe_id: frame.keyframe_id,
    timestamp_ms: frame.timestamp_ms,
  };
  if (state.submission.some((item) => submissionKey(item) === submissionKey(entry))) {
    setStatus("Frame này đã nằm trong danh sách nộp.");
    return;
  }
  if (state.submission.length >= 100) {
    setStatus("Danh sách nộp đã đủ 100 đáp án.");
    return;
  }
  state.submission.push(entry);
  persistSubmission();
  renderSubmission();
  setStatus(`Đã thêm ${entry.video_id}, frame ${entry.frame_id} vào danh sách nộp.`);
}

function removeSubmission(index) {
  state.submission.splice(index, 1);
  persistSubmission();
  renderSubmission();
}

function moveSubmissionTo(source, target) {
  if (source === null || source === target || source < 0 || target < 0 ||
      source >= state.submission.length || target >= state.submission.length) return;
  const [entry] = state.submission.splice(source, 1);
  // Dropping on a row places the dragged row immediately before it.
  state.submission.splice(source < target ? target - 1 : target, 0, entry);
  persistSubmission();
  renderSubmission();
}

function clearSubmission() {
  if (!state.submission.length || !window.confirm("Xóa toàn bộ danh sách nộp?")) return;
  state.submission = [];
  persistSubmission();
  renderSubmission();
  setStatus("Đã xóa danh sách nộp.");
}

function downloadSubmission(kind) {
  if (!state.submission.length) return;
  const isCsv = kind === "csv";
  const content = isCsv
    ? state.submission.map((entry) => `${entry.video_id},${entry.frame_id}`).join("\n") + "\n"
    : JSON.stringify(state.submission.map(({ video_id, frame_id }) =>
      ({ video_id, frame_id })), null, 2) + "\n";
  const blob = new Blob([content], { type: isCsv ? "text/csv" : "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `aic-kis-submission.${kind}`;
  link.click();
  URL.revokeObjectURL(url);
  setStatus(`Đã xuất ${state.submission.length} đáp án KIS dạng ${kind.toUpperCase()}.`);
}

function renderResults() {
  const container = $("results");
  container.replaceChildren();
  state.results.slice(0, state.visibleResults).forEach((r, i) => {
    const activeFrame = selectedFrame(r);
    const activeTimestamp = activeFrame?.timestamp_ms ?? r.timestamp_ms;
    const card = document.createElement("article");
    card.className = "card";
    card.tabIndex = 0;
    if (i === state.selected) card.classList.add("selected");
    if (state.positives.has(r.shot_key)) card.classList.add("fed-positive");
    if (state.negatives.has(r.shot_key)) card.classList.add("fed-negative");

    const cardTop = document.createElement("div");
    cardTop.className = "card-top";
    const rank = document.createElement("div");
    rank.className = "rank";
    rank.textContent = `#${i + 1} · ${r.shot_key}`;
    cardTop.appendChild(rank);
    if (Number.isFinite(Number(r.score))) {
      const score = document.createElement("span");
      score.className = "score";
      score.textContent = `score ${Number(r.score).toFixed(3)}`;
      cardTop.appendChild(score);
    }
    card.appendChild(cardTop);

    const location = document.createElement("div");
    location.className = "location";
    location.textContent = activeFrame
      ? `${r.video_id} · frame ${activeFrame.frame_id} · ${fmtTime(activeTimestamp)}`
      : `${r.video_id} · ${fmtTime(activeTimestamp)}`;
    card.appendChild(location);

    if (Array.isArray(r.frames) && r.frames.length) {
      const strip = document.createElement("div");
      strip.className = "strip";
      for (const frame of r.frames.slice(0, 4)) {
        const img = document.createElement("img");
        img.src = frame.url;
        img.loading = "lazy";
        img.decoding = "async";
        img.fetchPriority = "low";
        img.alt = `${r.video_id}, frame ${frame.frame_id}`;
        img.title = `Chọn frame ${frame.frame_id} (${fmtTime(frame.timestamp_ms)})`;
        img.classList.toggle("frame-selected", frame.keyframe_id === activeFrame?.keyframe_id);
        img.addEventListener("click", (event) => {
          event.stopPropagation();
          state.selectedFrames.set(r.shot_key, frame.keyframe_id);
          select(i);
          renderResults();
        });
        strip.appendChild(img);
      }
      card.appendChild(strip);
    }

    const caption = document.createElement("div");
    caption.className = "caption";
    caption.textContent = r.caption || "Chưa có caption cho shot này.";
    if (!r.caption) caption.classList.add("empty");
    card.appendChild(caption);

    const snippetParts = [];
    if (r.ocr_lines?.length) snippetParts.push(`OCR: ${r.ocr_lines.join(" | ")}`);
    if (r.asr_text) snippetParts.push(`ASR: ${r.asr_text}`);
    if (snippetParts.length) {
      const evidence = document.createElement("details");
      evidence.className = "evidence";
      const summary = document.createElement("summary");
      summary.textContent = "Xem bằng chứng OCR / ASR";
      const snippet = document.createElement("div");
      snippet.className = "snippet";
      snippet.textContent = snippetParts.join("\n");
      evidence.append(summary, snippet);
      card.appendChild(evidence);
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
    pos.style.left = `${(100 * (activeTimestamp - r.t_start_ms)) / span}%`;
    timeline.appendChild(pos);
    card.appendChild(timeline);

    card.addEventListener("click", () => select(i));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select(i);
      }
    });

    const actions = document.createElement("div");
    actions.className = "card-actions";
    const makeAction = (label, title, callback, className = "") => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.title = title;
      if (className) button.className = className;
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        select(i);
        callback();
      });
      return button;
    };
    actions.append(
      makeAction("Chọn", "Đánh dấu kết quả đang xem", () => select(i)),
      makeAction("Phù hợp", "Phản hồi: kết quả phù hợp (F)", () =>
        sendFeedback(r.shot_key, true), "positive"),
      makeAction("Không đúng", "Phản hồi: kết quả không phù hợp (X)", () =>
        sendFeedback(r.shot_key, false), "negative"),
      makeAction(
        "Thêm để nộp",
        "Thêm keyframe đang chọn vào danh sách nộp KIS (S)",
        () => addToSubmission(r),
        "submit",
      ),
    );
    card.appendChild(actions);
    container.appendChild(card);
  });
  renderResultControls();
}

function resultPageSize() {
  return Number($("result-page-size")?.value) || DEFAULT_RESULT_PAGE_SIZE;
}

function resetVisibleResults() {
  state.visibleResults = resultPageSize();
  if (state.selected >= state.visibleResults) {
    state.selected = state.results.length ? 0 : -1;
  }
}

function renderResultControls() {
  const controls = $("result-controls");
  const count = $("result-count");
  const more = $("results-more");
  if (!controls || !count || !more) return;

  const shown = Math.min(state.visibleResults, state.results.length);
  controls.hidden = state.results.length === 0;
  count.textContent = `${shown}/${state.results.length}`;
  more.hidden = shown >= state.results.length;
  if (!more.hidden) {
    const increment = Math.min(resultPageSize(), state.results.length - shown);
    more.textContent = `Hiển thị thêm ${increment}`;
  }
}

function renderSession() {
  const panel = $("session-info");
  const aside = $("constraints");
  const sessionButton = $("session-new");
  if (!state.session) {
    panel.hidden = true;
    aside.hidden = true;
    sessionButton.textContent = "Bắt đầu KIS-C";
    return;
  }
  panel.hidden = false;
  sessionButton.textContent = "Tạo phiên KIS-C mới";
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
  resetVisibleResults();
  if (data.result_version !== null) state.resultVersion = data.result_version;
  if (data.session) state.session = data.session;
  renderResults();
  renderSession();
  renderPlanner(data.planner);
  renderAdvisor(data.advisor ?? null);
  renderSubmission();
}

// Read-only preview of the fused ranking (streaming): show the candidates but
// touch no session/version state — the authoritative `results` event replaces
// them and carries the submittable result_version.
function applyPreview(results) {
  state.results = results;
  state.selected = results.length ? 0 : -1;
  resetVisibleResults();
  renderResults();
}

function select(i) {
  if (!state.results.length) return;
  const previous = state.selected;
  state.selected = Math.max(0, Math.min(i, state.results.length - 1));
  if (state.selected >= state.visibleResults) {
    state.visibleResults = state.selected + 1;
    renderResults();
  } else {
    const cards = $("results").children;
    if (cards[previous]) cards[previous].classList.remove("selected");
    if (cards[state.selected]) cards[state.selected].classList.add("selected");
  }
  const card = $("results").children[state.selected];
  if (card) card.scrollIntoView({ block: "nearest" });
}

async function runSearch() {
  const query = $("query").value.trim();
  if (!query) return;
  if (state.lastQuery && query !== state.lastQuery && state.submission.length &&
      !window.confirm("Đây là query mới. Xóa danh sách nộp của query trước?")) {
    return;
  }
  if (state.lastQuery && query !== state.lastQuery && state.submission.length) {
    state.submission = [];
    persistSubmission();
    renderSubmission();
  }
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

function submitSelected() {
  const r = state.results[state.selected];
  if (!r) return;
  addToSubmission(r);
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
renderSubmission();

$("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch();
  $("query").blur();
});
$("query-clear").addEventListener("click", () => {
  $("query").value = "";
  $("query").focus();
  setStatus("Đã xoá truy vấn.");
});
$("session-new").addEventListener("click", newSession);
$("reveal-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addReveal();
  }
});
$("submission-export-csv").addEventListener("click", () => downloadSubmission("csv"));
$("submission-export-json").addEventListener("click", () => downloadSubmission("json"));
$("submission-clear").addEventListener("click", clearSubmission);
$("submit-close").addEventListener("click", () => $("submit-dialog").close());
$("submit-copy").addEventListener("click", async () => {
  const text = $("submit-payload").textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    $("submit-note").textContent = "Đã sao chép payload vào clipboard.";
  } catch {
    $("submit-note").textContent = "Không thể tự sao chép; hãy chọn và copy nội dung bên dưới.";
  }
});
$("result-page-size").addEventListener("change", () => {
  resetVisibleResults();
  renderResults();
});
$("results-more").addEventListener("click", () => {
  state.visibleResults = Math.min(
    state.results.length,
    state.visibleResults + resultPageSize(),
  );
  renderResults();
});
