(() => {
  const $ = (id) => document.getElementById(id);
  const PREFS_KEY = "adamlm.prefs.v1";
  const CATEGORY_LABELS = { "production": "Production", "general-pretraining": "General pretraining", "sft": "Instruction-tuned (SFT)", "smoke-test": "Smoke tests" };
  const CATEGORY_ORDER = ["production", "general-pretraining", "sft", "smoke-test"];
  const state = {
    runs: [], checkpoints: [], byName: {},
    selectedRun: null, activeRun: null, activeSession: null,
    metricsFor: null, followActive: true,
    stopping: false, starting: false, stopTimer: null,
    auto: { status: "none" }, assistantRun: "sft-assistant", autoStarting: false,
    assistant: { default: null, head: null, lineage: [] },
    dismissedHeadPath: null, lastSeed: null,
  };
  const number = (value) => value == null || Number.isNaN(Number(value)) ? "Unavailable" : new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(Number(value));
  const integer = (value) => value == null || Number.isNaN(Number(value)) ? "Unavailable" : new Intl.NumberFormat("en-US").format(Number(value));
  const orUnavailable = (value) => (value == null || value === "" || Number.isNaN(value)) ? "Unavailable" : value;
  const duration = (seconds) => { if (seconds == null || !Number.isFinite(Number(seconds))) return "Unavailable"; const s = Math.max(0, Math.round(Number(seconds))); if (s < 60) return `${s}s`; const m = Math.floor(s / 60), h = Math.floor(m / 60); return h ? `${h}h ${String(m % 60).padStart(2, "0")}m` : `${m}m ${String(s % 60).padStart(2, "0")}s`; };
  const fmtStep = (v) => v == null ? "—" : `step ${integer(v)}`;
  const fmtClock = (epoch) => { if (epoch == null || !Number.isFinite(Number(epoch))) return "Unavailable"; try { return new Date(Number(epoch) * 1000).toLocaleString(); } catch { return "Unavailable"; } };
  const html = (v) => String(v ?? "—").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const json = async (url, options) => { const response = await fetch(url, options); const body = await response.json(); if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`); return body; };
  const post = (url, body) => json(url, { method: "POST", headers: { "Content-Type": "application/json", "X-Requested-With": "AdamLM" }, body: JSON.stringify(body) });

  // Human-friendly state names. Every label maps 1:1 to a real backend
  // display_state — no synthetic states are introduced.
  const STATE_LABELS = {
    running: "RUNNING", saving: "SAVING CHECKPOINT", initializing: "STARTING",
    verifying_data: "VERIFYING DATA", stopping: "STOPPING",
    stopped: "STOPPED", session_limit: "SESSION COMPLETE", interrupted: "INTERRUPTED",
    disk_pause: "PAUSED · LOW DISK", target_reached: "COMPLETED", error: "FAILED",
    idle: "IDLE", unknown: "IDLE",
  };
  // Short, honest labels for both the current API (display_state) and older
  // backends that report raw status.json states (e.g. "data_stop: ...").
  const stateLabel = (s) => {
    const k = String(s || "idle");
    if (k.startsWith("data_stop")) return "DATA STOP";
    return STATE_LABELS[k] || k.replaceAll("_", " ").toUpperCase();
  };
  const compactState = (run) => stateLabel(run?.display_state || run?.state || "idle");
  // Synthesize the prominent session view from a bare active run when the
  // backend predates the dedicated active_session payload. Every field is
  // either present on the run or explicitly null (rendered as unavailable);
  // remaining tokens are plain arithmetic on real counters, never invented.
  function sessionFromRun(run) {
    const target = run.target_tokens ?? null, tokens = run.tokens ?? 0;
    return {
      run_name: run.name, directory: run.directory,
      display_state: run.display_state || run.state || "running",
      stage: run.stage || "pretrain", dataset: run.dataset || "unknown",
      mixture: run.mixture || null, parent_checkpoint: run.parent_checkpoint || null,
      step: run.step ?? null, tokens, target_tokens: target,
      remaining_tokens: target != null ? Math.max(0, target - tokens) : null,
      progress: run.progress ?? 0, elapsed_seconds: null, session_max_seconds: null,
      session_remaining_seconds: null, live_throughput: run.live_throughput ?? null,
      throughput: run.throughput ?? null, eta: run.eta ?? null,
      train_loss: run.train_loss ?? null, validation_loss: run.validation_loss ?? null,
      learning_rate: run.learning_rate ?? null, checkpoint: run.checkpoint || null,
      checkpoint_step: run.checkpoint_step ?? null,
      checkpoint_exists: run.checkpoint_exists ?? null, checkpoint_time: null,
      updated_at: run.updated_at || null, stop_pending: run.stop_pending || false,
    };
  }

  function loadPrefs() { try { return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}"); } catch { return {}; } }
  function savePrefs() {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify({
        runName: $("runName").value, mode: $("mode").value, dataset: $("dataset").value,
        additionalTokens: $("additionalTokens").value, duration: $("duration").value,
        durationUnit: $("durationUnit").value, untilStopped: $("untilStopped").checked,
        mixture: $("mixture").value, playCheckpoint: $("playCheckpoint").value,
        playMode: $("playMode").value, temperature: $("temperature").value,
        topK: $("topK").value, generationTokens: $("generationTokens").value,
        seedRandom: $("seedRandom") ? $("seedRandom").checked : true,
        seedValue: $("seedValue") ? $("seedValue").value : "42",
        chatTokens: $("chatTokens").value, chatTemperature: $("chatTemperature").value,
        chatTopK: $("chatTopK").value,
        chatSeedRandom: $("chatSeedRandom") ? $("chatSeedRandom").checked : true,
        chatSeedValue: $("chatSeedValue") ? $("chatSeedValue").value : "42",
        autoTokens: $("autoTokens").value, autoDuration: $("autoDuration").value,
        autoDurationUnit: $("autoDurationUnit").value,
        autoPassPolicy: $("autoPassPolicy").value, autoStageCount: $("autoStageCount").value,
      }));
    } catch { /* private mode: dashboard still works, preferences just don't persist */ }
  }
  function applyPrefs(prefs) {
    if (prefs.mode) $("mode").value = prefs.mode;
    if (prefs.dataset) $("dataset").value = prefs.dataset;
    if (prefs.additionalTokens) $("additionalTokens").value = prefs.additionalTokens;
    if (prefs.duration) $("duration").value = prefs.duration;
    if (prefs.durationUnit) $("durationUnit").value = prefs.durationUnit;
    if (prefs.untilStopped != null) $("untilStopped").checked = !!prefs.untilStopped;
    if (prefs.mixture) $("mixture").value = prefs.mixture;
    if (prefs.playMode) $("playMode").value = prefs.playMode;
    if (prefs.temperature) $("temperature").value = prefs.temperature;
    if (prefs.topK) $("topK").value = prefs.topK;
    if (prefs.generationTokens) $("generationTokens").value = prefs.generationTokens;
    if ($("seedRandom") && prefs.seedRandom != null) $("seedRandom").checked = !!prefs.seedRandom;
    if ($("seedValue") && prefs.seedValue) $("seedValue").value = prefs.seedValue;
    if ($("chatTokens") && prefs.chatTokens) $("chatTokens").value = prefs.chatTokens;
    if ($("chatTemperature") && prefs.chatTemperature) $("chatTemperature").value = prefs.chatTemperature;
    if ($("chatTopK") && prefs.chatTopK) $("chatTopK").value = prefs.chatTopK;
    if ($("chatSeedRandom") && prefs.chatSeedRandom != null) $("chatSeedRandom").checked = !!prefs.chatSeedRandom;
    if ($("chatSeedValue") && prefs.chatSeedValue) $("chatSeedValue").value = prefs.chatSeedValue;
    if (prefs.autoTokens) $("autoTokens").value = prefs.autoTokens;
    if (prefs.autoDuration) $("autoDuration").value = prefs.autoDuration;
    if (prefs.autoDurationUnit) $("autoDurationUnit").value = prefs.autoDurationUnit;
    if (prefs.autoPassPolicy) $("autoPassPolicy").value = prefs.autoPassPolicy;
    if (prefs.autoStageCount) $("autoStageCount").value = prefs.autoStageCount;
  }

  const payload = () => ({ mode: $("mode").value, run_name: $("runName").value, dataset: $("dataset").value, parent_checkpoint: $("parentCheckpoint").value, additional_tokens: $("additionalTokens").value, duration: $("duration").value, duration_unit: $("durationUnit").value, until_stopped: $("untilStopped").checked, mixture: $("mixture").value, stage_name: $("stageName").value });
  const selectedRun = () => state.byName[$("runName").value];
  const runOptionLabel = (run) => `${run.is_production ? "PRODUCTION · " : run.category === "smoke-test" ? "SMOKE · " : ""}${run.name} · ${fmtStep(run.checkpoint_step)} · ${number(run.tokens)} / ${number(run.target_tokens)} tokens · ${run.stage || "?"} / ${run.dataset || "?"} · ${compactState(run)}`;

  function populateRuns(fallbackName) {
    const select = $("runName"), current = select.value;
    select.replaceChildren();
    state.runs.forEach((run) => { const option = document.createElement("option"); option.value = run.name; option.textContent = runOptionLabel(run); select.append(option); });
    const names = new Set(state.runs.map((r) => r.name));
    const wanted = names.has(current) ? current : (names.has(fallbackName) ? fallbackName : (state.runs[0]?.name || ""));
    select.value = wanted;
  }
  // The trained assistant is the simple default; every historical checkpoint
  // stays selectable below with its provenance intact — nothing is deleted.
  // The default run comes from the backend promotion pointer when present.
  // Lineage (not bare step number) decides freshness: steps restart every
  // stage, so tokens + saved prefixes + promotion status win.
  const assistantCheckpoints = () => state.checkpoints
    .filter((cp) => cp.run === state.assistantRun)
    .sort((a, b) => (b.step || 0) - (a.step || 0));
  const lineageEntries = () => (state.assistant && state.assistant.lineage && state.assistant.lineage.length)
    ? state.assistant.lineage
    : assistantCheckpoints().map((cp) => ({ run: cp.run, checkpoint: cp.path, path: cp.path, tokens: cp.tokens || 0, step: cp.step || 0 }));
  const approvedPath = () => {
    const def = state.assistant && state.assistant.default;
    if (def && def.checkpoint) return def.checkpoint;
    return assistantCheckpoints()[0]?.path || null;
  };
  const headEntry = () => (state.assistant && state.assistant.head) || null;
  const entryTokens = (path) => {
    const hit = lineageEntries().find((e) => (e.checkpoint || e.path) === path);
    if (hit) return hit.tokens || 0;
    const cp = state.checkpoints.find((c) => c.path === path);
    return cp?.tokens || 0;
  };
  const isExperimentalPath = (path) => {
    const head = headEntry(), approved = approvedPath();
    if (!head || !head.checkpoint) return false;
    if (head.checkpoint === path && approved !== path) return true;
    return false;
  };
  function lineageLabelFor(cpPath, cpStep, cpRun) {
    const approved = approvedPath();
    const tokens = entryTokens(cpPath);
    const tok = tokens ? ` · ${number(tokens)} tokens` : "";
    if (cpPath === approved) return `Approved default · ${cpRun} · ${fmtStep(cpStep)}${tok}`;
    if (isExperimentalPath(cpPath)) return `Experimental · ${cpRun} · ${fmtStep(cpStep)}${tok} (unapproved)`;
    return `AdamLM Assistant · ${cpRun} · ${fmtStep(cpStep)}${tok}`;
  }
  function populateCheckpointSelect(select, currentValue, fallbackValue) {
    select.replaceChildren();
    const assistant = select.id === "playCheckpoint" ? assistantCheckpoints() : [];
    if (assistant.length) {
      const group = document.createElement("optgroup");
      group.label = "AdamLM Assistant — lineage (Approved vs Experimental)";
      assistant.forEach((cp) => {
        const option = document.createElement("option");
        option.value = cp.path;
        option.textContent = lineageLabelFor(cp.path, cp.step, cp.run);
        group.append(option);
      });
      select.append(group);
    }
    CATEGORY_ORDER.forEach((category) => {
      const items = state.checkpoints.filter((cp) => (cp.category || "general-pretraining") === category && cp.run !== state.assistantRun);
      if (!items.length) return;
      const group = document.createElement("optgroup");
      group.label = `Advanced · ${CATEGORY_LABELS[category] || category}`;
      items.forEach((cp) => {
        const option = document.createElement("option");
        option.value = cp.path;
        const badge = cp.path === approvedPath() ? " · approved" : (isExperimentalPath(cp.path) ? " · experimental" : "");
        option.textContent = `${cp.is_production ? "PRODUCTION · " : ""}${cp.run} · ${fmtStep(cp.step)}${cp.tokens ? ` · ${number(cp.tokens)} tokens` : ""}${cp.stage === "sft" ? " · chat-ready" : ""}${badge}`;
        group.append(option);
      });
      select.append(group);
    });
    const paths = new Set(state.checkpoints.map((cp) => cp.path));
    select.value = paths.has(currentValue) ? currentValue : (paths.has(fallbackValue) ? fallbackValue : (state.checkpoints[0]?.path || ""));
  }
  function populateCheckpoints(prefs) {
    populateCheckpointSelect($("parentCheckpoint"), $("parentCheckpoint").value, null);
    // Initial fallback prefers the approved default; later refreshes preserve
    // the user's current selection (never auto-switch mid-chat).
    const assistantDefault = approvedPath() || assistantCheckpoints()[0]?.path;
    populateCheckpointSelect($("playCheckpoint"), $("playCheckpoint").value, prefs.playCheckpoint || assistantDefault);
  }
  let advancedListKey = "";
  function renderAdvancedList() {
    const box = $("advancedList");
    if (!box) return;
    const key = state.checkpoints.map((cp) => `${cp.path}@${cp.step}`).join("|");
    if (key === advancedListKey) return;
    advancedListKey = key;
    box.replaceChildren();
    const rest = state.checkpoints.filter((cp) => cp.run !== state.assistantRun);
    if (!rest.length) { box.textContent = "No other checkpoints."; return; }
    const groups = new Map();
    rest.forEach((cp) => {
      const key = cp.category || "general-pretraining";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(cp);
    });
    CATEGORY_ORDER.forEach((category) => {
      const items = (groups.get(category) || []).sort((a, b) => (b.step || 0) - (a.step || 0));
      if (!items.length) return;
      const div = document.createElement("div");
      div.className = "prov-group";
      const head = document.createElement("strong");
      head.textContent = CATEGORY_LABELS[category] || category;
      div.append(head);
      items.slice(0, 8).forEach((cp) => {
        const line = document.createElement("div");
        const run = state.byName[cp.run];
        line.textContent = `${cp.run} · step ${cp.step} · ${number(cp.tokens)} tokens · ${run?.stage || "?"} · ${cp.path}`;
        div.append(line);
      });
      if (items.length > 8) {
        const more = document.createElement("div");
        more.textContent = `… and ${items.length - 8} more in this group (see Training tab selectors)`;
        div.append(more);
      }
      box.append(div);
    });
  }
  function setSelectedCheckpointForRun() { const run = selectedRun(); if (run?.checkpoint) $("parentCheckpoint").value = run.checkpoint; }

  function updateSelectedContext() {
    const run = selectedRun(), checkpoint = state.checkpoints.find((cp) => cp.path === $("parentCheckpoint").value), checkpointRun = state.byName[checkpoint?.run] || run;
    state.selectedRun = run || null;
    $("runHint").textContent = run ? `Existing run · ${run.stage} · ${run.dataset} · ${compactState(run)}` : "Select a saved lineage.";
    $("checkpointHint").textContent = run?.checkpoint ? `Latest saved checkpoint: ${run.checkpoint}` : "Actual saved path and lineage progress appear below.";
    if (checkpoint?.path) {
      $("checkpointPath").textContent = checkpoint.path;
      $("checkpointProgress").textContent = checkpointRun ? `${(checkpointRun.progress ?? 0).toFixed(1)}% of target` : `${fmtStep(checkpoint.step)} recorded`;
      $("checkpointTokens").textContent = checkpointRun ? `${number(checkpointRun.tokens)} / ${number(checkpointRun.target_tokens)}` : fmtStep(checkpoint.step);
      $("checkpointBar").style.width = `${checkpointRun?.progress || 0}%`;
    } else { $("checkpointPath").textContent = "No checkpoint available"; $("checkpointProgress").textContent = "—"; $("checkpointTokens").textContent = "—"; $("checkpointBar").style.width = "0"; }
    const completable = !!(run?.checkpoint && (run.display_state === "target_reached" || (run.target_tokens && run.tokens >= run.target_tokens)));
    $("continueButton").hidden = !completable || !run;
    if (run && completable) $("continueButton").textContent = `Continue training from ${run.name}`;
  }

  function pillClass(displayState) {
    const s = String(displayState || "idle");
    if (s.startsWith("data_stop")) return "error";
    if (["running", "saving", "initializing", "verifying_data"].includes(s)) return "running";
    if (s === "stopping") return "stopping";
    if (["stopped", "session_limit", "interrupted", "disk_pause"].includes(s)) return "warn";
    if (["target_reached"].includes(s)) return "done";
    if (s === "error") return "error";
    return "";
  }
  function setBar(barId, percent) {
    const bar = $(barId);
    if (!bar) return;
    bar.style.width = `${Math.min(100, Math.max(0, percent || 0))}%`;
    const track = bar.closest(".progress-track");
    if (track) track.setAttribute("aria-valuenow", String(Math.round(Math.min(100, Math.max(0, percent || 0)))));
  }
  function renderActiveSession() {
    const session = state.activeSession, has = !!session;
    $("activeSession").hidden = !has;
    $("idleSession").hidden = has;
    document.querySelectorAll(".session-panel .progress-track").forEach((t) => t.classList.toggle("live", has && !["stopping"].includes(session?.display_state)));
    if (!has) return;
    $("sessionTitle").textContent = `Active session · ${session.run_name}`;
    const pill = $("sessionPill");
    pill.textContent = stateLabel(session.display_state);
    pill.className = `session-pill ${pillClass(session.display_state)}`;
    const lineage = session.parent_checkpoint
      ? `${session.dataset} · ${session.stage} · from ${session.parent_checkpoint.split(/[/\\]/).slice(-3).join("/")} · ${session.directory}`
      : `${session.dataset} · ${session.stage} · ${session.directory}`;
    $("sessionLineage").textContent = lineage;
    $("sessionTokenLabel").textContent = session.target_tokens ? `${number(session.tokens)} / ${number(session.target_tokens)} · ${number(session.remaining_tokens)} left` : `${number(session.tokens)} processed`;
    setBar("sessionTokenBar", session.progress || 0);
    if (session.session_max_seconds) {
      $("sessionTimeLabel").textContent = session.elapsed_seconds != null ? `${duration(session.elapsed_seconds)} elapsed · ${duration(session.session_remaining_seconds)} left` : "Session timing unavailable";
      setBar("sessionTimeBar", session.elapsed_seconds != null ? (session.elapsed_seconds / session.session_max_seconds) * 100 : 0);
    } else if (session.elapsed_seconds != null) {
      $("sessionTimeLabel").textContent = `${duration(session.elapsed_seconds)} elapsed · run until stopped`;
      setBar("sessionTimeBar", 100);
    } else {
      $("sessionTimeLabel").textContent = "Session timing unavailable";
      setBar("sessionTimeBar", 0);
    }
    const rate = session.live_throughput ?? session.throughput;
    const facts = [
      ["Progress", `${(session.progress ?? 0).toFixed(1)}%`],
      ["Tokens remaining", session.target_tokens ? integer(session.remaining_tokens) : "Unavailable"],
      ["Speed", rate != null ? `${number(rate)}/s${session.live_throughput != null ? " live" : " historic"}` : "Unavailable"],
      ["ETA", session.eta != null ? duration(session.eta) : "Unavailable"],
      ["Train loss", session.train_loss ?? "Unavailable"],
      ["Validation loss", session.validation_loss ?? "Unavailable"],
      ["Learning rate", session.learning_rate ?? "Unavailable"],
      ["Checkpoint", session.checkpoint ? session.checkpoint.split(/[/\\]/).pop() : "Unavailable"],
      ["Saved", session.checkpoint_time != null ? fmtClock(session.checkpoint_time) : "Unavailable"],
    ];
    $("sessionFacts").replaceChildren(...facts.map(([label, value]) => { const div = document.createElement("div"), span = document.createElement("span"), strong = document.createElement("strong"); span.textContent = label; strong.textContent = String(value); div.append(span, strong); return div; }));
    $("sessionNote").textContent = session.stop_pending ? "Stopping — saving checkpoint…" : "Training continues if the browser closes.";
    $("sessionStopButton").disabled = state.stopping;
  }

  function updateTelemetry(data) {
    const active = data.runs.find((run) => run.active), shown = active || selectedRun();
    const status = shown?.display_state || shown?.state || "idle";
    $("stateValue").textContent = stateLabel(status);
    $("stateSub").textContent = active ? `active · ${active.name}` : "No active trainer";
    $("stepValue").textContent = integer(shown?.step);
    $("tokensValue").textContent = number(shown?.tokens);
    $("tokensSub").textContent = shown?.target_tokens ? `of ${number(shown.target_tokens)} target` : "of target";
    const rate = shown?.live_throughput ?? shown?.throughput;
    $("throughputValue").textContent = rate != null ? `${number(rate)}/s` : "Unavailable";
    $("etaValue").textContent = shown?.eta != null ? `ETA ${duration(shown.eta)}` : "ETA unavailable";
    const gpu = data.gpu || {};
    $("gpuValue").textContent = gpu.utilization == null ? "Unavailable" : `${Math.round(gpu.utilization)}%`;
    $("vramValue").textContent = gpu.memory_used_mib == null ? "Unavailable" : `${Math.round(gpu.memory_used_mib)} / ${gpu.memory_total_mib != null ? Math.round(gpu.memory_total_mib) : "?"} MiB`;
    $("tempValue").textContent = gpu.temperature_c == null ? "temperature unavailable" : `${Math.round(gpu.temperature_c)}°C`;
    const free = data.disk?.free;
    $("storageValue").textContent = free == null ? "Unavailable" : `${(free / 2 ** 30).toFixed(1)} GiB`;
    $("storageSub").textContent = data.disk?.total ? `${(data.disk.total / 2 ** 30).toFixed(1)} GiB total` : "free";
    const stateClass = (status === "error" || String(status).startsWith("data_stop")) ? "error" : status === "target_reached" ? "done" : ["running", "initializing", "saving", "verifying_data"].includes(status) ? "active" : ["stopped", "stopping", "interrupted", "disk_pause", "session_limit"].includes(status) ? "warn" : "";
    $("stateValue").parentElement.className = `stat stat-state ${stateClass}`;
    document.body.dataset.state = status;
    document.title = active ? `● ${active.name} · AdamLM Studio` : "AdamLM · Local Training Studio";
    const anyActive = data.runs.some((run) => run.active);
    $("activeBadge").textContent = data.runs.filter((run) => run.active).length;
    $("stopButton").disabled = !anyActive || state.stopping;
    $("startButton").disabled = anyActive || state.starting;
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  function chartText(svg, x, y, text, anchor = "start", cls = "axis-label") { const el = document.createElementNS(SVG_NS, "text"); el.setAttribute("x", x); el.setAttribute("y", y); el.setAttribute("text-anchor", anchor); el.setAttribute("class", cls); el.textContent = text; svg.append(el); }
  function renderChart(rows, runName) {
    const svg = $("lossChart"); svg.replaceChildren();
    const width = 760, height = 320, pad = { l: 56, r: 20, t: 24, b: 36 }, x0 = pad.l, x1 = width - pad.r, y0 = pad.t, y1 = height - pad.b;
    // Null/empty losses (e.g. graceful_stop marker rows) must be skipped, not
    // plotted as zero — Number(null) is 0 and would fake a loss collapse.
    const num = (v) => (v === null || v === undefined || v === "" ? null : Number(v));
    const point = (step, loss) => { const s = num(step), l = num(loss); return (s !== null && l !== null && Number.isFinite(s) && Number.isFinite(l)) ? [[s, l]] : []; };
    const train = rows.flatMap((r) => point(r.step, r.train_loss ?? (r.reason ? r.last_train_loss : null)));
    const valid = rows.flatMap((r) => point(r.step, r.validation_loss ?? (r.reason ? r.final_validation_loss : null)));
    const all = train.concat(valid);
    for (let i = 0; i < 4; i++) { const y = y0 + (y1 - y0) * i / 3; const line = document.createElementNS(SVG_NS, "line"); line.setAttribute("x1", x0); line.setAttribute("x2", x1); line.setAttribute("y1", y); line.setAttribute("y2", y); line.setAttribute("class", "grid-line"); svg.append(line); }
    if (!all.length) { chartText(svg, width / 2, height / 2, rows.length ? "No plottable loss in this run's metrics" : "No loss metrics recorded for this run", "middle", "empty-label"); $("chartMin").textContent = "step —"; $("chartMax").textContent = "step —"; $("chartScope").textContent = `${runName || "Selected run"} · metrics.jsonl`; return; }
    const xmin = Math.min(...all.map((p) => p[0])), xmax = Math.max(...all.map((p) => p[0])), ymin0 = Math.min(...all.map((p) => p[1])), ymax0 = Math.max(...all.map((p) => p[1])), xspan = Math.max(1, xmax - xmin), yspan0 = Math.max(.01, ymax0 - ymin0), ymin = ymin0 - yspan0 * .1, ymax = ymax0 + yspan0 * .1, yspan = ymax - ymin;
    const X = (x) => x0 + (x - xmin) / xspan * (x1 - x0), Y = (y) => y1 - (y - ymin) / yspan * (y1 - y0);
    chartText(svg, x0 - 9, y0 + 4, ymax.toFixed(2), "end"); chartText(svg, x0 - 9, y1, ymin.toFixed(2), "end"); chartText(svg, x0, height - 10, fmtStep(xmin)); chartText(svg, x1, height - 10, fmtStep(xmax), "end");
    $("chartMin").textContent = fmtStep(xmin); $("chartMax").textContent = fmtStep(xmax);
    const trailing = state.activeRun && runName === state.activeRun ? " · live" : "";
    $("chartScope").textContent = `${runName || "Selected run"} · actual training steps${trailing}`;
    const draw = (points, cls, withArea) => {
      if (!points.length) return;
      if (withArea && points.length > 1) {
        const area = document.createElementNS(SVG_NS, "path");
        area.setAttribute("d", `M${X(points[0][0])},${y1} ` + points.map(([x, y]) => `L${X(x)},${Y(y)}`).join(" ") + ` L${X(points[points.length - 1][0])},${y1} Z`);
        area.setAttribute("class", "train-area");
        svg.append(area);
      }
      const d = points.map(([x, y], i) => `${i ? "L" : "M"}${X(x)},${Y(y)}`).join(" ");
      const path = document.createElementNS(SVG_NS, "path"); path.setAttribute("d", d); path.setAttribute("class", cls); svg.append(path);
      points.forEach(([x, y]) => {
        const c = document.createElementNS(SVG_NS, "circle");
        c.setAttribute("cx", X(x)); c.setAttribute("cy", Y(y)); c.setAttribute("r", 3); c.setAttribute("class", `${cls}-point`);
        const tip = document.createElementNS(SVG_NS, "title");
        tip.textContent = `step ${Math.round(x)} · loss ${Number(y).toFixed(4)}`;
        c.append(tip);
        svg.append(c);
      });
    };
    draw(train, "train-line", true); draw(valid, "valid-line", false);
  }
  async function loadMetrics() {
    const name = $("runName").value;
    if (!name) return;
    try {
      const result = await json(`/api/metrics?run=${encodeURIComponent(name)}`);
      if (result.run === $("runName").value) { state.metricsFor = result.run; renderChart(result.metrics, result.run); }
    } catch (error) { showError(error); }
  }
  function renderPlan(plan) {
    $("planStatus").textContent = "READY";
    $("planStatus").classList.remove("bad");
    $("planTitle").textContent = `${plan.mode === "resume" ? "Resume immutable stage" : "Create continuation stage"} · ${number(plan.remaining_tokens)} tokens remaining`;
    $("planPath").textContent = plan.run_dir;
    const facts = [["Additional", number(plan.additional_tokens)], ["Cumulative", number(plan.cumulative_tokens)], ["Effective target", number(plan.effective_target)], ["Est. completion", plan.estimated_duration], ["LR schedule", `${plan.learning_rate} → ${plan.minimum_learning_rate}`], ["Model", `${integer(plan.params)} params`]];
    $("planFacts").replaceChildren(...facts.map(([label, value]) => { const div = document.createElement("div"), span = document.createElement("span"), strong = document.createElement("strong"); span.textContent = label; strong.textContent = value; div.append(span, strong); return div; }));
    $("planCheckpoint").innerHTML = `Checkpoint ${html(plan.checkpoint)}<br>Run folder ${html(plan.run_dir)}`;
  }
  function showError(error) {
    $("planStatus").textContent = "CHECK INPUT";
    $("planStatus").classList.add("bad");
    $("planTitle").textContent = error.message;
    $("planPath").textContent = "Existing checkpoints and schedules remain unchanged.";
    $("notice").textContent = error.message;
    $("notice").className = "notice error";
  }
  function showNotice(message, isError = false) { $("notice").textContent = message; $("notice").className = isError ? "notice error" : "notice"; }
  let planTimer;
  async function refreshPlan() {
    clearTimeout(planTimer);
    planTimer = setTimeout(async () => {
      try {
        renderPlan((await post("/api/plan", payload())).plan);
        if (!state.stopping && !state.starting) showNotice("Preflight only. No training starts until Start training is pressed.");
      } catch (error) { showError(error); }
    }, 120);
  }

  // ---- Auto Train (default workflow): two inputs, one plan, one button. ----
  const autoSeconds = () => {
    const value = parseFloat($("autoDuration").value);
    if (!Number.isFinite(value) || value <= 0) return null;
    return value * ($("autoDurationUnit").value === "hours" ? 3600 : 60);
  };
  const autoPayload = () => ({ additional_tokens: $("autoTokens").value, max_seconds: autoSeconds(), pass_policy: $("autoPassPolicy").value, stage_count: $("autoStageCount").value, repeat_confirm: $("autoRepeatConfirm").checked });
  const AUTO_PILL = { none: ["NO PLAN", ""], running: ["RUNNING", "running"], stopped: ["STOPPED", "warn"], waiting: ["WAITING", "warn"], regression: ["REGRESSION", "warn"], "time-exceeded": ["TIME UP", "warn"], failed: ["FAILED", "error"], evaluating: ["EVALUATING", "running"], done: ["DONE", "done"] };
  function autoFact(label, value, mono = false) {
    const div = document.createElement("div"); div.className = "auto-fact";
    const span = document.createElement("span"); span.textContent = label;
    const strong = document.createElement("strong"); if (mono) strong.className = "mono"; strong.textContent = value;
    div.append(span, strong); return div;
  }
  function budgetLine(pf, eff) {
    const budget = pf.budget || {};
    const requested = budget.requested_tokens ?? pf.requested_tokens ?? 0;
    const effective = budget.effective_tokens ?? eff;
    const source = budget.source_epoch_tokens ?? budget.epoch_tokens ?? 0;
    return `requested ${number(requested)} → scheduled ${number(effective)} · ${budget.stage_count ?? (pf.stages || []).length} stage(s)`
      + ` · ${budget.pass_policy || "unique_once"} · ${number(source)} source tokens/pass`;
  }
  function renderAutoPreflight(pf) {
    const box = $("autoPreflight");
    box.replaceChildren();
    const head = pf.head || {}, foundation = pf.foundation || {}, diet = pf.diet || {}, alloc = pf.allocation || {};
    const dd = diet.dailydialog || {}, dolly = diet.dolly || {};
    const eff = pf.effective_tokens ?? 0;
    box.append(
      autoFact("Starting checkpoint", `${head.run || "?"} · step ${head.step ?? "?"} · ${number(head.tokens)} tokens trained`),
      autoFact("Foundation", `${foundation.run || "?"} · ${foundation.state || "?"} (preserved, never retrained)`),
      autoFact("Diet", `DailyDialog ${integer(dd.rows)} rows / ${number(dd.tokens)} + Dolly ${integer(dolly.rows)} rows / ${number(dolly.tokens)} · ${(diet.revision || "")} · response-only masking`),
      autoFact("Allocation", `DailyDialog ${number(eff * (alloc.dd_share ?? 0.7))} + Dolly ${number(eff * (alloc.dolly_share ?? 0.3))} — ${alloc.basis || ""}`),
      autoFact("Budget", budgetLine(pf, eff)),
    );
    (pf.stages || []).forEach((s, i) => box.append(autoFact(`Stage ${i + 1}`, `${s.run_dir} · ${number(s.target_tokens)} tokens · ${s.epoch_label}`, true)));
    const st = pf.storage || {};
    box.append(autoFact("Storage", `project ${(st.project_gib ?? 0).toFixed(2)} GiB + ~${(st.est_new_gib ?? 0).toFixed(2)} GiB new vs ${st.budget_gib ?? "?"} GiB budget · floor ${(st.floor_gib ?? 5).toFixed(0)} GiB intact · free ${(st.free_gib ?? 0).toFixed(1)} GiB`));
    const tm = pf.time || {};
    box.append(autoFact("Stopping", `time limit ${tm.max_duration || "—"}${tm.est_duration && tm.est_duration !== "Unavailable" ? ` · est. ${tm.est_duration}` : ""} · Stop button saves gracefully · disk pause halts safely`));
    box.append(autoFact("Evaluation", pf.eval || "held-out comparison before any promotion"));
    const warn = $("autoWarn"), warnings = pf.warnings || [];
    warn.hidden = !warnings.length;
    warn.replaceChildren(...warnings.map((w) => { const d = document.createElement("div"); d.textContent = w; return d; }));
    const repeats = !!(pf.budget?.repeated_data || (pf.requested_tokens != null && pf.effective_tokens != null && pf.requested_tokens > pf.effective_tokens));
    state.autoRepeats = repeats;
    $("autoRepeatRow").hidden = !repeats;
    if (!repeats) $("autoRepeatConfirm").checked = false;
  }
  function renderAutoStatus() {
    const auto = state.auto || { status: "none" };
    const [label, cls] = AUTO_PILL[auto.status] || [String(auto.status).toUpperCase(), ""];
    const pill = $("autoPill");
    const doneLabel = auto.status === "done" ? (auto.promotion?.promoted ? "DONE · PROMOTED" : "DONE · KEPT DEFAULT") : label;
    pill.textContent = doneLabel;
    pill.className = `session-pill ${auto.status === "done" ? (auto.promotion?.promoted ? "done" : "") : cls}`;
    const anyActive = state.runs.some((run) => run.active);
    const busy = auto.status === "running" || auto.status === "evaluating";
    $("autoStartButton").disabled = anyActive || busy || state.autoStarting;
    $("autoStopButton").disabled = !(auto.status === "running" || auto.status === "evaluating") || state.stopping;
    $("autoResumeButton").disabled = !["stopped", "waiting", "time-exceeded"].includes(auto.status) || anyActive;
    const note = $("autoNote");
    if (busy) note.textContent = `Auto Train ${auto.plan_id} active${auto.current_stage ? ` · stage ${auto.current_stage.split("/").pop()}` : ""} — Stop saves gracefully.`;
    else if (auto.status === "done") note.textContent = auto.promotion?.promoted ? `Promoted: ${auto.promotion.reason}` : `Finished, default unchanged. ${auto.promotion?.reason || auto.stop_reason || ""}`;
    else if (auto.status === "stopped" || auto.status === "failed" || auto.status === "time-exceeded") note.textContent = `${auto.stop_reason || auto.failure || auto.status} — Resume continues without replanning.`;
    else note.textContent = "Preflight only — nothing starts until you press Start Training.";
    renderAutoActual(auto);
    renderEvalScorecard(auto);
    const preview = $("autoPreviewCaption");
    if (preview) preview.textContent = (auto.status && auto.status !== "none")
      ? "Next plan preview — hypothetical from the inputs above; it does not change the saved plan shown above."
      : "No saved plan yet — preview of what Start Training would create from the inputs above.";
  }
  function renderAutoActual(auto) {
    const box = $("autoActual");
    if (!box) return;
    if (!auto || auto.status === "none") { box.hidden = true; box.replaceChildren(); return; }
    box.hidden = false;
    box.replaceChildren();
    const title = document.createElement("strong");
    title.textContent = `Actual plan ${auto.plan_id || ""} · ${String(auto.status || "").toUpperCase()}`;
    box.append(title);
    (auto.stages || []).forEach((s, i) => {
      const line = document.createElement("div");
      line.className = "stage-line";
      const done = s.tokens != null && s.status === "done" ? ` · ${integer(s.tokens)} tokens done` : "";
      line.textContent = `Stage ${i + 1} · ${(s.run_dir || "").split("/").pop()} · ${s.status || "?"} · ${number(s.target_tokens)} target${done}`;
      box.append(line);
    });
    const foot = document.createElement("div");
    foot.className = "stage-line";
    foot.textContent = `${number(auto.done_tokens)} / ${number(auto.effective_tokens)} tokens done`
      + (auto.produced_checkpoint ? ` · produced ${String(auto.produced_checkpoint).split(/[/\\]/).pop()}` : "")
      + (auto.stop_reason ? ` · ${auto.stop_reason}` : "")
      + (auto.failure ? ` · FAILED: ${auto.failure}` : "");
    box.append(foot);
  }
  function renderEvalScorecard(auto) {
    const box = $("autoEval");
    if (!box) return;
    const ev = auto && auto.eval, promo = auto && auto.promotion;
    if (!ev && !(promo && promo.reason)) { box.hidden = true; box.replaceChildren(); return; }
    box.hidden = false;
    box.replaceChildren();
    const head = document.createElement("h4");
    head.textContent = "Evaluation scorecard";
    box.append(head);
    const grid = document.createElement("div");
    grid.className = "eval-grid";
    const cell = (label, value) => {
      const div = document.createElement("div");
      const span = document.createElement("span"); span.textContent = label;
      const strong = document.createElement("strong"); strong.textContent = value;
      div.append(span, strong); return div;
    };
    const f1 = (v) => (v == null || !Number.isFinite(Number(v))) ? "—" : Number(v).toFixed(3);
    if (ev) {
      grid.append(cell("Instruction F1", `${f1(ev.instr_f1_old)} → ${f1(ev.instr_f1_new)}`));
      grid.append(cell("Conversation F1", `${f1(ev.conv_f1_old)} → ${f1(ev.conv_f1_new)}`));
      const n = ev.hygiene_problem_count ?? (ev.hygiene_problems || []).length;
      grid.append(cell("Hygiene", n === 0 ? "pass" : `${n} problem${n === 1 ? "" : "s"}`));
    } else {
      grid.append(cell("Instruction F1", "—"));
      grid.append(cell("Conversation F1", "—"));
      grid.append(cell("Hygiene", "—"));
    }
    box.append(grid);
    const problems = (ev && ev.hygiene_problems) || [];
    if (problems.length) {
      const list = document.createElement("div");
      list.className = "eval-reason";
      list.textContent = `Hygiene: ${problems.slice(0, 3).join("; ")}${problems.length > 3 ? ` (+${problems.length - 3} more)` : ""}`;
      box.append(list);
    }
    if (promo && promo.reason) {
      const reason = document.createElement("div");
      reason.className = "eval-reason";
      reason.textContent = `${promo.promoted ? "Promoted. " : "Held. "}${promo.reason}`;
      box.append(reason);
    }
  }
  let autoPlanTimer, lastAutoSig = "", lastAutoAt = 0;
  async function refreshAutoPlan(force = false) {
    clearTimeout(autoPlanTimer);
    autoPlanTimer = setTimeout(async () => {
      const sig = `${$("autoTokens").value}|${$("autoDuration").value}|${$("autoDurationUnit").value}|${$("autoPassPolicy").value}|${$("autoStageCount").value}`;
      if (!force && sig === lastAutoSig && Date.now() - lastAutoAt < 15000) return;
      try {
        const pf = (await post("/api/auto/plan", autoPayload())).preflight;
        lastAutoSig = sig; lastAutoAt = Date.now();
        renderAutoPreflight(pf);
      } catch (error) {
        lastAutoSig = ""; // retry: errors may clear as runs/checkpoints change
        $("autoPreflight").replaceChildren(autoFact("Cannot plan", error.message));
        $("autoWarn").hidden = true;
        state.autoRepeats = false;
        $("autoRepeatRow").hidden = true;
      }
    }, 150);
  }
  async function autoStart() {
    if (state.autoStarting) return;
    if (state.autoRepeats && !$("autoRepeatConfirm").checked) {
      $("autoNote").textContent = "This plan repeats training rows — tick the repetition confirmation first.";
      return;
    }
    try {
      state.autoStarting = true;
      $("autoStartButton").disabled = true;
      $("autoNote").textContent = "Starting Auto Train…";
      const result = await post("/api/auto/start", autoPayload());
      state.followActive = true;
      $("autoNote").textContent = result.message;
      await refresh(false);
    } catch (error) {
      $("autoPreflight").replaceChildren(autoFact("Cannot start", error.message));
      const warn = $("autoWarn"); warn.hidden = false;
      warn.replaceChildren(((w) => { const d = document.createElement("div"); d.textContent = w; return d; })(error.message));
    } finally { state.autoStarting = false; await refresh(false); }
  }
  async function autoStop() {
    try {
      $("autoStopButton").disabled = true;
      const result = await post("/api/auto/stop", {});
      $("autoNote").textContent = result.message;
      await refresh(false);
    } catch (error) { $("autoNote").textContent = error.message; }
  }
  async function autoResume() {
    try {
      $("autoResumeButton").disabled = true;
      // Resume re-arms a fresh time allowance from the current inputs without
      // changing the token plan or stage layout.
      const result = await post("/api/auto/resume", { max_seconds: autoSeconds() });
      $("autoNote").textContent = result.message;
      await refresh(false);
    } catch (error) { $("autoNote").textContent = error.message; }
  }

  async function refresh(initial = false) {
    const prefs = initial ? loadPrefs() : null;
    if (initial && prefs) applyPrefs(prefs);
    try {
      const data = await json(`/api/overview?run=${encodeURIComponent($("runName").value || "")}`);
      state.runs = data.runs;
      state.checkpoints = data.checkpoints;
      state.byName = Object.fromEntries(data.runs.map((r) => [r.name, r]));
      state.auto = data.auto || { status: "none" };
      state.assistant = data.assistant || { default: data.assistant_default || null, head: null, lineage: [] };
      state.assistantChat = data.assistant_chat || null;
      state.gpuSession = data.gpu_session || null;
      state.provider = data.provider || {};
      if (data.research && data.research.status && data.research.status !== "none") {
        const changed = JSON.stringify(data.research) !== JSON.stringify(researchState.session);
        researchState.session = data.research;
        if (changed) renderResearch();
      }
      if (!state.assistant.default && data.assistant_default) state.assistant.default = data.assistant_default;
      state.assistantRun = (state.assistant.default && state.assistant.default.run)
        || (data.assistant_default && data.assistant_default.run) || "sft-assistant";
      // Newer backends report active_run/active_session directly; older ones
      // only flag runs with active:true, so derive the same view from that.
      const activeObj = data.runs.find((run) => run.active) || null;
      state.activeRun = data.active_run || (activeObj ? activeObj.name : null);
      state.activeSession = data.active_session || (activeObj ? sessionFromRun(activeObj) : null);
      if (initial) {
        const remembered = prefs?.runName;
        populateRuns(remembered || data.preferred_run);
        populateCheckpoints(prefs || {});
        if (!state.byName[$("runName").value] && remembered) showNotice(`Remembered run '${remembered}' is no longer present; showing ${$("runName").value || "nothing"}.`);
      } else {
        if (state.followActive && state.activeRun && $("runName").value !== state.activeRun) {
          $("runName").value = state.activeRun;
          await loadMetrics();
        }
        populateRuns(null);
        populateCheckpointSelect($("parentCheckpoint"), $("parentCheckpoint").value, state.byName[$("runName").value]?.checkpoint);
        populateCheckpointSelect($("playCheckpoint"), $("playCheckpoint").value, prefs?.playCheckpoint);
        applySelectFilters();
      }
      if (initial || !state.selectedRun) setSelectedCheckpointForRun();
      updateSelectedContext();
      updatePlaygroundState();
      updateChatIdentity();
      renderActiveSession();
      updateTelemetry(data);
      renderAutoStatus();
      await refreshAutoPlan();
      if (!initial && state.followActive && state.activeRun) await loadMetrics();
      else if (initial) await loadMetrics();
      await refreshPlan();
      savePrefs();
      $("connectionDot").className = "status-dot ok";
      $("connectionText").textContent = "Backend online";
      $("lastRefresh").textContent = new Date().toLocaleTimeString();
    } catch (error) {
      $("connectionDot").className = "status-dot error";
      $("connectionText").textContent = "Backend unavailable";
      showNotice(error.message, true);
    }
  }

  async function waitForActive(runName, timeoutMs = 15000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        const data = await json(`/api/overview?run=${encodeURIComponent(runName)}`);
        if (data.active_run === runName) return true;
        const hit = (data.runs || []).find((r) => r.name === runName);
        if (hit && hit.active) return true;
        if (!hit) return false;
      } catch { /* keep waiting for the backend */ }
      await new Promise((r) => setTimeout(r, 1000));
    }
    return false;
  }
  async function startTraining() {
    if (state.starting || state.stopping) return;
    try {
      state.starting = true;
      $("startButton").disabled = true;
      $("actionNote").textContent = "Launching trainer…";
      const result = await post("/api/training/start", payload());
      // Automatically select and monitor the newly created stage.
      state.followActive = true;
      if (result.run_name) {
        await refresh(false);
        if (state.byName[result.run_name]) { $("runName").value = result.run_name; setSelectedCheckpointForRun(); }
        savePrefs();
      }
      $("actionNote").textContent = `${result.message} Confirming the trainer is active…`;
      const launched = result.run_name ? await waitForActive(result.run_name) : false;
      await refresh(false);
      await loadMetrics();
      if (launched) {
        $("actionNote").textContent = `Training active in ${result.run_name}. Metrics and checkpoint info now follow that stage.`;
        showNotice(`Training active in ${result.run_name}. The session continues if the browser closes.`);
      } else {
        $("actionNote").textContent = `Start may have failed for ${result.run_name || "the new stage"} — check status.json and web-training.log in its run folder.`;
        showNotice(`The new stage '${result.run_name || "?"}' is not reporting as active. Check its web-training.log; nothing was overwritten.`, true);
      }
    } catch (error) { showError(error); }
    finally { state.starting = false; await refresh(false); }
  }
  async function waitForIdle(timeoutMs = 300000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        const data = await json("/api/overview");
        if (!data.active_run && !(data.runs || []).some((r) => r.active)) return data;
      } catch { /* keep waiting while the backend restarts or the trainer saves */ }
      await new Promise((r) => setTimeout(r, 1500));
    }
    return null;
  }
  async function stopTraining() {
    if (state.stopping) return;
    const target = state.activeRun || $("runName").value;
    if (!target) { showNotice("No run selected and no active trainer.", true); return; }
    try {
      state.stopping = true;
      $("stopButton").disabled = true;
      $("sessionStopButton").disabled = true;
      $("actionNote").textContent = "Stopping — saving checkpoint. Waiting for the trainer's safe save…";
      $("sessionNote").textContent = "Stopping — saving checkpoint…";
      showNotice("Stopping — saving checkpoint. The trainer finishes its safe save before exiting; unsaved progress is not discarded.");
      const result = await post("/api/training/stop", { run_name: target });
      if (result.already_stopped) {
        state.stopping = false;
        $("actionNote").textContent = result.message;
        showNotice(result.message);
        await refresh(false);
        return;
      }
      // The stop request targets the real active trainer; follow it even if the selection differs.
      if (result.run_name && result.run_name !== $("runName").value) {
        state.followActive = true;
        $("runName").value = result.run_name;
      }
      $("actionNote").textContent = `${result.message} Waiting for the save to complete…`;
      const final = await waitForIdle();
      state.stopping = false;
      await refresh(false);
      await loadMetrics();
      await refreshPlan();
      if (final) {
        const done = state.byName[result.run_name];
        $("actionNote").textContent = done?.checkpoint
          ? `Stopped. Saved checkpoint ${done.checkpoint} · ${number(done.tokens)} / ${number(done.target_tokens)} tokens.`
          : `Stopped ${result.run_name}.`;
        showNotice(done?.checkpoint ? `Graceful stop complete. Saved ${done.checkpoint}.` : `Graceful stop complete for ${result.run_name}.`);
      } else {
        $("actionNote").textContent = "Still stopping — the trainer is finishing a long save. Leave the backend running; progress is not lost.";
        showNotice("Still stopping — the trainer is finishing its save. Leave the backend running.", true);
      }
    } catch (error) { state.stopping = false; showError(error); }
    finally { $("sessionStopButton").disabled = false; }
  }
  function continueTraining() {
    const run = selectedRun();
    if (!run?.checkpoint) { showNotice("Select a completed run with a saved checkpoint first.", true); return; }
    state.followActive = false;
    $("mode").value = "continuation";
    populateCheckpointSelect($("parentCheckpoint"), run.checkpoint, run.checkpoint);
    $("parentCheckpoint").value = run.checkpoint;
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    $("stageName").value = `${run.name}-continuation-${stamp}`;
    if (run.dataset && run.dataset !== "mixture") $("dataset").value = run.dataset;
    else if (run.dataset === "mixture") $("dataset").value = "mixture";
    updateSelectedContext();
    refreshPlan();
    savePrefs();
    showNotice(`Continuation prepared from ${run.name}'s latest checkpoint. The completed run's schedule stays immutable; pick Additional tokens and press Start training.`);
    $("additionalTokens").focus();
  }

  function updatePlaygroundState() {
    const cp = state.checkpoints.find((c) => c.path === $("playCheckpoint").value);
    const chatOption = $("playMode").querySelector('option[value="chat"]');
    // Older backends omit cp.stage/cp.category; fall back to the run's stage.
    const sft = (cp?.stage || state.byName[cp?.run]?.stage) === "sft";
    if (chatOption) chatOption.disabled = !sft;
    if (!sft && $("playMode").value === "chat") $("playMode").value = "story";
    $("chatNote").textContent = sft
      ? `Conversation enabled: ${cp.run} is an instruction-tuned SFT checkpoint. Undertrained models may still answer poorly.`
      : "Story completion works with any compatible model. Conversation unlocks only for instruction-tuned SFT checkpoints.";
    const approved = approvedPath();
    $("playCheckpointHint").textContent = !cp
      ? "Production, general-pretraining, SFT, and smoke-test models are grouped separately."
      : cp.path === approved
        ? `Approved default · ${cp.run} · step ${cp.step}${sft ? " · chat-ready" : ""}`
        : isExperimentalPath(cp.path)
          ? `Experimental · ${cp.run} · step ${cp.step} (unapproved — newer lineage weights, not yet promoted)${sft ? " · chat-ready" : ""}`
          : `${CATEGORY_LABELS[cp.category] || cp.category || "Checkpoint"} · ${cp.run} · step ${cp.step}${sft ? " · chat-ready" : ""}`;
    const playMode = $("playMode") ? $("playMode").value : "story";
    const rec = $("playRecommend");
    if (rec) rec.textContent = playMode === "chat"
      ? "Recommended: Approved assistant (chat-ready SFT) for conversation."
      : "Recommended: Production (general pretraining) for story completion.";
    const seedRandom = $("seedRandom") ? $("seedRandom").checked : true;
    if ($("seedValue")) $("seedValue").disabled = seedRandom;
    if ($("seedHint")) $("seedHint").textContent = seedRandom
      ? (state.lastSeed != null ? `Fresh random seed each Generate · last seed ${state.lastSeed}. Uncheck to lock a seed.` : "Fresh random seed each Generate. Uncheck to lock a seed for reproducible tests.")
      : "Locked seed — same prompt + checkpoint + settings reproduces the reply. Use Compare with this seed for fair A/B.";
    updateRegenerateState();
    updateAssistantBanner();
    renderAdvancedList();
  }
  function updateRegenerateState() {
    const regen = $("regenerateButton"), cmp = $("compareButton");
    if (!regen || !cmp) return;
    const busy = ($("generateButton") && $("generateButton").disabled) || (state._genBusy === true);
    const blocked = !!state.activeRun;
    const mode = $("playMode") ? $("playMode").value : "story";
    const key = $("playCheckpoint") ? $("playCheckpoint").value : null;
    const history = key ? historyFor(key) : [];
    const hasLast = mode === "chat" ? history.length >= 2 : !!$("outputBody").querySelector(".assistant pre");
    if (regen) regen.disabled = !!busy || blocked || !hasLast;
    if (regen && blocked) regen.title = `Training '${state.activeRun}' is active — inference paused to protect GPU memory.`;
    else if (regen) regen.title = "";
    const head = headEntry();
    const canCompare = !!head && !!head.checkpoint && !!key && head.checkpoint !== key && !busy && !blocked;
    if (cmp) cmp.disabled = !canCompare;
    if (cmp && blocked) cmp.title = `Training '${state.activeRun}' is active — inference paused to protect GPU memory.`;
    else if (cmp && head && key && head.checkpoint === key) cmp.title = "Select the older checkpoint to compare against the newer lineage head";
    else if (cmp) cmp.title = "";
    const guard = $("inferenceGuard");
    if (guard) {
      if (blocked) {
        guard.hidden = false;
        guard.textContent = `Training '${state.activeRun}' is active — inference is paused so training and generation never compete for GPU memory.`;
      } else guard.hidden = true;
    }
    if ($("generateButton")) {
      $("generateButton").disabled = !!busy || blocked;
      if (blocked) $("generateButton").title = `Training '${state.activeRun}' is active — inference paused to protect GPU memory.`;
      else $("generateButton").title = "";
    }
  }
  function updateAssistantBanner() {
    const banner = $("assistantBanner");
    if (!banner) return;
    const head = headEntry(), selected = $("playCheckpoint") ? $("playCheckpoint").value : null;
    const hide = () => { banner.hidden = true; return; };
    if (!head || !head.checkpoint || !selected) return hide();
    if (head.checkpoint === selected) return hide();
    if (state.dismissedHeadPath === head.checkpoint) return hide();
    // Lineage freshness by tokens, never bare step number (steps restart per stage).
    const headTokens = head.tokens || 0, selTokens = entryTokens(selected);
    if (!(headTokens > selTokens)) return hide();
    const approved = approvedPath();
    const tag = head.checkpoint === approved ? "Approved default" : "Experimental — unapproved";
    $("assistantBannerTitle").textContent = "Newer saved checkpoint available";
    $("assistantBannerBody").textContent = `${head.run} · ${head.tokens != null ? `${Number(head.tokens).toLocaleString("en-US")} tokens` : ""} · ${head.checkpoint} · ${tag}. Current selection stays active; Switch only on click.`;
    banner.hidden = false;
  }
  function skeletonBubble() {
    const div = document.createElement("div");
    div.className = "skeleton";
    div.setAttribute("aria-hidden", "true");
    ["92%", "78%", "85%"].forEach((w) => { const i = document.createElement("i"); i.style.width = w; div.append(i); });
    return div;
  }

  // ---- Multi-turn conversation (chat mode only; story mode stays stateless).
  // The composed prompt uses the exact SFT training format from
  // scripts/prepare_dailydialog.py (User: / Context: history / Assistant:),
  // sent verbatim (chat:false) so the backend never rewraps or truncates it.
  // Histories are keyed by checkpoint path, so each model keeps its own
  // conversation. Weights are never modified during chat.
  const CHAT_BLOCK = 512, CHAT_MARGIN = 32, CHAT_CHARS_PER_TOKEN = 2, CHAT_MAX_TURNS = 30;
  const chatHistories = new Map(); // checkpoint path -> [{role:'user'|'assistant', text}]
  const historyFor = (key) => { if (!chatHistories.has(key)) chatHistories.set(key, []); return chatHistories.get(key); };
  function formatHistory(texts) {
    return texts.map((t, i) => `${i % 2 === 0 ? "User:" : "Assistant:"}\n${t}`).join("\n");
  }
  function composeChatPrompt(stored, latest, newTokens) {
    const budget = Math.max(64, (CHAT_BLOCK - (newTokens || 160) - CHAT_MARGIN)) * CHAT_CHARS_PER_TOKEN;
    const totalEntries = stored.length;
    let context = stored.map((t) => t.text);
    let trimmed = false;
    const build = (ctx, instruction) => `User:\n${instruction}\n` + (ctx.length ? `\nContext:\n${formatHistory(ctx)}\n` : "") + `\nAssistant:\n`;
    let instruction = latest, prompt = build(context, instruction);
    while (prompt.length > budget && context.length >= 2) { context = context.slice(2); trimmed = true; prompt = build(context, instruction); }
    if (prompt.length > budget && context.length) { context = []; trimmed = true; prompt = build(context, instruction); }
    if (prompt.length > budget) { instruction = instruction.slice(prompt.length - budget); trimmed = true; prompt = build(context, instruction); }
    // Turns here are individual messages (user or assistant). kept/total let
    // the context note report honestly what the model actually sees.
    return { prompt, trimmed, keptEntries: context.length, totalEntries };
  }
  function updateChatNote(info = null) {
    const el = $("chatHistoryNote");
    if (!el) return;
    if ($("playMode").value !== "chat") { el.textContent = "Story mode is stateless. Switch to Conversation for multi-turn chat."; return; }
    const history = historyFor($("playCheckpoint").value);
    if (!history.length && !info) {
      el.textContent = "New conversation — earlier turns will be included as you chat.";
      return;
    }
    // Prefer the last composed prompt's kept/total; fall back to full history.
    const total = info && info.totalEntries != null ? info.totalEntries + 1 : history.length + (info ? 1 : 0);
    const kept = info && info.keptEntries != null ? info.keptEntries + 1 : history.length;
    const trimmed = !!(info && info.trimmed);
    el.textContent = trimmed || kept < total
      ? `Using ${kept} of ${total} turns · older turns trimmed to fit the 512-token context.`
      : `Using ${kept} of ${total} turns · full history is included with each reply.`;
  }
  function bubble(role, label, text, error = false) {
    const div = document.createElement("div");
    div.className = role === "user" ? "bubble user" : `bubble assistant${error ? " error-bubble" : ""}`;
    const span = document.createElement("span");
    span.className = "bubble-label";
    span.textContent = label;
    const pre = document.createElement("pre");
    pre.textContent = text;
    div.append(span, pre);
    return div;
  }
  function renderConversation(key, errorText = null) {
    const body = $("outputBody");
    body.replaceChildren();
    const history = historyFor(key);
    const chat = $("playMode").value === "chat";
    if (!history.length && errorText == null) {
      const div = document.createElement("div");
      div.className = "bubble system";
      div.textContent = "Generated output will appear here. Nothing is sent anywhere; inference runs from the selected local checkpoint.";
      body.append(div);
      $("copyButton").disabled = true;
      return;
    }
    history.forEach((turn) => body.append(bubble(
      turn.role,
      turn.role === "user" ? (chat ? "You" : "Prompt") : (chat ? "Assistant · local model" : "Continuation · local model"),
      turn.text,
    )));
    if (errorText != null) body.append(bubble("assistant", "Error", errorText, true));
    const meta = document.createElement("div");
    meta.className = "bubble system";
    meta.textContent = `Generated locally from ${key || "selected checkpoint"} · no external service involved.`;
    body.append(meta);
    $("copyButton").disabled = !history.some((t) => t.role === "assistant");
  }
  function renderOutput(mode, promptText, outputText, checkpointPath, isError = false) {
    const body = $("outputBody");
    body.replaceChildren();
    const promptBubble = document.createElement("div");
    promptBubble.className = "bubble user";
    const promptLabel = document.createElement("span");
    promptLabel.className = "bubble-label";
    promptLabel.textContent = mode === "chat" ? "You" : "Prompt";
    const promptPre = document.createElement("pre");
    promptPre.textContent = promptText;
    promptBubble.append(promptLabel, promptPre);
    const answerBubble = document.createElement("div");
    answerBubble.className = `bubble assistant${isError ? " error-bubble" : ""}`;
    const answerLabel = document.createElement("span");
    answerLabel.className = "bubble-label";
    answerLabel.textContent = isError ? "Error" : mode === "chat" ? "Assistant · local model" : "Continuation · local model";
    const answerPre = document.createElement("pre");
    answerPre.textContent = outputText || "No output returned.";
    answerBubble.append(answerLabel, answerPre);
    const meta = document.createElement("div");
    meta.className = "bubble system";
    meta.textContent = `Generated locally from ${checkpointPath || "selected checkpoint"} · no external service involved.`;
    body.append(promptBubble, answerBubble, meta);
    $("copyButton").disabled = !outputText || isError;
  }
  const seedPayload = () => {
    // Playground-only randomness: random by default, locked seed for
    // reproducible tests and fair A/B. Training/eval seeds are untouched.
    if (!$("seedRandom") || $("seedRandom").checked) return "random";
    return $("seedValue") ? $("seedValue").value : "random";
  };
  const stripReplyFor = (sendPrompt, output) => {
    const text = output || "";
    for (const prefix of [sendPrompt, sendPrompt.replace(/\s+$/, "")]) {
      if (prefix && text.startsWith(prefix)) {
        const rest = text.slice(prefix.length).trimStart();
        if (rest) return rest;
      }
    }
    const re = /\nAssistant:(?:\n| )/g;
    let last = -1, mlen = 0, m;
    while ((m = re.exec(text)) !== null) { last = m.index; mlen = m[0].length; }
    if (last >= 0) {
      const rest = text.slice(last + mlen).trimStart();
      if (rest) return rest;
    }
    return text || "No output returned.";
  };
  function setGenerateBusy(busy) {
    state._genBusy = busy;
    $("generateButton").disabled = busy;
    if ($("regenerateButton")) $("regenerateButton").disabled = busy;
    if ($("compareButton")) $("compareButton").disabled = busy;
    if ($("cancelButton")) $("cancelButton").hidden = !busy;
    $("clearChatButton").disabled = busy;
    $("copyButton").disabled = busy;
    $("generationIndicator").hidden = !busy;
    if (busy) $("outputStatus").textContent = "WORKING";
    updateRegenerateState();
  }
  let activeGenJobs = [], genStartAt = 0, genElapsedTimer = null, genCancelled = false;
  function genElapsed() { return Math.max(0, Math.round((Date.now() - genStartAt) / 1000)); }
  function stopElapsedTimer() { if (genElapsedTimer) { clearInterval(genElapsedTimer); genElapsedTimer = null; } }
  function renderPartial(mode, promptText, partialText, key) {
    // Live streaming view only — history is mutated solely on success.
    if (mode === "chat") {
      renderConversation(key);
      if (partialText) $("outputBody").append(bubble("assistant", "Assistant · local model (streaming…)", partialText));
    } else {
      renderOutput(mode, promptText, (partialText || "") + " …", key, false);
      const note = document.createElement("div");
      note.className = "bubble system";
      note.textContent = "Streaming…";
      $("outputBody").append(note);
    }
  }
  async function cancelGeneration() {
    genCancelled = true;
    stopElapsedTimer();
    const jobs = activeGenJobs.splice(0);
    await Promise.all(jobs.map((id) => post("/api/generate/cancel", { job_id: id }).catch(() => null)));
    setGenerateBusy(false);
    $("outputStatus").textContent = "LOCAL";
    const key = $("playCheckpoint").value;
    if ($("playMode").value === "chat") renderConversation(key);
    if (!$("comparePanel").hidden) { $("compareMeta").textContent += " Cancelled — history untouched."; }
    else $("generateNote").textContent = "Cancelled — conversation history unchanged.";
    updatePlaygroundState();
  }
  function downloadJson(filename, obj) {
    try {
      const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      document.body.append(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
    } catch { /* download unsupported: results stay visible in the panel */ }
  }
  async function generate(regenMode = "new") {
    if ($("generateButton").disabled) return;
    const mode = $("playMode").value;
    const key = $("playCheckpoint").value;
    const latest = $("prompt").value;
    const newTokens = Math.min(4096, Math.max(1, parseInt($("generationTokens").value, 10) || 160));
    const isRegen = regenMode === "regenerate";
    // Regenerate reuses the last user turn so history length never grows:
    // chat history [..., user, assistant] becomes [..., user, assistant'].
    let history = historyFor(key), regenUser = null, regenPrefix = null;
    if (isRegen && mode === "chat") {
      if (history.length < 2) return;
      regenUser = history[history.length - 2].text;
      regenPrefix = history.slice(0, -2);
    }
    const effectiveLatest = isRegen && mode === "chat" ? regenUser : latest;
    const effectiveStored = isRegen && mode === "chat" ? regenPrefix : historyFor(key);
    let sendPrompt = effectiveLatest, sendChat = mode === "chat", composeInfo = null;
    if (mode === "chat") {
      const composed = composeChatPrompt(effectiveStored, effectiveLatest, newTokens);
      sendPrompt = composed.prompt; sendChat = false; composeInfo = composed;
    }
    const noteInfo = (ok) => (mode === "chat" && composeInfo)
      ? { trimmed: composeInfo.trimmed && ok, keptEntries: composeInfo.keptEntries, totalEntries: composeInfo.totalEntries }
      : null;
    try {
      setGenerateBusy(true);
      genCancelled = false;
      genStartAt = Date.now();
      $("generationIndicator").hidden = false;
      $("generationIndicator").lastChild.textContent = "Generating locally… 0s";
      stopElapsedTimer();
      genElapsedTimer = setInterval(() => {
        const ind = $("generationIndicator");
        if (ind && !ind.hidden) ind.lastChild.textContent = `Generating locally… ${genElapsed()}s`;
      }, 1000);
      $("generateNote").textContent = isRegen ? "Regenerating locally — replacing last reply…" : "Generating locally…";
      if (mode === "chat") { renderConversation(key); $("outputBody").append(skeletonBubble()); }
      else $("outputBody").replaceChildren(skeletonBubble());
      const job = await post("/api/generate", { checkpoint: key, prompt: sendPrompt, chat: sendChat, temperature: $("temperature").value, top_k: $("topK").value, tokens: $("generationTokens").value, seed: seedPayload() });
      if (job.seed != null) { state.lastSeed = job.seed; }
      activeGenJobs = [job.job_id];
      let shownPartial = "";
      const poll = async () => {
        if (genCancelled) return;
        const result = await json(`/api/jobs/${job.job_id}`);
        if (genCancelled) return;
        if (result.state === "running") {
          // Stream the backend's incremental output without touching history.
          const partial = result.output && result.output !== shownPartial ? stripReplyFor(sendPrompt, result.output) : null;
          if (partial && partial !== shownPartial) {
            shownPartial = partial;
            renderPartial(mode, effectiveLatest, partial, key);
          }
          return setTimeout(poll, 700);
        }
        const failed = result.state !== "done";
        const reply = stripReplyFor(sendPrompt, result.output);
        if (mode === "chat" && !failed) {
          const hist = historyFor(key);
          if (isRegen) {
            // Replace, never append: history length unchanged.
            hist[hist.length - 1] = { role: "assistant", text: reply };
          } else {
            hist.push({ role: "user", text: effectiveLatest }, { role: "assistant", text: reply });
            while (hist.length > CHAT_MAX_TURNS * 2) hist.splice(0, 2);
          }
          renderConversation(key);
        } else if (mode === "chat") {
          renderConversation(key, result.output);
        } else {
          // Story mode is stateless; Regenerate simply replaces the display.
          renderOutput(mode, effectiveLatest, result.output, key, failed);
        }
        activeGenJobs = [];
        stopElapsedTimer();
        if (genCancelled) return;
        setGenerateBusy(false);
        $("outputStatus").textContent = failed ? "ERROR" : "LOCAL";
        $("generateNote").textContent = failed ? `Generation failed (${genElapsed()}s).`
          : (state.lastSeed != null ? `${isRegen ? "Regenerated" : "Generation complete"} · seed ${state.lastSeed} · ${genElapsed()}s.` : `Generation complete · ${genElapsed()}s.`);
        updateChatNote(noteInfo(!failed));
        updatePlaygroundState();
        savePrefs();
      };
      poll();
    } catch (error) {
      activeGenJobs = [];
      stopElapsedTimer();
      if (genCancelled) return;
      if (mode === "chat") renderConversation(key, error.message);
      else renderOutput(mode, effectiveLatest, error.message, key, true);
      setGenerateBusy(false);
      $("outputStatus").textContent = "ERROR";
      $("generateNote").textContent = /active/i.test(error.message || "") ? error.message : "Generation could not start.";
    }
  }
  async function regenerate() {
    const mode = $("playMode").value;
    if (mode !== "chat") { await generate("regenerate"); return; }
    const key = $("playCheckpoint").value;
    if (historyFor(key).length < 2) return;
    await generate("regenerate");
  }
  async function compareCurrentVsNewer() {
    const head = headEntry();
    const keyA = $("playCheckpoint").value;
    if (!head || !head.checkpoint || head.checkpoint === keyA) return;
    const keyB = head.checkpoint;
    const mode = $("playMode").value;
    const latest = $("prompt").value;
    if (!latest.trim()) { $("generateNote").textContent = "Enter a prompt to compare."; return; }
    const newTokens = Math.min(4096, Math.max(1, parseInt($("generationTokens").value, 10) || 160));
    // Fair A/B: one identical prompt, seed, temperature, top-k and budget to
    // both checkpoints. History is never touched here.
    const seed = ($("seedRandom") && $("seedRandom").checked)
      ? String(Math.floor(Math.random() * 2147483647))
      : String(($("seedValue") && $("seedValue").value) || "42");
    let sendPromptA = latest, sendChatA = mode === "chat";
    if (mode === "chat") {
      const composed = composeChatPrompt(historyFor(keyA), latest, newTokens);
      sendPromptA = composed.prompt; sendChatA = false;
    }
    // Compare uses the same composed prompt for both sides so only weights differ.
    const approved = approvedPath();
    const bTag = keyB === approved ? "Approved default" : "Experimental — unapproved";
    const panel = $("comparePanel"), meta = $("compareMeta");
    panel.hidden = false;
    meta.textContent = `A=${keyA} B=${keyB} · seed ${seed} · ${mode} · temp ${$("temperature").value} top_k ${$("topK").value} tokens ${$("generationTokens").value}`;
    $("compareA").innerHTML = ""; $("compareB").innerHTML = "";
    $("compareA").append(skeletonBubble().cloneNode(true));
    $("compareB").append(skeletonBubble().cloneNode(true));
    setGenerateBusy(true);
    genCancelled = false;
    genStartAt = Date.now();
    stopElapsedTimer();
    genElapsedTimer = setInterval(() => {
      if (!genCancelled) meta.textContent = `A=${keyA} B=${keyB} · seed ${seed} · comparing… ${genElapsed()}s`;
    }, 1000);
    $("generateNote").textContent = `Comparing with identical seed ${seed} — history untouched…`;
    const paintPartial = (el, label, text) => {
      el.replaceChildren();
      const span = document.createElement("span"); span.className = "bubble-label"; span.textContent = label;
      const pre = document.createElement("pre"); pre.textContent = (text || "") + " …";
      el.append(span, pre);
    };
    try {
      const [jobA, jobB] = await Promise.all([
        post("/api/generate", { checkpoint: keyA, prompt: sendPromptA, chat: sendChatA, temperature: $("temperature").value, top_k: $("topK").value, tokens: $("generationTokens").value, seed }),
        post("/api/generate", { checkpoint: keyB, prompt: sendPromptA, chat: sendChatA, temperature: $("temperature").value, top_k: $("topK").value, tokens: $("generationTokens").value, seed }),
      ]);
      activeGenJobs = [jobA.job_id, jobB.job_id];
      const labelA = `A · current · ${keyA.split(/[/\\]/).slice(-3).join("/")}`, labelB = `B · ${head.run} (${bTag}) · ${keyB.split(/[/\\]/).slice(-3).join("/")}`;
      const waitJob = (job, el, label) => (async () => {
        for (;;) {
          if (genCancelled) return { state: "cancelled", output: "" };
          const r = await json(`/api/jobs/${job.job_id}`);
          if (genCancelled) return { state: "cancelled", output: "" };
          if (r.state !== "running") return r;
          if (r.output) paintPartial(el, label + " (streaming…)", stripReplyFor(sendPromptA, r.output));
          await new Promise((res) => setTimeout(res, 700));
        }
      })();
      const [resA, resB] = await Promise.all([waitJob(jobA, $("compareA"), labelA), waitJob(jobB, $("compareB"), labelB)]);
      if (genCancelled) return;
      const fill = (el, label, res) => {
        el.replaceChildren();
        const span = document.createElement("span"); span.className = "bubble-label"; span.textContent = label;
        const pre = document.createElement("pre"); pre.textContent = res.state === "done" ? stripReplyFor(sendPromptA, res.output) : (res.output || "Error");
        const metaLine = document.createElement("div"); metaLine.className = "gen-meta"; metaLine.textContent = `seed ${seed}`;
        el.classList.toggle("error-bubble", res.state !== "done");
        el.append(span, pre, metaLine);
      };
      fill($("compareA"), labelA, resA);
      fill($("compareB"), labelB, resB);
      state.lastSeed = seed;
      state.lastCompare = {
        prompt: latest, seed, mode, temperature: $("temperature").value,
        top_k: $("topK").value, tokens: $("generationTokens").value,
        a: { checkpoint: keyA, reply: resA.state === "done" ? stripReplyFor(sendPromptA, resA.output) : null, error: resA.state === "done" ? null : resA.output },
        b: { checkpoint: keyB, reply: resB.state === "done" ? stripReplyFor(sendPromptA, resB.output) : null, error: resB.state === "done" ? null : resB.output, tag: bTag },
      };
      const ok = resA.state === "done" && resB.state === "done";
      $("outputStatus").textContent = ok ? "LOCAL" : (resA.state === "cancelled" || resB.state === "cancelled" ? "LOCAL" : "ERROR");
      $("generateNote").textContent = ok ? `Compare complete · seed ${seed} · ${genElapsed()}s · history untouched.`
        : `Compare finished with errors · seed ${seed} · history untouched.`;
    } catch (error) {
      if (genCancelled) return;
      meta.textContent = `Compare failed: ${error.message} · history untouched.`;
      $("outputStatus").textContent = "ERROR";
    } finally {
      activeGenJobs = [];
      stopElapsedTimer();
      if (genCancelled) return;
      setGenerateBusy(false);
      updatePlaygroundState();
    }
  }
  function clearChat() {
    chatHistories.set($("playCheckpoint").value, []);
    renderConversation($("playCheckpoint").value);
    updateChatNote();
    $("generateNote").textContent = "Chat cleared. Start a new conversation below.";
  }
  function wireSelectFilter(inputId, selectId) {
    const input = $(inputId), select = $(selectId);
    if (!input || !select) return;
    const apply = () => {
      const q = (input.value || "").trim().toLowerCase();
      [...select.querySelectorAll("option")].forEach((opt) => {
        opt.hidden = !!q && !(opt.textContent || "").toLowerCase().includes(q);
      });
    };
    input.addEventListener("input", apply);
    select._applyFilter = apply;
  }
  function applySelectFilters() {
    ["runName", "parentCheckpoint", "playCheckpoint"].forEach((id) => {
      const select = $(id);
      if (select && select._applyFilter) select._applyFilter();
    });
  }
  async function downloadMetrics() {
    const name = $("runName").value;
    if (!name) return;
    try {
      const result = await json(`/api/metrics?run=${encodeURIComponent(name)}`);
      downloadJson(`${name}-metrics.json`, result);
      showNotice(`Metrics for '${name}' downloaded.`);
    } catch (error) { showNotice(error.message, true); }
  }
  function downloadComparison() {
    if (!state.lastCompare) { $("generateNote").textContent = "Run a comparison first, then download it."; return; }
    downloadJson(`compare-seed-${state.lastCompare.seed}.json`, state.lastCompare);
    $("generateNote").textContent = "Comparison downloaded as JSON.";
  }

  function updatePromptCount() {
    const el = $("promptCount");
    if (!el) return;
    const len = ($("prompt").value || "").length;
    el.textContent = len ? `${new Intl.NumberFormat("en-US").format(len)} chars` : "—";
  }
  function autogrowPrompt() {
    const ta = $("prompt");
    ta.style.height = "auto";
    ta.style.height = `${Math.min(320, Math.max(110, ta.scrollHeight))}px`;
  }
  function selectTab(name, focus = false) {
    document.querySelectorAll(".tab").forEach((tab) => {
      const on = tab.dataset.tab === name;
      tab.classList.toggle("active", on);
      tab.setAttribute("aria-selected", on ? "true" : "false");
      if (on && focus) tab.focus();
    });
    const views = { training: "trainingView", chat: "chatView", research: "researchView", versions: "versionsView" };
    Object.entries(views).forEach(([key, id]) => {
      const el = $(id);
      if (!el) return;
      const on = key === name;
      el.classList.toggle("active-view", on);
      el.hidden = !on;
    });
    if (name === "versions") refreshVersions();
    if (name === "research") refreshResearch();
  }

  $("runName").addEventListener("change", () => { state.followActive = !!state.activeRun && $("runName").value === state.activeRun; setSelectedCheckpointForRun(); updateSelectedContext(); updateTelemetry({ runs: state.runs, gpu: {}, disk: {} }); state.metricsFor = null; loadMetrics(); refreshPlan(); savePrefs(); });
  $("parentCheckpoint").addEventListener("change", () => { updateSelectedContext(); refreshPlan(); savePrefs(); });
  ["mode", "dataset", "additionalTokens", "duration", "durationUnit", "untilStopped", "mixture", "stageName"].forEach((id) => $(id).addEventListener("input", () => { refreshPlan(); savePrefs(); }));
  $("untilStopped").addEventListener("change", () => { refreshPlan(); savePrefs(); });
  document.querySelectorAll("[data-tokens]").forEach((button) => button.addEventListener("click", () => { $("additionalTokens").value = button.dataset.tokens; refreshPlan(); savePrefs(); }));
  document.querySelectorAll("[data-auto-tokens]").forEach((button) => button.addEventListener("click", () => { $("autoTokens").value = button.dataset.autoTokens; refreshAutoPlan(true); savePrefs(); }));
  ["autoTokens", "autoDuration", "autoDurationUnit", "autoPassPolicy", "autoStageCount"].forEach((id) => $(id).addEventListener("input", () => { refreshAutoPlan(); savePrefs(); }));
  $("autoStartButton").addEventListener("click", autoStart);
  $("autoStopButton").addEventListener("click", autoStop);
  $("autoResumeButton").addEventListener("click", autoResume);
  $("startButton").addEventListener("click", startTraining);
  $("stopButton").addEventListener("click", stopTraining);
  $("sessionStopButton").addEventListener("click", stopTraining);
  $("followActiveButton").addEventListener("click", () => { if (state.activeRun) { state.followActive = true; $("runName").value = state.activeRun; setSelectedCheckpointForRun(); updateSelectedContext(); loadMetrics(); refreshPlan(); savePrefs(); } });
  $("continueButton").addEventListener("click", continueTraining);
  $("generateButton").addEventListener("click", () => generate("new"));
  $("chatSendButton").addEventListener("click", () => sendChat(false));
  $("chatStopButton").addEventListener("click", cancelChat);
  $("chatRegenButton").addEventListener("click", () => sendChat(true));
  $("chatClearButton").addEventListener("click", clearChatView);
  $("chatCopyButton").addEventListener("click", async () => {
    const text = chat.messages.filter((m) => m.role === "assistant").map((m) => m.text).join("\n");
    if (!text) return;
    try { await navigator.clipboard.writeText(text); $("chatStatus").textContent = "Response copied to clipboard."; }
    catch { $("chatStatus").textContent = "Copy failed — select the text manually."; }
  });
  $("chatInput").addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); sendChat(false); } });
  $("chatSaveToggle").addEventListener("change", () => { if ($("chatSaveToggle").checked) saveChatConversation(true); });
  if ($("feedbackSubmitButton")) $("feedbackSubmitButton").addEventListener("click", submitFeedback);
  if ($("chatSeedRandom")) $("chatSeedRandom").addEventListener("change", savePrefs);
  $("researchStartButton").addEventListener("click", startResearch);
  $("researchPreviewButton").addEventListener("click", previewResearch);
  $("researchResumeButton").addEventListener("click", resumeResearch);
  $("researchStopButton").addEventListener("click", stopResearch);
  $("researchReportButton").addEventListener("click", () => {
    const s = researchState.session;
    if (!s) return;
    const blob = new Blob([researchState.reportText || "No report yet."], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${s.session_id}.report.md`;
    document.body.append(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  });
  $("researchJsonButton").addEventListener("click", async () => {
    const s = researchState.session;
    if (!s) return;
    try {
      const full = await json(`/api/research/${encodeURIComponent(s.session_id)}`);
      downloadJson(`${s.session_id}.json`, full);
    } catch (error) { $("researchNote").textContent = error.message; }
  });
  $("providerSaveButton").addEventListener("click", async () => {
    try {
      await post("/api/research/provider", {
        enabled: $("providerEnabled").value === "on",
        base_url: $("providerBaseUrl").value,
        model: $("providerModel").value,
        fallback_models: ($("providerFallbacks").value || "").split("\n").map(x => x.trim()).filter(Boolean),
        allow_external_eval: $("providerAllowEval").value === "on",
        max_api_requests: parseInt($("providerMaxRequests").value, 10) || 25,
      });
      $("providerNote").textContent = "Provider settings saved (key handled separately).";
      await refreshResearch();
    } catch (error) { $("providerNote").textContent = error.message; }
  });
  $("providerKeyButton").addEventListener("click", async () => {
    try {
      await post("/api/research/provider/key", { key: $("providerKey").value });
      $("providerKey").value = "";
      $("providerNote").textContent = "API key stored locally. It is never sent to the browser again.";
      await refreshResearch();
    } catch (error) { $("providerNote").textContent = error.message; }
  });
  $("providerKeyClearButton").addEventListener("click", async () => {
    await post("/api/research/provider/key/clear", {});
    $("providerNote").textContent = "API key removed.";
    await refreshResearch();
  });
  $("providerTestButton").addEventListener("click", async () => {
    try {
      $("providerNote").textContent = "Testing connection…";
      const res = await post("/api/research/provider/test", {});
      $("providerNote").textContent = `Connected: ${res.model_count} model(s) listed.` + (res.configured_model_listed === false ? " Configured model NOT found — check the identifier." : "");
    } catch (error) { $("providerNote").textContent = error.message; }
  });
  $("rollbackButton").addEventListener("click", rollbackVersion);
  if ($("versionFilter")) $("versionFilter").addEventListener("input", renderVersions);
  if ($("versionShowArchived")) $("versionShowArchived").addEventListener("change", renderVersions);
  $("evalCompareButton").addEventListener("click", runEvalCompare);
  $("evalDownloadButton").addEventListener("click", () => {
    if (versionState.evalResult) downloadJson("eval-scorecard.json", versionState.evalResult);
  });
  $("promoteButton").addEventListener("click", promoteCandidate);
  if ($("cancelButton")) $("cancelButton").addEventListener("click", cancelGeneration);
  if ($("metricsDownloadButton")) $("metricsDownloadButton").addEventListener("click", downloadMetrics);
  if ($("compareDownloadButton")) $("compareDownloadButton").addEventListener("click", downloadComparison);
  wireSelectFilter("runNameFilter", "runName");
  wireSelectFilter("parentCheckpointFilter", "parentCheckpoint");
  wireSelectFilter("playCheckpointFilter", "playCheckpoint");
  if ($("regenerateButton")) $("regenerateButton").addEventListener("click", regenerate);
  if ($("compareButton")) $("compareButton").addEventListener("click", compareCurrentVsNewer);
  if ($("compareCloseButton")) $("compareCloseButton").addEventListener("click", () => { $("comparePanel").hidden = true; });
  if ($("seedRandom")) $("seedRandom").addEventListener("change", () => { updatePlaygroundState(); savePrefs(); });
  if ($("seedValue")) $("seedValue").addEventListener("input", () => { savePrefs(); });
  if ($("assistantSwitchButton")) $("assistantSwitchButton").addEventListener("click", () => {
    const head = headEntry();
    if (!head || !head.checkpoint) return;
    // Explicit Switch only — never auto-switch mid-chat. Old checkpoint
    // history stays in chatHistories under its own key.
    $("playCheckpoint").value = head.checkpoint;
    state.dismissedHeadPath = null;
    updatePlaygroundState();
    renderConversation(head.checkpoint);
    updateChatNote();
    savePrefs();
    const switchedTag = head.checkpoint === approvedPath() ? "Approved default" : "Experimental — unapproved";
    $("generateNote").textContent = `Switched to ${head.run} (${switchedTag}). Previous conversation kept under its own checkpoint.`;
  });
  if ($("assistantCompareButton")) $("assistantCompareButton").addEventListener("click", compareCurrentVsNewer);
  if ($("assistantDismissButton")) $("assistantDismissButton").addEventListener("click", () => {
    const head = headEntry();
    state.dismissedHeadPath = head ? head.checkpoint : null;
    updateAssistantBanner();
  });
  $("playCheckpoint").addEventListener("change", () => { updatePlaygroundState(); renderConversation($("playCheckpoint").value); updateChatNote(); savePrefs(); });
  // Short replies suit chat; longer continuations suit story mode. Switching
  // modes restores the other default only when the user never customized it.
  const MODE_TOKEN_DEFAULTS = { story: "160", chat: "60" };
  let lastPlayMode = $("playMode").value;
  $("playMode").addEventListener("change", () => {
    const next = $("playMode").value, tokens = $("generationTokens");
    if (tokens.value === (MODE_TOKEN_DEFAULTS[lastPlayMode] || "")) tokens.value = MODE_TOKEN_DEFAULTS[next] || tokens.value;
    lastPlayMode = next;
    renderConversation($("playCheckpoint").value); updateChatNote(); savePrefs();
  });
  $("clearChatButton").addEventListener("click", clearChat);
  $("prompt").addEventListener("input", () => { updatePromptCount(); autogrowPrompt(); });
  $("prompt").addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); generate("new"); } });
  let copyTimer;
  $("copyButton").addEventListener("click", async () => {
    const text = [...$("outputBody").querySelectorAll(".assistant pre")].map((el) => el.textContent).join("\n");
    if (!text) return;
    try { await navigator.clipboard.writeText(text); $("generateNote").textContent = "Response copied to clipboard."; }
    catch { $("generateNote").textContent = "Copy failed — select the text manually."; }
    const btn = $("copyButton"), original = "Copy";
    btn.textContent = "Copied";
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { btn.textContent = original; }, 1600);
  });
  $("dataset").addEventListener("change", () => { $("mixture").closest(".field").style.opacity = $("dataset").value === "mixture" ? "1" : ".62"; refreshPlan(); savePrefs(); });
  document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => selectTab(button.dataset.tab)));
  document.querySelector(".tabs").addEventListener("keydown", (event) => {
    const tabs = [...document.querySelectorAll(".tab")];
    const index = tabs.findIndex((t) => t.classList.contains("active"));
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      let next = index;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      selectTab(tabs[next].dataset.tab, true);
    }
  });
  if (!$("stageName").value) $("stageName").value = `web-stage-${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}`;
  // ---- ONE AdamLM Chat: a single approved assistant, server-side prompt. ----
  // The browser never picks a checkpoint here: /api/chat always resolves the
  // approved pointer, composes the exact SFT prompt with a 512-token budget,
  // and reports honest kept/total turn accounting.
  const chat = { messages: [], busy: false, jobId: null, cancelled: false, startAt: 0, timer: null, lastSeed: null, context: null };
  function chatSeedPayload() {
    if (!$("chatSeedRandom") || $("chatSeedRandom").checked) return "random";
    return $("chatSeedValue") ? $("chatSeedValue").value : "random";
  }
  function renderChat(errorText = null) {
    const box = $("chatMessages");
    box.replaceChildren();
    if (!chat.messages.length && errorText == null) {
      const div = document.createElement("div");
      div.className = "bubble system";
      div.textContent = "Say hello to AdamLM. Replies come from the approved local model — nothing leaves this PC.";
      box.append(div);
      $("chatCopyButton").disabled = true;
      $("chatRegenButton").disabled = true;
      return;
    }
    chat.messages.forEach((m) => {
      const div = document.createElement("div");
      div.className = m.role === "user" ? "bubble user" : "bubble assistant";
      const span = document.createElement("span");
      span.className = "bubble-label";
      span.textContent = m.role === "user" ? "You" : "AdamLM";
      const pre = document.createElement("pre");
      pre.textContent = m.text;
      div.append(span, pre);
      box.append(div);
    });
    if (errorText != null) {
      const div = document.createElement("div");
      div.className = "bubble assistant error-bubble";
      const span = document.createElement("span");
      span.className = "bubble-label";
      span.textContent = "Error";
      const pre = document.createElement("pre");
      pre.textContent = errorText;
      div.append(span, pre);
      box.append(div);
    }
    box.scrollTop = box.scrollHeight;
    $("chatCopyButton").disabled = !chat.messages.some((m) => m.role === "assistant");
    const lastAssistant = [...chat.messages].reverse().find((m) => m.role === "assistant");
    $("chatRegenButton").disabled = chat.busy || !lastAssistant;
  }
  function setChatBusy(busy, note) {
    chat.busy = busy;
    $("chatSendButton").disabled = busy;
    $("chatStopButton").hidden = !busy;
    $("chatClearButton").disabled = busy;
    $("chatIndicator").hidden = !busy;
    if (busy) {
      chat.startAt = Date.now();
      $("chatIndicator").lastChild.textContent = "AdamLM is thinking… 0s";
      clearInterval(chat.timer);
      chat.timer = setInterval(() => {
        const s = Math.max(0, Math.round((Date.now() - chat.startAt) / 1000));
        const ind = $("chatIndicator");
        if (ind && !ind.hidden) ind.lastChild.textContent = `AdamLM is thinking… ${s}s`;
      }, 1000);
    } else {
      clearInterval(chat.timer);
    }
    if (note != null) $("chatStatus").textContent = note;
    renderChatButtonsOnly();
  }
  function renderChatButtonsOnly() {
    const lastAssistant = [...chat.messages].reverse().find((m) => m.role === "assistant");
    if (!chat.busy) $("chatRegenButton").disabled = !lastAssistant;
    $("chatCopyButton").disabled = !chat.messages.some((m) => m.role === "assistant");
  }
  function updateChatContextNote(ctx = null) {
    if (ctx) chat.context = ctx;
    const el = $("chatContextNote");
    const total = chat.messages.length;
    if (!total) { el.textContent = "New conversation — earlier turns will be included as you chat."; return; }
    if (chat.context && chat.context.total_turns != null) {
      const { kept_turns, total_turns, trimmed } = chat.context;
      el.textContent = (trimmed || kept_turns < total_turns)
        ? `Using ${kept_turns} of ${total_turns} turns · older turns trimmed to fit the 512-token context.`
        : `Using ${kept_turns} of ${total_turns} turns · full history is included with each reply.`;
    } else {
      el.textContent = `${total} message${total === 1 ? "" : "s"} in this conversation.`;
    }
  }
  async function sendChat(regen = false) {
    if (chat.busy) return;
    if (state.activeRun) { $("chatStatus").textContent = `Training '${state.activeRun}' is active — chat is paused so training keeps the GPU.`; return; }
    let outgoing = chat.messages;
    if (regen) {
      const idx = [...chat.messages].map((m) => m.role).lastIndexOf("assistant");
      if (idx < 0) return;
      outgoing = chat.messages.slice(0, idx);
      if (!outgoing.length || outgoing[outgoing.length - 1].role !== "user") return;
    } else {
      const text = ($("chatInput").value || "").trim();
      if (!text) return;
      outgoing = [...chat.messages, { role: "user", text }];
      $("chatInput").value = "";
    }
    chat.cancelled = false;
    setChatBusy(true, regen ? "Regenerating — replacing last reply…" : "AdamLM is thinking…");
    renderChat();
    const skel = document.createElement("div");
    skel.className = "skeleton";
    ["92%", "78%", "85%"].forEach((w) => { const i = document.createElement("i"); i.style.width = w; skel.append(i); });
    $("chatMessages").append(skel);
    try {
      const job = await post("/api/chat", {
        messages: outgoing, tokens: $("chatTokens").value,
        temperature: $("chatTemperature").value, top_k: $("chatTopK").value, seed: chatSeedPayload(),
      });
      if (job.seed != null) { chat.lastSeed = job.seed; state.lastSeed = job.seed; }
      if (job.kept_turns != null) updateChatContextNote({ kept_turns: job.kept_turns, total_turns: job.total_turns, trimmed: job.trimmed });
      chat.jobId = job.job_id;
      const poll = async () => {
        if (chat.cancelled) return;
        const result = await json(`/api/jobs/${job.job_id}`);
        if (chat.cancelled) return;
        if (result.state === "running") {
          if (result.output) {
            skel.replaceChildren();
            const pre = document.createElement("pre");
            pre.textContent = result.output;
            skel.append(pre);
          }
          return setTimeout(poll, 700);
        }
        const failed = result.state !== "done";
        if (!failed) {
          const reply = (result.output || "").trim() || "No output returned.";
          if (regen) chat.messages = [...outgoing, { role: "assistant", text: reply }];
          else chat.messages = [...outgoing, { role: "assistant", text: reply }];
          const s = Math.max(0, Math.round((Date.now() - chat.startAt) / 1000));
          setChatBusy(false, chat.lastSeed != null ? `Reply in ${s}s · seed ${chat.lastSeed}.` : `Reply in ${s}s.`);
          renderChat();
          updateChatContextNote();
          if ($("chatSaveToggle").checked) saveChatConversation(true);
        } else {
          const msg = result.output || "Generation failed.";
          const next = /busy|active/i.test(msg)
            ? " Next step: stop the training session first, then resend."
            : " Next step: retry, or lower New tokens and resend.";
          setChatBusy(false, "Reply failed." + next);
          renderChat(msg + next);
        }
      };
      poll();
    } catch (error) {
      if (chat.cancelled) return;
      setChatBusy(false, /active|busy/i.test(error.message || "") ? error.message : "Chat could not start.");
      renderChat(error.message);
    }
  }
  async function cancelChat() {
    chat.cancelled = true;
    clearInterval(chat.timer);
    if (chat.jobId) await post("/api/generate/cancel", { job_id: chat.jobId }).catch(() => null);
    chat.jobId = null;
    setChatBusy(false, "Stopped — conversation unchanged.");
    renderChat();
  }
  function clearChatView() {
    chat.messages = [];
    chat.context = null;
    renderChat();
    updateChatContextNote();
    $("chatStatus").textContent = "New conversation started.";
  }
  async function saveChatConversation(quiet = false) {
    if (!chat.messages.length) { if (!quiet) $("chatStatus").textContent = "Nothing to save yet."; return; }
    try {
      const res = await post("/api/conversations", {
        messages: chat.messages,
        title: (chat.messages[0]?.text || "chat").slice(0, 60),
        approved_version: state.assistantChat?.version || null,
      });
      if (!quiet) $("chatStatus").textContent = `Conversation saved on this PC (${res.message_count} messages). Saved history is never training data.`;
    } catch (error) { if (!quiet) $("chatStatus").textContent = error.message; }
  }
  async function submitFeedback() {
    const correction = ($("feedbackCorrection").value || "").trim();
    if (!correction) { $("feedbackNote").textContent = "Write the corrected answer first."; return; }
    const lastAssistant = [...chat.messages].reverse().find((m) => m.role === "assistant");
    if (!lastAssistant) { $("feedbackNote").textContent = "No assistant reply to correct yet."; return; }
    try {
      await post("/api/feedback", { assistant_text: lastAssistant.text, corrected_text: correction });
      $("feedbackNote").textContent = "Saved as a pending candidate. Approve it in Versions → Feedback to make it training-eligible.";
      $("feedbackCorrection").value = "";
      refreshFeedback();
    } catch (error) { $("feedbackNote").textContent = error.message; }
  }
  function updateChatIdentity() {
    const info = state.assistantChat;
    if (!info || !info.approved) {
      $("chatVersion").textContent = "no approved version";
      return;
    }
    $("chatVersion").textContent = `Version ${info.version} · Approved`;
    const active = state.activeRun;
    const gpuBusy = state.gpuSession && state.gpuSession.busy;
    $("chatActivity").textContent = active ? `training: ${active}` : (gpuBusy ? `${state.gpuSession.kind} running` : "idle · local model");
    const head = state.assistant && state.assistant.head;
    const showCandidate = !!(head && head.checkpoint && info.checkpoint && head.checkpoint !== info.checkpoint);
    $("chatCandidate").hidden = !showCandidate;
    if (showCandidate) $("chatCandidate").textContent = `new candidate available (${head.run})`;
    $("chatSendButton").disabled = chat.busy || !!active;
    if (active) $("chatSendButton").title = `Training '${active}' is active — chat paused to protect GPU memory.`;
    else $("chatSendButton").title = "";
  }

  // ---- Research sessions (bounded AFK loop; local-first, AI optional). ----
  const researchState = { session: null, reportText: "", lastEval: null };
  const researchSeconds = () => {
    const value = parseFloat($("researchDuration").value);
    if (!Number.isFinite(value) || value <= 0) return null;
    return value * ($("researchDurationUnit").value === "hours" ? 3600 : 60);
  };
  async function refreshResearch() {
    try {
      const data = await json("/api/research/latest");
      researchState.session = data.status === "none" ? null : data;
      renderResearch();
    } catch { /* backend unreachable: leave last state */ }
    try {
      const provider = await json("/api/research/provider");
      if (document.activeElement !== $("providerBaseUrl")) $("providerBaseUrl").value = provider.base_url || "";
      if (document.activeElement !== $("providerModel")) $("providerModel").value = provider.model || "";
      if (document.activeElement !== $("providerFallbacks")) $("providerFallbacks").value = (provider.fallback_models || []).join("\n");
      $("providerEnabled").value = provider.enabled ? "on" : "off";
      $("providerAllowEval").value = provider.allow_external_eval ? "on" : "off";
      $("providerMaxRequests").value = provider.max_api_requests ?? 25;
      const keyLine = provider.key_configured ? "A key is stored (never displayed)." : "No key stored.";
      const checkedLine = (provider.checked || []).map(c =>
        `${c.model} · ${c.listed ? "listed" : "NOT LISTED"} · ${c.free ? "free" : "NOT FREE"}`
      ).join(" | ") || "—";
      $("providerKeyState").textContent = `${keyLine}  Checked: ${checkedLine}`;
    } catch { /* ignore */ }
  }
  function renderResearch() {
    const s = researchState.session;
    const pill = $("researchPill"), status = $("researchStatus");
    const box = $("researchActual");
    if (!s) {
      pill.textContent = "NO SESSION"; pill.className = "session-pill";
      status.textContent = "NO SESSION";
      $("researchTitle").textContent = "No research session yet.";
      $("researchPath").textContent = "Preview a first decision, then start when you explicitly choose to.";
      box.hidden = true; box.replaceChildren();
      $("researchStopButton").disabled = true;
      $("researchResumeButton").disabled = true;
      $("researchReportButton").disabled = true;
      $("researchJsonButton").disabled = true;
      return;
    }
    const label = { running: "RUNNING", stopped: "STOPPED", failed: "FAILED", done: "DONE", created: "READY", "paused-provider": "PAUSED" }[s.status] || s.status.toUpperCase();
    pill.textContent = `${label} · ${s.session_id}`;
    pill.className = `session-pill ${s.status === "running" ? "running" : (s.status === "done" ? "done" : (s.status === "failed" ? "error" : "warn"))}`;
    status.textContent = label;
    const used = s.budget_used || {}, lim = s.limits || {};
    $("researchTitle").textContent = `Session ${s.session_id} · ${label}`;
    $("researchPath").textContent = `${number(used.tokens || 0)} / ${number(lim.max_tokens || 0)} tokens · ${(s.experiments || []).length} / ${lim.max_experiments || "?"} experiments`
      + (lim.allowed_datasets ? ` · diets: ${lim.allowed_datasets.join(", ")}` : "")
      + (lim.stage_token_cap ? ` · stage cap ${number(lim.stage_token_cap)}` : "")
      + (lim.allow_repetition ? " · repetition confirmed" : "")
      + (s.stop_reason ? ` · ${s.stop_reason}` : "");
    box.hidden = false; box.replaceChildren();
    const title = document.createElement("strong");
    title.textContent = `Decision log · ${s.session_id}`;
    box.append(title);
    if (s.provider_state && s.provider_state.paused) {
      const ps = s.provider_state;
      const div = document.createElement("div");
      div.className = "eval-reason warn";
      div.innerHTML = `<strong>Provider paused:</strong> ${ps.reason || ps.kind || "exhausted"}`
        + (ps.last_model ? ` · last model: ${ps.last_model}` : "")
        + (ps.attempts && ps.attempts.length
           ? ` · tried: ${ps.attempts.map(a => `${a.model} (${a.ok ? "ok" : a.kind})`).join(", ")}` : "");
      box.append(div);
    }
    (s.experiments || []).forEach((e) => {
      const line = document.createElement("div");
      line.className = "stage-line";
      const plan = e.plan || {};
      const diet = plan.dataset ? ` · diet ${plan.dataset}${plan.mixture ? ` (${plan.mixture})` : ""}` : "";
      const alloc = plan.target_tokens != null ? ` · ${integer(plan.target_tokens)} tokens` : "";
      const supervisor = e.supervisor_model ? ` · supervisor: ${e.supervisor_model}` : " · supervisor: local heuristic";
      const attempts = e.supervisor_attempts && e.supervisor_attempts.length
        ? ` · attempts: ${e.supervisor_attempts.map(a => `${a.model} (${a.ok ? "ok" : a.kind})`).join(", ")}` : "";
      line.textContent = `Experiment ${e.index} · ${e.status || "?"}${e.tokens_trained != null ? ` · ${integer(e.tokens_trained)} trained` : ""}${diet}${alloc}${e.decision ? ` · ${e.decision}` : ""}${e.failure ? ` · FAILED: ${e.failure}` : ""}${supervisor}${attempts}`;
      box.append(line);
      if (plan.hypothesis) {
        const hyp = document.createElement("div");
        hyp.className = "timeline-meta";
        hyp.textContent = `Why: ${plan.hypothesis}`;
        box.append(hyp);
      }
      const ev = e.eval;
      if (ev && (ev.instr_f1_new != null)) {
        const score = document.createElement("div");
        score.className = "eval-reason";
        score.textContent = `instr F1 ${ev.instr_f1_old}→${ev.instr_f1_new} · conv F1 ${ev.conv_f1_old}→${ev.conv_f1_new}`
          + (ev.suite_delta != null ? ` · suite Δ ${ev.suite_delta}` : "")
          + (ev.old_validation_loss != null || ev.new_validation_loss != null ? ` · val-loss ${ev.old_validation_loss ?? "—"}→${ev.new_validation_loss ?? "—"} (reported only)` : "");
        box.append(score);
      }
    });
    const resumable = ["stopped", "paused-provider", "failed", "created"].includes(s.status);
    $("researchStopButton").disabled = s.status !== "running";
    $("researchResumeButton").disabled = !resumable || !s.session_id;
    $("researchStartButton").disabled = s.status === "running";
    $("researchReportButton").disabled = !(s.session_id);
    $("researchJsonButton").disabled = !(s.session_id);
    if (s.session_id) loadResearchReport(s.session_id);
  }
  async function loadResearchReport(sessionId) {
    try {
      const data = await json(`/api/research/report/${encodeURIComponent(sessionId)}`);
      researchState.reportText = data.report || "";
      const el = $("researchReport");
      el.replaceChildren();
      if (!data.report) { el.textContent = "Report will appear here as the session runs."; return; }
      const pre = document.createElement("pre");
      pre.textContent = data.report.slice(0, 6000);
      el.append(pre);
    } catch { /* ignore */ }
  }
  function researchForm() {
    const tokens = parseInt(String($("researchTokens").value).replace(/,/g, ""), 10);
    const seconds = researchSeconds();
    const datasets = Array.from(document.querySelectorAll('input[name="researchDataset"]:checked')).map((c) => c.value);
    const stageCap = parseInt(String($("researchStageCap").value).replace(/,/g, ""), 10);
    return {
      tokens, seconds, datasets, stageCap,
      valid: tokens > 0 && !!seconds && datasets.length > 0 && stageCap > 0,
    };
  }
  async function startResearch() {
    const form = researchForm();
    if (!form.tokens || form.tokens <= 0) { $("researchNote").textContent = "Enter a positive token budget."; return; }
    if (!form.seconds) { $("researchNote").textContent = "Enter a positive overall duration."; return; }
    if (!form.datasets.length) { $("researchNote").textContent = "Select at least one dataset."; return; }
    if (!form.stageCap || form.stageCap <= 0) { $("researchNote").textContent = "Enter a positive per-stage token cap."; return; }
    try {
      $("researchStartButton").disabled = true;
      $("researchNote").textContent = "Creating bounded session…";
      const created = await post("/api/research/sessions", {
        max_tokens: form.tokens, max_seconds: form.seconds,
        max_experiments: parseInt($("researchExperiments").value, 10) || 3,
        max_api_requests: 25, max_api_cost_usd: 0, max_repetition: 3,
        allow_external_eval: false,
        require_manual_promotion: $("researchPromotion").value !== "auto",
        allowed_ops: ["train", "evaluate", "compare"],
        use_ai: $("researchUseAi").value === "ai",
        goal: $("researchGoal").value,
        allowed_datasets: form.datasets,
        stage_token_cap: form.stageCap,
        allow_repetition: $("researchAllowRepetition").checked,
        mixture: $("researchMixture").value,
      });
      const started = await post("/api/research/start", { session_id: created.session_id });
      $("researchNote").textContent = `Session ${started.session_id} running. You can close the browser; the report will be here.`;
      await refreshResearch();
    } catch (error) { $("researchNote").textContent = error.message; }
    finally { $("researchStartButton").disabled = false; }
  }
  async function previewResearch() {
    const form = researchForm();
    const panel = $("researchPreview");
    if (!form.tokens || form.tokens <= 0) { $("researchNote").textContent = "Enter a positive token budget."; return; }
    if (!form.seconds) { $("researchNote").textContent = "Enter a positive overall duration."; return; }
    if (!form.datasets.length) { $("researchNote").textContent = "Select at least one dataset."; return; }
    try {
      $("researchNote").textContent = "Previewing first decision — nothing starts.";
      const res = await post("/api/research/preview", {
        max_tokens: form.tokens, max_seconds: form.seconds,
        max_experiments: parseInt($("researchExperiments").value, 10) || 3,
        use_ai: $("researchUseAi").value === "ai",
        goal: $("researchGoal").value,
        allowed_datasets: form.datasets,
        stage_token_cap: form.stageCap,
        allow_repetition: $("researchAllowRepetition").checked,
        mixture: $("researchMixture").value,
      });
      renderResearchPreview(res.preview);
      $("researchNote").textContent = "Preview ready — review, then Start when you explicitly choose to.";
    } catch (error) { $("researchNote").textContent = error.message; }
  }
  function renderResearchPreview(pf) {
    const panel = $("researchPreview");
    panel.hidden = false; panel.replaceChildren();
    if (!pf) { panel.textContent = "No preview available."; return; }
    const head = document.createElement("div");
    head.className = "preview-caption";
    head.textContent = `First decision preview — diets: ${(pf.limits?.allowed_datasets || []).join(", ")} · stage cap ${number(pf.limits?.stage_token_cap ?? 0)} · repetition ${pf.limits?.allow_repetition ? "CONFIRMED" : "not confirmed"} · starts nothing.`;
    panel.append(head);
    const first = pf.first_decision || {};
    const line = document.createElement("div");
    line.className = "stage-line";
    if (first.stop) {
      line.textContent = `Supervisor would decline: ${first.reason}`;
    } else {
      line.textContent = `Would propose: ${first.dataset || "?"} · ${number(first.target_tokens || 0)} tokens · parent ${(first.parent_checkpoint || "").split(/[/\\]/).slice(-3, -1).join("/") || "?"}`;
      const hyp = document.createElement("div");
      hyp.className = "timeline-meta";
      hyp.textContent = `Why: ${first.hypothesis || "—"}`;
      panel.append(line, hyp);
      const verdict = document.createElement("div");
      verdict.className = pf.first_decision_error ? "eval-reason warn" : "eval-reason";
      verdict.textContent = pf.first_decision_error
        ? `Would be rejected: ${pf.first_decision_error}`
        : "Passes session limits (dataset, stage cap, repetition rule).";
      panel.append(verdict);
      if (pf.ai_note) {
        const ai = document.createElement("div");
        ai.className = "timeline-meta";
        ai.textContent = pf.ai_note;
        panel.append(ai);
      }
      return;
    }
    panel.append(line);
  }
  async function resumeResearch() {
    const s = researchState.session;
    if (!s) return;
    try {
      $("researchResumeButton").disabled = true;
      await post("/api/research/start", { session_id: s.session_id });
      $("researchNote").textContent = `Session ${s.session_id} resumed — completed stages are never re-run.`;
      await refreshResearch();
    } catch (error) { $("researchNote").textContent = error.message; }
    finally { $("researchResumeButton").disabled = false; }
  }
  async function stopResearch() {
    const s = researchState.session;
    if (!s) return;
    try {
      await post("/api/research/stop", { session_id: s.session_id });
      $("researchNote").textContent = "Stopping safely — current work saves gracefully, completed stages are kept.";
      await refreshResearch();
    } catch (error) { $("researchNote").textContent = error.message; }
  }

  // ---- Versions: timeline, compare, promote, rollback, feedback. ----
  const versionState = { registry: null, evalResult: null };
  async function refreshVersions() {
    try {
      versionState.registry = await json("/api/versions");
      renderVersions();
    } catch { /* ignore */ }
    refreshFeedback();
    const info = state.assistantChat;
    if (info && info.checkpoint) $("evalOld").value = `${info.version} · ${info.checkpoint}`;
    const sel = $("evalNew");
    sel.replaceChildren();
    (versionState.registry?.versions || []).filter((v) => ["candidate", "experimental"].includes(v.approval) && !v.archived).slice(0, 12).forEach((v) => {
      const option = document.createElement("option");
      option.value = v.checkpoint;
      option.textContent = `${v.label}`;
      sel.append(option);
    });
  }
  function renderVersions() {
    const box = $("versionTimeline");
    box.replaceChildren();
    const reg = versionState.registry;
    if (!reg || !reg.versions) { box.textContent = "No versions indexed yet."; return; }
    const order = { approved: 0, "previous-approved": 1, candidate: 2, experimental: 3, general: 4, foundation: 5, smoke: 6 };
    const query = ($("versionFilter")?.value || "").trim().toLowerCase();
    const showArchived = !!$("versionShowArchived")?.checked;
    const matches = (v) => !query || [v.label, v.run, v.dataset, v.stage, v.approval, v.display_state].filter(Boolean).join(" ").toLowerCase().includes(query);
    const sorted = [...reg.versions].sort((a, b) => (order[a.approval] ?? 9) - (order[b.approval] ?? 9) || ((b.tokens || 0) - (a.tokens || 0)));
    const visible = sorted.filter((v) => (showArchived || !v.archived) && matches(v));
    const groups = [
      { title: "Approved assistant", keys: ["approved", "previous-approved"] },
      { title: "Candidates & experiment lineage", keys: ["candidate", "experimental"] },
      { title: "Foundations & general pretraining", keys: ["foundation", "general"] },
      { title: "Smoke tests (collapsed, never Chat candidates)", keys: ["smoke"], collapsed: true },
    ];
    groups.forEach((g) => {
      const items = visible.filter((v) => g.keys.includes(v.approval));
      if (!items.length) return;
      const wrap = document.createElement(g.collapsed ? "details" : "div");
      if (g.collapsed) wrap.className = "version-group smoke-group";
      const heading = document.createElement(g.collapsed ? "summary" : "h4");
      heading.className = "group-title";
      heading.textContent = `${g.title} · ${items.length}`;
      wrap.append(heading);
      items.forEach((v) => wrap.append(versionEntry(v, showArchived)));
      box.append(wrap);
    });
    const hiddenArchived = sorted.filter((v) => v.archived && !showArchived && matches(v)).length;
    if (hiddenArchived) {
      const more = document.createElement("div");
      more.className = "timeline-meta";
      more.textContent = `${hiddenArchived} archived version(s) hidden — files preserved on disk. Tick “Show archived” to review.`;
      box.append(more);
    }
    if (!box.children.length) box.textContent = "No versions match the filter.";
    const hasPrevious = sorted.some((v) => v.approval === "previous-approved");
    $("rollbackButton").disabled = !hasPrevious;
    $("rollbackNote").textContent = hasPrevious ? "Rollback swaps the pointer only — weights and history are untouched." : "No previous approved version recorded.";
  }
  function versionEntry(v, showArchived) {
    const div = document.createElement("div");
    div.className = `timeline-entry approval-${v.approval}`;
    const strong = document.createElement("strong");
    strong.textContent = v.label;
    const meta = document.createElement("div");
    meta.className = "timeline-meta";
    meta.textContent = `${number(v.tokens)} tokens · ${v.stage}/${v.dataset}` + (v.parent_run ? ` · from ${v.parent_run}` : (v.ancestry_verified ? "" : " · ancestry unknown")) + (v.active ? " · ACTIVE" : "") + (v.plan_id ? ` · plan ${v.plan_id}` : "") + (v.archived ? " · ARCHIVED" : "");
    div.append(strong, meta);
    const protect = ["approved", "previous-approved"].includes(v.approval);
    if (!protect) {
      const row = document.createElement("div");
      row.className = "action-row";
      const btn = document.createElement("button");
      btn.className = "button secondary small";
      btn.textContent = v.archived ? "Unarchive" : "Archive (hide)";
      btn.addEventListener("click", async () => {
        if (!v.archived && !confirm(`Hide ${v.version_id} from the default view? Files stay on disk.`)) return;
        try {
          await post("/api/versions/archive", { version_id: v.version_id, archived: !v.archived });
          await refreshVersions();
        } catch (error) { $("rollbackNote").textContent = error.message; }
      });
      row.append(btn);
      div.append(row);
    }
    return div;
  }
  async function rollbackVersion() {
    if (!confirm("Revert Chat to the previous approved version? Current weights and history stay intact.")) return;
    try {
      const res = await post("/api/assistant/rollback", { confirm: true });
      $("rollbackNote").textContent = `Rolled back to ${res.run || res.checkpoint}. New conversations use it.`;
      await refreshVersions();
      await refresh(false);
    } catch (error) { $("rollbackNote").textContent = error.message; }
  }
  async function runEvalCompare() {
    const oldPath = state.assistantChat?.checkpoint;
    const newPath = $("evalNew").value;
    if (!oldPath || !newPath) { $("evalNote").textContent = "Need an approved version and a candidate."; return; }
    try {
      $("evalStatus").textContent = "EVALUATING";
      $("evalNote").textContent = "Running fixed-seed probes on both checkpoints (CPU)…";
      const started = await post("/api/eval/compare", { old: oldPath, new: newPath, include_suite: true });
      const poll = async () => {
        const res = await json(`/api/eval/${started.eval_id}`);
        if (res.state === "running") {
          $("evalNote").textContent = "Evaluating… (identical prompts, seeds, settings)";
          return setTimeout(poll, 3000);
        }
        if (res.state !== "done") { $("evalStatus").textContent = "ERROR"; $("evalNote").textContent = res.error || "Evaluation failed."; return; }
        versionState.evalResult = res;
        renderScorecard(res.scorecard, res.examples);
        $("evalStatus").textContent = res.scorecard.promote ? "PROMOTABLE" : "HELD";
        $("evalNote").textContent = res.scorecard.reason;
        $("evalDownloadButton").disabled = false;
      };
      poll();
    } catch (error) { $("evalStatus").textContent = "ERROR"; $("evalNote").textContent = error.message; }
  }
  function renderScorecard(card, examples) {
    const box = $("evalScorecard");
    box.hidden = false;
    box.replaceChildren();
    const head = document.createElement("h4");
    head.textContent = "Evaluation scorecard";
    box.append(head);
    const grid = document.createElement("div");
    grid.className = "eval-grid";
    const f1 = (v) => (v == null || !Number.isFinite(Number(v))) ? "Not evaluated" : Number(v).toFixed(3);
    const cell = (label, value) => {
      const div = document.createElement("div");
      const span = document.createElement("span"); span.textContent = label;
      const strong = document.createElement("strong"); strong.textContent = value;
      div.append(span, strong); return div;
    };
    grid.append(cell("Instruction F1", `${f1(card.instr_f1_old)} → ${f1(card.instr_f1_new)}`));
    grid.append(cell("Conversation F1", `${f1(card.conv_f1_old)} → ${f1(card.conv_f1_new)}`));
    grid.append(cell("Capability suite", card.suite_accuracy_old == null ? "Not evaluated" : `${Number(card.suite_accuracy_old).toFixed(3)} → ${Number(card.suite_accuracy_new).toFixed(3)}`));
    grid.append(cell("Hygiene", (card.hygiene_problems || []).length === 0 ? "pass" : `${card.hygiene_problems.length} problem(s)`));
    grid.append(cell("Validation loss", (card.old_validation_loss == null && card.new_validation_loss == null) ? "Not evaluated" : `${card.old_validation_loss ?? "—"} → ${card.new_validation_loss ?? "—"}`));
    grid.append(cell("Decision", card.promote ? "PROMOTABLE" : "HELD"));
    box.append(grid);
    const reason = document.createElement("div");
    reason.className = "eval-reason";
    reason.textContent = card.reason + (card.note ? ` ${card.note}` : "");
    box.append(reason);
    const ex = $("evalExamples");
    ex.replaceChildren();
    (examples || []).slice(0, 4).forEach((e) => {
      const div = document.createElement("div");
      div.className = "example-pair";
      const q = document.createElement("div"); q.className = "example-prompt"; q.textContent = `Prompt: ${e.prompt}`;
      const a = document.createElement("div"); a.textContent = `Approved: ${JSON.stringify(e.approved_reply)}`;
      const c = document.createElement("div"); c.textContent = `Candidate: ${JSON.stringify(e.candidate_reply)}`;
      const note = document.createElement("div"); note.className = "timeline-meta"; note.textContent = "Inspect the actual replies — lexical overlap alone never decides.";
      div.append(q, a, c, note);
      ex.append(div);
    });
  }
  async function promoteCandidate() {
    const newPath = $("evalNew").value;
    const reason = ($("promoteReason").value || "").trim();
    if (!newPath) { $("promoteNote").textContent = "Pick a candidate first."; return; }
    if (!reason) { $("promoteNote").textContent = "A written reason is required."; return; }
    const card = versionState.evalResult?.scorecard;
    if (card && !card.promote && !confirm(`The comparison says HELD (${card.reason}). Promote anyway with your written reason?`)) return;
    if (!confirm("Promote this candidate to the approved AdamLM pointer? The previous version is kept for rollback.")) return;
    try {
      const evalInfo = card ? {
        instr_f1_old: card.instr_f1_old, instr_f1_new: card.instr_f1_new,
        conv_f1_old: card.conv_f1_old, conv_f1_new: card.conv_f1_new,
        hygiene_problems: card.hygiene_problems || [],
      } : {};
      const res = await post("/api/assistant/promote", { checkpoint: newPath, reason, eval: evalInfo });
      $("promoteNote").textContent = `Promoted. Previous version kept for rollback (${res.pointer?.previous?.run || "recorded"}).`;
      await refreshVersions();
      await refresh(false);
    } catch (error) { $("promoteNote").textContent = error.message; }
  }
  async function refreshFeedback() {
    try {
      const data = await json("/api/feedback?status=pending");
      const box = $("feedbackList");
      box.replaceChildren();
      const items = data.feedback || [];
      if (!items.length) { box.textContent = "No pending corrections."; return; }
      items.slice(0, 10).forEach((item) => {
        const div = document.createElement("div");
        div.className = "feedback-item";
        const span = document.createElement("span");
        span.textContent = `${item.feedback_id} · ${item.note || "correction"}`;
        const approve = document.createElement("button");
        approve.className = "button secondary small";
        approve.textContent = "Approve";
        approve.addEventListener("click", async () => {
          await post("/api/feedback/review", { feedback_id: item.feedback_id, approve: true });
          refreshFeedback();
        });
        const reject = document.createElement("button");
        reject.className = "button secondary small";
        reject.textContent = "Reject";
        reject.addEventListener("click", async () => {
          await post("/api/feedback/review", { feedback_id: item.feedback_id, approve: false });
          refreshFeedback();
        });
        div.append(span, approve, reject);
        box.append(div);
      });
    } catch { /* ignore */ }
  }

  // One studio at a time. The backend already refuses a second server on 8765,
  // but a second browser tab would still open a second studio against the same
  // backend -- two pollers, two sets of controls over one training run. Tabs
  // elect a single owner over BroadcastChannel; the others sit idle behind an
  // overlay until the operator explicitly takes over.
  const TAB_CHANNEL = "adamlm.studio.v1";
  const TAB_ID = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const lockEl = $("tabLock");
  const takeoverEl = $("tabLockTakeover");
  let channel = null;
  let owner = false;

  const startStudio = () => {
    if (owner) return;
    owner = true;
    if (lockEl) lockEl.hidden = true;
    updatePromptCount();
    autogrowPrompt();
    updateChatNote();
    renderChat();
    updateChatContextNote();
    refresh(true);
    state.timer = setInterval(() => refresh(false), 2200);
  };

  const standDown = () => {
    owner = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    if (lockEl) lockEl.hidden = false;
  };

  if (takeoverEl) {
    takeoverEl.addEventListener("click", () => {
      if (channel) channel.postMessage({ type: "takeover", from: TAB_ID });
      startStudio();
    });
  }

  try {
    channel = new BroadcastChannel(TAB_CHANNEL);
  } catch { channel = null; }

  if (!channel) {
    // No BroadcastChannel (or it is blocked): degrade to a working studio
    // rather than a blank page.
    startStudio();
  } else {
    channel.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.from === TAB_ID) return;
      if (message.type === "who" && owner) {
        channel.postMessage({ type: "here", from: TAB_ID });
      } else if (message.type === "here" && !owner) {
        clearTimeout(claimTimer);
        standDown();
      } else if (message.type === "takeover" && owner) {
        standDown();
      }
    });
    channel.postMessage({ type: "who", from: TAB_ID });
    // No answer in this window means no live studio tab, so claim ownership.
    var claimTimer = setTimeout(startStudio, 300);
  }
})();
