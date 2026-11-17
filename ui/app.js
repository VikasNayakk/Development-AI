(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const terminalStates = new Set(["succeeded", "failed", "cancelled"]);
  const state = { status: null, currentJobId: null, pollHandle: null };

  function toast(message, isError = false) {
    const region = $("#toast-region");
    const item = document.createElement("div");
    item.className = "toast" + (isError ? " error" : "");
    item.textContent = message;
    region.append(item);
    window.setTimeout(() => item.remove(), 4200);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) },
      cache: "no-store"
    });
    let data;
    try { data = await response.json(); }
    catch { throw new Error("The local server returned an unreadable response."); }
    if (!response.ok) throw new Error(data.error || "Local request failed.");
    return data;
  }

  function showView(name) {
    const title = ({ overview: "Overview", playground: "Playground", training: "Training", evaluation: "Evaluation" })[name] || "Overview";
    $$(".view").forEach((view) => view.classList.toggle("active", view.id === "view-" + name));
    $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
    $("#current-section").textContent = title;
    if (name === "overview") loadStatus();
    if (name === "training" || name === "evaluation") loadJobs();
  }

  $$("[data-view]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
  $$("[data-go]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.go)));

  function setConnection(connected) {
    const element = $("#connection-indicator");
    element.classList.toggle("connected", connected);
    element.classList.toggle("disconnected", !connected);
    element.querySelector("span").textContent = connected ? "Local server connected" : "Server disconnected";
  }

  function renderStatus(data) {
    state.status = data;
    const model = data.model || {};
    const config = model.config || {};
    const params = model.parameters || {};
    const checkpoint = data.checkpoint || {};
    $("#metric-model-status").textContent = checkpoint.trained
      ? "Trained"
      : checkpoint.status === "invalid" ? "Invalid checkpoint"
      : checkpoint.exists ? "Untrained" : "Not trained";
    $("#metric-model-description").textContent = checkpoint.trained
      ? "Best-validation checkpoint available"
      : checkpoint.status === "invalid" ? "Checkpoint metadata or weights are invalid"
      : checkpoint.exists ? "Checkpoint file exists but training is not verified" : "Train a model to enable inference";
    $("#metric-parameters").textContent = Number(params.trainable || 0).toLocaleString();
    $("#metric-vocab").textContent = Number((model.tokenizer || {}).vocab_size || 0).toLocaleString();
    $("#metric-tokenizer").textContent = (model.tokenizer || {}).name || "—";
    $("#metric-steps").textContent = checkpoint.steps_completed == null ? "—" : Number(checkpoint.steps_completed).toLocaleString();
    $("#arch-context").textContent = config.block_size ?? "—";
    $("#arch-width").textContent = config.d_model ?? "—";
    $("#arch-heads").textContent = config.n_heads ?? "—";
    $("#arch-layers").textContent = config.n_layers ?? "—";

    const notice = $("#model-notice");
    notice.hidden = checkpoint.trained;
    if (!checkpoint.trained) {
      notice.querySelector("strong").textContent = checkpoint.status === "invalid"
        ? "Checkpoint file is invalid"
        : checkpoint.exists ? "Checkpoint has not been trained" : "Training checkpoint not found";
      notice.querySelector("p").textContent = checkpoint.status === "invalid"
        ? "Move the invalid artifact files out of artifacts/ before starting a fresh run."
        : checkpoint.exists
          ? "Resume only from the matching training and resume checkpoints, or move the old artifacts before a fresh run."
          : "Prepare a dataset and complete a training run before using the Playground.";
    }
    $("#playground-warning").hidden = checkpoint.trained;
    $("#playground-checkpoint").textContent = checkpoint.trained
      ? "Checkpoint · step " + Number(checkpoint.steps_completed).toLocaleString()
      : checkpoint.exists ? "Checkpoint not verified as trained" : "No trained checkpoint";

    const cudaOptions = Array.from(document.querySelectorAll('select option[value="cuda"]'));
    cudaOptions.forEach((option) => { option.disabled = !data.runtime?.cuda_available; });
    if (!data.runtime?.cuda_available) {
      ["#generation-device", "#train-device", "#evaluation-device"].forEach((selector) => {
        if ($(selector).value === "cuda") $(selector).value = "cpu";
      });
    }
    renderRecentJobs(data.jobs || []);
    updateTrainingResumeHint(checkpoint);
  }

  function updateTrainingResumeHint(checkpoint) {
    const note = $(".resume-row small");
    if (!note) return;
    const resumeBox = $("#resume-training");
    resumeBox.disabled = !checkpoint.trained || !checkpoint.resume_exists;
    if (checkpoint.status === "invalid") {
      note.textContent = "The existing checkpoint is invalid. Move the old files out of artifacts/ before starting fresh.";
    } else if (checkpoint.exists && checkpoint.resume_exists) {
      note.textContent = "Both checkpoints exist. Enable this only to continue the same run with the same corpus and settings.";
    } else if (checkpoint.exists || checkpoint.resume_exists) {
      note.textContent = "A previous artifact exists without its matching pair. Move the old files out of artifacts/ before a fresh run.";
    } else {
      note.textContent = "Continue from artifacts/latest.resume.pt after an interrupted run using the same corpus and settings.";
    }
  }

  function renderRecentJobs(jobs) {
    $("#jobs-count").textContent = jobs.length + " job" + (jobs.length === 1 ? "" : "s");
    const host = $("#recent-jobs");
    if (!jobs.length) {
      host.className = "empty-state compact";
      host.textContent = "No training or evaluation jobs yet.";
      return;
    }
    host.className = "job-list";
    host.replaceChildren();
    jobs.slice(0, 5).forEach((job) => {
      const row = document.createElement("div");
      row.className = "job-row";
      const label = document.createElement("div");
      const title = document.createElement("strong");
      title.className = "job-type";
      title.textContent = job.kind || "job";
      const detail = document.createElement("small");
      detail.textContent = job.status + " · " + new Date(job.created_at).toLocaleString();
      label.append(title, detail);
      const badge = document.createElement("span");
      badge.className = "status-pill " + statusClass(job.status);
      badge.textContent = job.status;
      row.append(label, badge);
      host.append(row);
    });
  }

  function statusClass(status) {
    if (status === "running" || status === "queued" || status === "stopping") return "running";
    if (status === "succeeded") return "succeeded";
    if (status === "failed" || status === "cancelled") return status;
    return "neutral";
  }

  async function loadStatus() {
    try {
      const data = await api("/api/status");
      setConnection(true);
      renderStatus(data);
      const active = (data.jobs || []).find((job) => ["queued", "running", "stopping"].includes(job.status));
      if (active && !state.currentJobId) attachToJob(active.id);
    } catch (error) {
      setConnection(false);
      toast(error.message, true);
    }
  }

  async function loadJobs() {
    try {
      const data = await api("/api/jobs");
      renderRecentJobs(data.jobs || []);
      const active = (data.jobs || []).find((job) => ["queued", "running", "stopping"].includes(job.status));
      if (active && !state.currentJobId) attachToJob(active.id);
    } catch (error) { toast(error.message, true); }
  }

  function attachToJob(jobId) {
    state.currentJobId = jobId;
    if (state.pollHandle) window.clearInterval(state.pollHandle);
    pollJob(jobId);
    state.pollHandle = window.setInterval(() => pollJob(jobId), 1400);
  }

  async function pollJob(jobId) {
    try {
      const job = await api("/api/jobs/" + encodeURIComponent(jobId));
      renderJob(job);
      if (job.kind === "training") updateTrainingMonitor(job);
      if (job.kind === "evaluation") updateEvaluationMonitor(job);
      if (terminalStates.has(job.status)) {
        if (state.pollHandle) window.clearInterval(state.pollHandle);
        state.pollHandle = null;
        state.currentJobId = null;
        if (job.status === "succeeded") toast(job.kind + " completed successfully.");
        else if (job.status === "failed") toast(job.error || (job.kind + " failed."), true);
        else toast(job.kind + " was stopped.");
        loadStatus();
      }
    } catch (error) {
      toast(error.message, true);
      if (state.pollHandle) window.clearInterval(state.pollHandle);
      state.pollHandle = null;
      state.currentJobId = null;
    }
  }

  function renderJob(job) {
    if (job.kind === "training") updateTrainingMonitor(job);
    if (job.kind === "evaluation") updateEvaluationMonitor(job);
  }

  function updateTrainingMonitor(job) {
    const status = $("#training-state");
    status.textContent = job.status;
    status.className = "status-pill " + statusClass(job.status);
    const step = Number(job.step || 0);
    const target = Number(job.target_steps || 0);
    const percent = target > 0 ? Math.min(100, step / target * 100) : 0;
    $("#training-progress-value").textContent = Math.floor(percent) + "%";
    $("#training-step-value").textContent = target ? "Step " + step.toLocaleString() + " / " + target.toLocaleString() : "Waiting for first update";
    $("#training-progress-bar").style.width = percent + "%";
    const logs = job.logs || [];
    $("#training-log").textContent = logs.length ? logs.join("\n") : "Waiting for training output…";
    $("#training-log").scrollTop = $("#training-log").scrollHeight;
    const latest = [...logs].reverse().find((line) => line.includes("validation_loss"));
    const val = latest?.match(/validation_loss\s+([0-9.]+)/)?.[1];
    const speed = latest?.match(/measured_tokens_per_second\s+([0-9.]+)/)?.[1];
    $("#training-live-metrics").replaceChildren();
    [["Current validation loss", val || "—"], ["Measured throughput", speed ? Number(speed).toFixed(1) + " byte/s" : "—"]].forEach(([label, value]) => {
      const box = document.createElement("div");
      const caption = document.createElement("span");
      caption.textContent = label;
      const metric = document.createElement("strong");
      metric.textContent = value;
      box.append(caption, metric);
      $("#training-live-metrics").append(box);
    });
    $("#stop-training-button").disabled = !["queued", "running", "stopping"].includes(job.status);
    $("#start-training-button").disabled = ["queued", "running", "stopping"].includes(job.status);
  }

  function updateEvaluationMonitor(job) {
    const status = $("#evaluation-state");
    status.textContent = job.status;
    status.className = "status-pill " + statusClass(job.status);
    const active = ["queued", "running", "stopping"].includes(job.status);
    $("#start-evaluation-button").disabled = active;
    $("#stop-evaluation-button").disabled = !active;
    if (job.status === "succeeded" && job.result) renderEvaluationResult(job.result);
    else if (job.status === "failed") {
      $("#evaluation-results").textContent = job.error || "Evaluation failed.";
      $("#evaluation-json").hidden = true;
    }
  }

  function renderEvaluationResult(payload) {
    const evaluation = payload.evaluation || {};
    const trained = evaluation.language_split_metrics || [];
    const baseline = evaluation.baseline_comparison?.language_split_metrics || [];
    const host = $("#evaluation-results");
    host.replaceChildren();
    host.className = "";
    if (!trained.length) {
      host.className = "empty-state";
      host.textContent = "No language/split metrics were returned. Check the evaluation manifest.";
    } else {
      const title = document.createElement("p");
      title.className = "result-disclaimer";
      title.textContent = trained.length + " language/split group(s) measured. Loss and perplexity indicate next-byte prediction only.";
      host.append(title);
      const grid = document.createElement("div");
      grid.className = "result-grid";
      trained.forEach((metric) => {
        const compare = baseline.find((item) => item.language === metric.language && item.split === metric.split);
        const card = document.createElement("div");
        card.className = "result-card";
        const name = document.createElement("strong");
        name.textContent = metric.language + " · " + metric.split;
        const loss = document.createElement("span");
        loss.textContent = "Loss " + Number(metric.loss).toFixed(4);
        const detail = document.createElement("small");
        detail.textContent = "Perplexity " + (metric.perplexity == null ? "overflow" : Number(metric.perplexity).toFixed(2)) + " · " + Number(metric.tokens_scored).toLocaleString() + " bytes scored";
        card.append(name, loss, detail);
        if (compare) {
          const delta = document.createElement("small");
          const prefix = compare.loss_improvement >= 0 ? "+" : "";
          delta.textContent = "vs untrained · " + prefix + Number(compare.loss_improvement).toFixed(4) + " loss";
          delta.style.color = compare.loss_improvement >= 0 ? "#158568" : "#b56b25";
          card.append(delta);
        }
        grid.append(card);
      });
      host.append(grid);
      const missing = evaluation.missing_language_split_combinations || [];
      if (missing.length) {
        const note = document.createElement("p");
        note.className = "result-disclaimer";
        note.textContent = "Missing groups: " + missing.map((item) => item.language + "/" + item.split).join(", ");
        host.append(note);
      }
    }
    $("#evaluation-json").hidden = false;
    $("#evaluation-json").textContent = JSON.stringify(payload, null, 2);
  }

  function showLoading(button, loading, text) {
    if (!button.dataset.originalText) button.dataset.originalText = button.textContent.trim();
    button.disabled = loading;
    button.textContent = loading ? text : button.dataset.originalText;
  }

  $("#refresh-button").addEventListener("click", loadStatus);

  $("#max-tokens").addEventListener("input", (event) => {
    $("#max-tokens-value").textContent = event.target.value;
  });

  $("#generation-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#generate-button");
    const prompt = $("#prompt-input").value;
    const seedText = $("#generation-seed").value.trim();
    const payload = {
      prompt,
      max_new_tokens: Number($("#max-tokens").value),
      temperature: Number($("#temperature").value),
      device: $("#generation-device").value,
      greedy: $("#greedy-mode").checked,
      seed: seedText === "" ? null : Number(seedText)
    };
    if (!state.status?.checkpoint?.trained) {
      toast("This checkpoint is not verified as trained. Train a model first.", true);
      showView("training");
      return;
    }
    showLoading(button, true, "Generating…");
    addChatMessage("user", prompt);
    $("#prompt-input").value = "";
    try {
      const result = await api("/api/chat/generate", { method: "POST", body: JSON.stringify(payload) });
      const answer = result.text.startsWith(prompt) ? result.text.slice(prompt.length) : result.text;
      addChatMessage("assistant", answer || "(No new tokens returned.)");
      if (!result.checkpoint_trained) toast("The checkpoint has no verified training metadata.", true);
    } catch (error) {
      addChatMessage("assistant", "Generation error: " + error.message);
      toast(error.message, true);
    } finally { showLoading(button, false); }
  });

  function addChatMessage(role, text) {
    const host = $("#chat-output");
    const welcome = host.querySelector(".welcome-message");
    if (welcome) welcome.remove();
    const message = document.createElement("div");
    message.className = "message " + role;
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = role === "user" ? "You" : "Development-AI";
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text;
    message.append(label, body);
    host.append(message);
    host.scrollTop = host.scrollHeight;
  }

  $("#training-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#start-training-button");
    const payload = {
      data_path: $("#training-data-path").value.trim(),
      steps: Number($("#train-steps").value),
      batch_size: Number($("#train-batch").value),
      gradient_accumulation_steps: Number($("#train-accumulation").value),
      block_size: Number($("#train-block").value),
      learning_rate: Number($("#train-lr").value),
      seed: Number($("#train-seed").value),
      eval_interval: Number($("#train-eval-interval").value),
      eval_iters: Number($("#train-eval-iters").value),
      validation_fraction: Number($("#train-validation").value),
      early_stopping_patience: Number($("#train-early-stopping-patience").value),
      early_stopping_min_delta: Number($("#train-early-stopping-min-delta").value),
      device: $("#train-device").value,
      resume: $("#resume-training").checked
    };
    showLoading(button, true, "Starting…");
    try {
      const response = await api("/api/training", { method: "POST", body: JSON.stringify(payload) });
      $("#training-log").textContent = "Training job queued…";
      attachToJob(response.job.id);
      toast("Training job started.");
    } catch (error) { toast(error.message, true); }
    finally {
      showLoading(button, false);
      if (state.currentJobId) button.disabled = true;
    }
  });

  $("#stop-training-button").addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      await api("/api/jobs/" + encodeURIComponent(state.currentJobId) + "/stop", {
        method: "POST", body: "{}"
      });
      toast("Stop requested.");
    } catch (error) { toast(error.message, true); }
  });

  $("#clear-log-button").addEventListener("click", () => {
    $("#training-log").textContent = "Log view cleared. The job continues running.";
  });

  $("#stop-evaluation-button").addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      await api("/api/jobs/" + encodeURIComponent(state.currentJobId) + "/stop", {
        method: "POST", body: "{}"
      });
      toast("Evaluation stop requested.");
    } catch (error) { toast(error.message, true); }
  });

  $("#evaluation-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#start-evaluation-button");
    const payload = {
      manifest_path: $("#evaluation-manifest").value.trim(),
      device: $("#evaluation-device").value,
      compare_untrained: $("#compare-untrained").checked,
      baseline_seed: Number($("#baseline-seed").value)
    };
    showLoading(button, true, "Queued…");
    try {
      const response = await api("/api/evaluation", { method: "POST", body: JSON.stringify(payload) });
      $("#evaluation-results").className = "empty-state";
      $("#evaluation-results").textContent = "Evaluation running. Results will appear here when complete.";
      $("#evaluation-json").hidden = true;
      attachToJob(response.job.id);
      toast("Evaluation job started.");
    } catch (error) { toast(error.message, true); }
    finally {
      showLoading(button, false);
      if (state.currentJobId) button.disabled = true;
    }
  });

  loadStatus();
  window.setInterval(() => {
    if (!state.currentJobId) loadStatus();
  }, 7000);
})();
