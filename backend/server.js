import http from "node:http";

const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.PORT || "8787", 10);
const AI_PROVIDER = (process.env.AI_PROVIDER || "ollama").toLowerCase();
const AI_MODEL = process.env.AI_MODEL || "qwen2.5:7b";
const OLLAMA_BASE_URL = (process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434").replace(/\/+$/, "");
const AI_BASE_URL = (process.env.AI_BASE_URL || "").replace(/\/+$/, "");
const AI_API_KEY = process.env.AI_API_KEY || "";
const allowedOrigins = (process.env.ALLOWED_EXTENSION_ORIGINS || "")
  .split(",").map((origin) => origin.trim()).filter(Boolean);

const MAX_BODY_BYTES = 32 * 1024;
const MAX_MESSAGES = 30;
const MAX_MESSAGE_CHARS = 8000;

function sendJson(response, status, payload, extraHeaders = {}) {
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    ...extraHeaders
  });
  response.end(JSON.stringify(payload));
}

function originAllowed(request) {
  const origin = request.headers.origin;
  return !origin || allowedOrigins.includes(origin);
}

function corsHeaders(request) {
  const origin = request.headers.origin;
  return origin && allowedOrigins.includes(origin)
    ? { "Access-Control-Allow-Origin": origin, "Vary": "Origin" }
    : {};
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    let size = 0;
    let raw = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      size += Buffer.byteLength(chunk);
      if (size > MAX_BODY_BYTES) {
        reject(new Error("Request body too large."));
        request.destroy();
        return;
      }
      raw += chunk;
    });
    request.on("end", () => {
      try { resolve(JSON.parse(raw || "{}")); }
      catch { reject(new Error("Request body must be valid JSON.")); }
    });
    request.on("error", reject);
  });
}

function validateMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0 || messages.length > MAX_MESSAGES) {
    return "messages must be an array containing 1–30 items.";
  }
  for (const message of messages) {
    if (!message || !["user", "assistant"].includes(message.role) ||
        typeof message.content !== "string" || !message.content.trim() ||
        message.content.length > MAX_MESSAGE_CHARS) {
      return "Each message must have a valid user/assistant role and 1–8000 characters of text.";
    }
  }
  return null;
}

function providerConfiguration() {
  if (!AI_MODEL) return false;
  if (AI_PROVIDER === "ollama") return Boolean(OLLAMA_BASE_URL);
  if (AI_PROVIDER === "openai-compatible") return Boolean(AI_BASE_URL && AI_API_KEY);
  return false;
}

async function callOllama(messages) {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: AI_MODEL, messages, stream: false }),
    signal: AbortSignal.timeout(120000)
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error("Ollama request failed.");
    error.status = response.status;
    throw error;
  }
  const content = result?.message?.content;
  if (typeof content !== "string" || !content.trim()) {
    throw new Error("Ollama returned an unexpected response.");
  }
  return content;
}

async function callOpenAICompatible(messages) {
  const response = await fetch(`${AI_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${AI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ model: AI_MODEL, messages, temperature: 0.7 }),
    signal: AbortSignal.timeout(120000)
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error("AI provider request failed.");
  const content = result?.choices?.[0]?.message?.content;
  if (typeof content !== "string" || !content.trim()) {
    throw new Error("AI provider returned an unexpected response.");
  }
  return content;
}

async function handleChat(request, response, headers) {
  if (!providerConfiguration()) {
    sendJson(response, 503, {
      error: { code: "AI_NOT_CONFIGURED", message: "Check AI_PROVIDER, AI_MODEL and provider connection settings in backend/.env." }
    }, headers);
    return;
  }

  let body;
  try { body = await readJson(request); }
  catch (error) {
    sendJson(response, 400, { error: { code: "INVALID_JSON", message: error.message } }, headers);
    return;
  }

  const validationError = validateMessages(body.messages);
  if (validationError) {
    sendJson(response, 400, { error: { code: "INVALID_MESSAGES", message: validationError } }, headers);
    return;
  }

  try {
    const content = AI_PROVIDER === "ollama"
      ? await callOllama(body.messages)
      : await callOpenAICompatible(body.messages);
    sendJson(response, 200, { message: { role: "assistant", content } }, headers);
  } catch (error) {
    console.error("AI provider request failed:", error.name || "Error");
    const ollamaUnavailable = AI_PROVIDER === "ollama" && error.cause?.code === "ECONNREFUSED";
    sendJson(response, 502, {
      error: {
        code: ollamaUnavailable ? "OLLAMA_OFFLINE" : "AI_PROVIDER_ERROR",
        message: ollamaUnavailable
          ? "Ollama is not running. Start Ollama and make sure the selected model is installed."
          : "The local AI provider could not complete the request. Check the model name and provider logs."
      }
    }, headers);
  }
}

async function getProviderStatus() {
  const configured = providerConfiguration();
  if (!configured) {
    return { configured: false, provider: AI_PROVIDER, model: AI_MODEL, available: false, modelInstalled: false };
  }
  if (AI_PROVIDER !== "ollama") {
    return { configured: true, provider: AI_PROVIDER, model: AI_MODEL, available: null, modelInstalled: null };
  }
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) throw new Error("Ollama unavailable");
    const data = await response.json();
    const models = Array.isArray(data.models) ? data.models.map((item) => item.name) : [];
    return {
      configured: true,
      provider: AI_PROVIDER,
      model: AI_MODEL,
      available: true,
      modelInstalled: models.some((name) => name === AI_MODEL || name.split(":")[0] === AI_MODEL.split(":")[0]),
      installedModels: models
    };
  } catch {
    return { configured: true, provider: AI_PROVIDER, model: AI_MODEL, available: false, modelInstalled: false };
  }
}

const server = http.createServer(async (request, response) => {
  const headers = corsHeaders(request);
  if (!originAllowed(request)) {
    sendJson(response, 403, {
      error: { code: "ORIGIN_NOT_ALLOWED", message: "This request origin is not allowed. Add the exact extension origin to ALLOWED_EXTENSION_ORIGINS." }
    });
    return;
  }

  if (request.method === "OPTIONS") {
    response.writeHead(204, {
      ...headers,
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Max-Age": "600"
    });
    response.end();
    return;
  }

  const url = new URL(request.url || "/", `http://${HOST}:${PORT}`);
  if (request.method === "GET" && url.pathname === "/health") {
    sendJson(response, 200, {
      status: "ok",
      service: "development-ai-backend",
      provider: AI_PROVIDER,
      model: AI_MODEL,
      aiConfigured: providerConfiguration()
    }, headers);
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/status") {
    sendJson(response, 200, await getProviderStatus(), headers);
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/chat") {
    await handleChat(request, response, headers);
    return;
  }

  sendJson(response, 404, { error: { code: "NOT_FOUND", message: "Route not found." } }, headers);
});

server.listen(PORT, HOST, () => {
  console.info(`Development AI backend listening on http://${HOST}:${PORT}`);
});
