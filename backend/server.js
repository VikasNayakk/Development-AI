import http from "node:http";

const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.PORT || "8787", 10);
const AI_BASE_URL = (process.env.AI_BASE_URL || "").replace(/\/+$/, "");
const AI_API_KEY = process.env.AI_API_KEY || "";
const AI_MODEL = process.env.AI_MODEL || "";
const allowedOrigins = (process.env.ALLOWED_EXTENSION_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

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
  if (origin && allowedOrigins.includes(origin)) {
    return { "Access-Control-Allow-Origin": origin, "Vary": "Origin" };
  }
  return {};
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
      try {
        resolve(JSON.parse(raw || "{}"));
      } catch {
        reject(new Error("Request body must be valid JSON."));
      }
    });
    request.on("error", reject);
  });
}

function validateMessages(value) {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_MESSAGES) {
    return "messages must be an array containing 1–30 items.";
  }

  for (const message of value) {
    if (
      !message ||
      !["user", "assistant"].includes(message.role) ||
      typeof message.content !== "string" ||
      message.content.trim().length === 0 ||
      message.content.length > MAX_MESSAGE_CHARS
    ) {
      return "Each message must have a valid user/assistant role and 1–8000 characters of text.";
    }
  }
  return null;
}

async function handleChat(request, response, headers) {
  if (!AI_BASE_URL || !AI_API_KEY || !AI_MODEL) {
    sendJson(response, 503, {
      error: {
        code: "AI_NOT_CONFIGURED",
        message: "Configure AI_BASE_URL, AI_API_KEY and AI_MODEL in backend/.env."
      }
    }, headers);
    return;
  }

  let body;
  try {
    body = await readJson(request);
  } catch (error) {
    sendJson(response, 400, {
      error: { code: "INVALID_JSON", message: error.message }
    }, headers);
    return;
  }

  const validationError = validateMessages(body.messages);
  if (validationError) {
    sendJson(response, 400, {
      error: { code: "INVALID_MESSAGES", message: validationError }
    }, headers);
    return;
  }

  try {
    const upstream = await fetch(`${AI_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${AI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: AI_MODEL,
        messages: body.messages,
        temperature: 0.7
      }),
      signal: AbortSignal.timeout(60000)
    });

    const result = await upstream.json().catch(() => ({}));
    if (!upstream.ok) {
      console.error("AI provider returned HTTP", upstream.status);
      sendJson(response, 502, {
        error: {
          code: "AI_PROVIDER_ERROR",
          message: "The configured AI provider could not complete the request."
        }
      }, headers);
      return;
    }

    const content = result?.choices?.[0]?.message?.content;
    if (typeof content !== "string" || !content.trim()) {
      sendJson(response, 502, {
        error: {
          code: "INVALID_PROVIDER_RESPONSE",
          message: "The AI provider returned an unexpected response."
        }
      }, headers);
      return;
    }

    sendJson(response, 200, {
      message: { role: "assistant", content }
    }, headers);
  } catch (error) {
    console.error("AI gateway request failed:", error.name);
    sendJson(response, 502, {
      error: {
        code: "AI_GATEWAY_FAILURE",
        message: "Could not reach the configured AI provider."
      }
    }, headers);
  }
}

const server = http.createServer(async (request, response) => {
  const headers = corsHeaders(request);

  if (!originAllowed(request)) {
    sendJson(response, 403, {
      error: {
        code: "ORIGIN_NOT_ALLOWED",
        message: "This request origin is not allowed. Add the exact extension origin to ALLOWED_EXTENSION_ORIGINS."
      }
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
      aiConfigured: Boolean(AI_BASE_URL && AI_API_KEY && AI_MODEL)
    }, headers);
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/chat") {
    await handleChat(request, response, headers);
    return;
  }

  sendJson(response, 404, {
    error: { code: "NOT_FOUND", message: "Route not found." }
  }, headers);
});

server.listen(PORT, HOST, () => {
  console.info(`Development AI backend listening on http://${HOST}:${PORT}`);
});
