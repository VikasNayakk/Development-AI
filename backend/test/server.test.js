import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { spawn } from "node:child_process";

const ORIGIN = "chrome-extension://test-extension-id";
let backend;
let provider;
let backendPort;
let providerPort;

async function unusedPort() {
  const server = http.createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function waitForHealth(url, child) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    if (child.exitCode !== null) throw new Error("Backend process exited during startup.");
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("Backend did not start in time.");
}

before(async () => {
  providerPort = await unusedPort();
  backendPort = await unusedPort();

  provider = http.createServer(async (request, response) => {
    if (request.url === "/api/tags") {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ models: [{ name: "qwen2.5:7b" }] }));
      return;
    }
    let raw = "";
    for await (const chunk of request) raw += chunk;
    const payload = JSON.parse(raw || "{}");
    assert.equal(request.url, "/api/chat");
    assert.equal(payload.stream, false);
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({
      message: { role: "assistant", content: `Local mock reply: ${payload.messages.at(-1).content}` }
    }));
  });
  await new Promise((resolve) => provider.listen(providerPort, "127.0.0.1", resolve));

  backend = spawn(process.execPath, ["server.js"], {
    cwd: new URL("..", import.meta.url),
    env: {
      ...process.env,
      HOST: "127.0.0.1",
      PORT: String(backendPort),
      AI_PROVIDER: "ollama",
      AI_MODEL: "qwen2.5:7b",
      OLLAMA_BASE_URL: `http://127.0.0.1:${providerPort}`,
      ALLOWED_EXTENSION_ORIGINS: ORIGIN
    },
    stdio: "ignore"
  });
  await waitForHealth(`http://127.0.0.1:${backendPort}/health`, backend);
});

after(async () => {
  if (backend && backend.exitCode === null) backend.kill();
  if (provider) await new Promise((resolve) => provider.close(resolve));
});

test("health endpoint reports local Ollama provider", async () => {
  const response = await fetch(`http://127.0.0.1:${backendPort}/health`);
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.equal(body.provider, "ollama");
  assert.equal(body.aiConfigured, true);
});

test("status endpoint detects installed local model", async () => {
  const response = await fetch(`http://127.0.0.1:${backendPort}/v1/status`);
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.equal(body.available, true);
  assert.equal(body.modelInstalled, true);
  assert.deepEqual(body.installedModels, ["qwen2.5:7b"]);
});

test("chat endpoint calls Ollama API and returns assistant message", async () => {
  const response = await fetch(`http://127.0.0.1:${backendPort}/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify({ messages: [{ role: "user", content: "Hello" }] })
  });
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.deepEqual(body.message, { role: "assistant", content: "Local mock reply: Hello" });
  assert.equal(response.headers.get("access-control-allow-origin"), ORIGIN);
});

test("rejects invalid message payload", async () => {
  const response = await fetch(`http://127.0.0.1:${backendPort}/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify({ messages: [{ role: "system", content: "bad role" }] })
  });
  const body = await response.json();
  assert.equal(response.status, 400);
  assert.equal(body.error.code, "INVALID_MESSAGES");
});

test("rejects an unapproved browser origin", async () => {
  const response = await fetch(`http://127.0.0.1:${backendPort}/health`, {
    headers: { Origin: "https://untrusted.example" }
  });
  const body = await response.json();
  assert.equal(response.status, 403);
  assert.equal(body.error.code, "ORIGIN_NOT_ALLOWED");
});
