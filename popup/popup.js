const API_BASE = "http://127.0.0.1:8787";
const conversation = document.querySelector("#conversation");
const messagesElement = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const composer = document.querySelector("#composer");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearButton");
const connectionText = document.querySelector("#connectionText");
const connectionDot = document.querySelector("#connectionDot");

function setConnection(online, text) {
  connectionText.textContent = text;
  connectionDot.style.background = online ? "#12b76a" : "#e3a008";
}

function render(messages) {
  messagesElement.replaceChildren();
  welcome.hidden = messages.length > 0;

  for (const message of messages) {
    const item = document.createElement("div");
    item.className = `message ${message.role === "user" ? "user" : "notice"}`;
    item.textContent = message.content;
    messagesElement.append(item);
  }
  conversation.scrollTop = conversation.scrollHeight;
}

async function loadMessages() {
  const result = await chrome.storage.local.get({ conversation: [] });
  render(result.conversation);
}

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!response.ok) throw new Error("Backend unavailable");
    const health = await response.json();
    setConnection(
      health.aiConfigured,
      health.aiConfigured ? "AI backend connected" : "Backend online · AI not configured"
    );
  } catch {
    setConnection(false, "AI backend offline · start backend locally");
  }
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || sendButton.disabled) return;

  sendButton.disabled = true;
  input.disabled = true;
  try {
    const stored = await chrome.runtime.sendMessage({
      type: "SAVE_USER_MESSAGE",
      content
    });
    if (!stored?.ok) throw new Error(stored?.error || "Could not save message.");
    render(stored.conversation);
    input.value = "";

    const history = stored.conversation
      .filter((item) => item.role === "user" || item.role === "assistant")
      .slice(-20)
      .map(({ role, content: text }) => ({ role, content: text }));

    const response = await fetch(`${API_BASE}/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history })
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = result?.error?.message || "AI request failed.";
      throw new Error(error);
    }

    const savedReply = await chrome.runtime.sendMessage({
      type: "SAVE_ASSISTANT_MESSAGE",
      content: result.message.content
    });
    if (!savedReply?.ok) throw new Error("AI replied, but the reply could not be saved.");
    render(savedReply.conversation);
    setConnection(true, "AI backend connected");
  } catch (error) {
    const notice = await chrome.runtime.sendMessage({
      type: "SAVE_NOTICE",
      content: error.message || "Could not reach the AI backend."
    });
    if (notice?.ok) render(notice.conversation);
    setConnection(false, "AI request failed");
  } finally {
    sendButton.disabled = false;
    input.disabled = false;
    input.focus();
  }
});

clearButton.addEventListener("click", async () => {
  const result = await chrome.runtime.sendMessage({ type: "CLEAR_CONVERSATION" });
  if (result?.ok) render([]);
});

loadMessages().catch((error) => console.error("Conversation load failed:", error));
checkBackend();
