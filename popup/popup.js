const conversation = document.querySelector("#conversation");
const messagesElement = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const composer = document.querySelector("#composer");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearButton");

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

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content) return;

  sendButton.disabled = true;
  try {
    const result = await chrome.runtime.sendMessage({
      type: "SAVE_USER_MESSAGE",
      content
    });
    if (!result?.ok) throw new Error(result?.error || "Could not save message.");
    input.value = "";
    render(result.conversation);
  } catch (error) {
    console.error("Message save failed:", error);
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
});

clearButton.addEventListener("click", async () => {
  const result = await chrome.runtime.sendMessage({ type: "CLEAR_CONVERSATION" });
  if (result?.ok) render([]);
});

loadMessages().catch((error) => console.error("Conversation load failed:", error));
