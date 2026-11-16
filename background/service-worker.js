const MAX_MESSAGES = 100;
const MAX_MESSAGE_CHARS = 8000;

chrome.runtime.onInstalled.addListener(() => {
  console.info("Development AI extension installed.");
});

function isValidContent(content) {
  return typeof content === "string" &&
    content.trim().length > 0 &&
    content.length <= MAX_MESSAGE_CHARS;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "SAVE_USER_MESSAGE" || message?.type === "SAVE_ASSISTANT_MESSAGE" || message?.type === "SAVE_NOTICE") {
    const content = message.content;
    if (!isValidContent(content)) {
      sendResponse({ ok: false, error: "Message must contain 1–8000 characters." });
      return false;
    }

    const role = {
      SAVE_USER_MESSAGE: "user",
      SAVE_ASSISTANT_MESSAGE: "assistant",
      SAVE_NOTICE: "notice"
    }[message.type];

    chrome.storage.local.get({ conversation: [] }).then(({ conversation }) => {
      const next = [
        ...conversation,
        { role, content: content.trim(), createdAt: new Date().toISOString() }
      ].slice(-MAX_MESSAGES);

      return chrome.storage.local.set({ conversation: next }).then(() => {
        sendResponse({ ok: true, conversation: next });
      });
    }).catch((error) => {
      console.error("Conversation storage error:", error);
      sendResponse({ ok: false, error: "Local storage is unavailable." });
    });
    return true;
  }

  if (message?.type === "CLEAR_CONVERSATION") {
    chrome.storage.local.set({ conversation: [] })
      .then(() => sendResponse({ ok: true }))
      .catch(() => sendResponse({ ok: false, error: "Could not clear conversation." }));
    return true;
  }

  return false;
});
