const MAX_MESSAGES = 100;

chrome.runtime.onInstalled.addListener(() => {
  console.info("Development AI extension installed.");
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "SAVE_USER_MESSAGE") {
    const content = typeof message.content === "string" ? message.content.trim() : "";
    if (!content || content.length > 8000) {
      sendResponse({ ok: false, error: "Message must contain 1–8000 characters." });
      return false;
    }

    chrome.storage.local.get({ conversation: [] }).then(({ conversation }) => {
      const next = [
        ...conversation,
        { role: "user", content, createdAt: new Date().toISOString() },
        {
          role: "notice",
          content: "Message saved locally. AI is not connected yet, so this message has not been processed.",
          createdAt: new Date().toISOString()
        }
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
