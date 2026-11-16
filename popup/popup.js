const status = document.querySelector("#status");
const checkButton = document.querySelector("#checkButton");

checkButton.addEventListener("click", async () => {
  const manifest = chrome.runtime.getManifest();
  status.textContent = `Ready · version ${manifest.version}`;
});
