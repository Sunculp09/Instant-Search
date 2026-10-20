import "./style.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const form = document.querySelector("#ask-form");
const statementInput = document.querySelector("#statement");
const submitButton = document.querySelector("#submit-button");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const answerText = document.querySelector("#answer-text");
const sourceList = document.querySelector("#sources");
const sourceCount = document.querySelector("#source-count");
const videoPlayer = document.querySelector("#video-player");
const playerTitle = document.querySelector("#player-title");
const playerTime = document.querySelector("#player-time");
const playerPlaceholder = document.querySelector("#player-placeholder");

function getVideoId(source) {
  if (source.video_id) return source.video_id;
  try {
    return new URL(source.url).searchParams.get("v");
  } catch {
    return null;
  }
}

function getStartSeconds(source) {
  if (Number.isFinite(Number(source.start_seconds))) return Math.max(0, Math.floor(Number(source.start_seconds)));
  try {
    const time = new URL(source.url).searchParams.get("t");
    const seconds = time?.match(/^(\d+)s?$/);
    if (seconds) return Number(seconds[1]);
  } catch {
    // Use the displayed timestamp if the URL has no start time.
  }
  return (source.timestamp || "0:0:0").split(":").reduce((total, part) => total * 60 + (Number(part) || 0), 0);
}

function selectSource(source, selectedButton) {
  const videoId = getVideoId(source);
  if (!videoId || !/^[\w-]{11}$/.test(videoId)) {
    status.textContent = "This result does not include a playable YouTube video.";
    status.className = "status is-error";
    return;
  }

  const startSeconds = getStartSeconds(source);
  // This is called only from a user's result click, so playback never starts
  // when the page loads or when a search merely returns its suggestions.
  videoPlayer.src = `https://www.youtube-nocookie.com/embed/${videoId}?start=${startSeconds}&autoplay=1&playsinline=1&rel=0`;
  videoPlayer.title = source.video_title || "Matching YouTube video";
  playerTitle.textContent = source.video_title || "Playlist video";
  playerTime.textContent = source.timestamp || "";
  playerPlaceholder.hidden = true;
  sourceList.querySelectorAll(".watch-link").forEach((button) => {
    const isSelected = button === selectedButton;
    button.textContent = isSelected ? "Playing" : "Play here";
    button.setAttribute("aria-pressed", String(isSelected));
  });
}

function renderSources(sources = []) {
  sourceList.replaceChildren();
  sourceCount.textContent = `${sources.length} ${sources.length === 1 ? "source" : "sources"}`;

  for (const [index, source] of sources.entries()) {
    const card = document.createElement("article");
    card.className = "source-card";

    const number = document.createElement("span");
    number.className = "source-number";
    number.textContent = String(index + 1).padStart(2, "0");

    const details = document.createElement("div");
    details.className = "source-details";
    const title = document.createElement("h3");
    title.textContent = source.video_title || "Playlist video";
    const time = document.createElement("span");
    time.className = "timestamp";
    time.textContent = source.timestamp || "Open video";
    details.append(title, time);

    const link = document.createElement("button");
    link.className = "watch-link";
    link.type = "button";
    link.textContent = "Play here";
    link.setAttribute("aria-label", `Play ${source.video_title || "video"} here at ${source.timestamp || "the relevant moment"}`);
    link.setAttribute("aria-pressed", "false");
    link.addEventListener("click", () => selectSource(source, link));

    card.append(number, details, link);
    sourceList.append(card);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const statement = statementInput.value.trim();
  if (!statement) return;

  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Searching...";
  status.textContent = "Looking through the playlist captions...";
  status.className = "status is-loading";
  results.hidden = true;
  videoPlayer.removeAttribute("src");
  playerPlaceholder.hidden = false;
  playerTitle.textContent = "Choose a suggested video";
  playerTime.textContent = "";

  try {
    const response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ statement, top_k: 5 }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The search request failed.");

    answerText.textContent = data.answer || "No answer was returned.";
    renderSources(data.sources || []);
    results.hidden = false;
    status.textContent = "";
    status.className = "status";
  } catch (error) {
    status.textContent = error.message || "Could not connect to the search service.";
    status.className = "status is-error";
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "Find it";
  }
});
