const input = document.getElementById("query");
const autocompleteBox = document.getElementById("autocompleteBox");
const correctionsBox = document.getElementById("corrections");
const suggestionsDiv = document.getElementById("suggestions");
input.addEventListener("input", async () => {
  const query = input.value.trim();
  if (!query) {
    autocompleteBox.innerHTML = "";
    correctionsBox.innerHTML = "";
    suggestionsDiv.innerHTML = "";
    return;
  }
  const dataset = document.getElementById("dataset").value;

  // Autocomplete
  // const autoRes = await fetch(`/api/autocomplete?prefix=${encodeURIComponent(query)}`);
  const autoRes = await fetch(
    `/api/autocomplete?prefix=${encodeURIComponent(
      query
    )}&dataset=${encodeURIComponent(dataset)}`
  );
  const autoData = await autoRes.json();
  autocompleteBox.innerHTML = "";
  autoData.results.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    li.onclick = () => {
      input.value = text;
      autocompleteBox.innerHTML = "";
      fetchSuggestions(text);
    };
    autocompleteBox.appendChild(li);
  });

  // Suggestions
  if (query.length > 2) fetchSuggestions(query);
});

async function fetchSuggestions(query) {
  const dataset = document.getElementById("dataset").value;
  // const res = await fetch(`/api/suggest?q=${encodeURIComponent(query)}`);
  const res = await fetch(
    `/api/suggest?q=${encodeURIComponent(query)}&dataset=${encodeURIComponent(
      dataset
    )}`
  );
  const data = await res.json();

  // فقط التصحيح الإملائي
  if (data.corrected !== query) {
    correctionsBox.innerHTML = `<strong>Spelling:</strong> ${data.corrected}`;
  } else {
    correctionsBox.innerHTML = "";
  }

  // اقتراحات
  suggestionsDiv.innerHTML = "";
  data.suggestions.forEach(([text]) => {
    const tag = document.createElement("div");
    tag.className = "suggestion-tag";
    tag.textContent = text;
    tag.onclick = () => {
      input.value = text;
      suggestionsDiv.innerHTML = "";
      fetchSuggestions(text);
    };
    suggestionsDiv.appendChild(tag);
  });
}

// تابع البحث الأساسي
let currentPage = 1;
const resultsPerPage = 4;
let allTexts = [];

async function sendQuery() {
  const query = input.value.trim();
  const dataset = document.getElementById("dataset").value;
  const resultDiv = document.getElementById("result");
  const paginationDiv = document.getElementById("pagination");
  const api = document.getElementById("search_type").value;
  const response = await fetch(api, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, dataset_path: dataset }),
  });

  const data = await response.json();
  allTexts = data.top_documents.map((doc) => doc.text);

  if (!allTexts.length) {
    resultDiv.innerHTML = "<p>No results found.</p>";
    paginationDiv.innerHTML = "";
    return;
  }

  currentPage = 1;
  renderPage();
  renderPaginationControls();
}

function renderPage() {
  const resultDiv = document.getElementById("result");
  const start = (currentPage - 1) * resultsPerPage;
  const end = start + resultsPerPage;
  const pageItems = allTexts.slice(start, end);

  resultDiv.innerHTML = pageItems
    .map(
      (text, i) => `
            <div class="card">
              <h3>Result ${start + i + 1}</h3>
              <p>${text}</p>
            </div>
          `
    )
    .join("");
}

function renderPaginationControls() {
  const paginationDiv = document.getElementById("pagination");
  const totalPages = Math.ceil(allTexts.length / resultsPerPage);

  let buttons = "";
  if (currentPage > 1) {
    buttons += `<button onclick="prevPage()">Previous</button>`;
  }
  if (currentPage < totalPages) {
    buttons += `<button onclick="nextPage()">Next</button>`;
  }

  paginationDiv.innerHTML = buttons;
}

function nextPage() {
  const totalPages = Math.ceil(allTexts.length / resultsPerPage);
  if (currentPage < totalPages) {
    currentPage++;
    renderPage();
    renderPaginationControls();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function prevPage() {
  if (currentPage > 1) {
    currentPage--;
    renderPage();
    renderPaginationControls();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}
