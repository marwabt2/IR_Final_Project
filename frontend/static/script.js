// async function sendQuery() {
//   const query = document.getElementById("query").value;
//   const dataset = document.getElementById("dataset").value;
//   const resultDiv = document.getElementById("result");

//   const response = await fetch("/tfidf/search", {
//     method: "POST",
//     headers: {
//       "Content-Type": "application/json",
//     },
//     body: JSON.stringify({
//       query: query,
//       dataset_path: dataset,
//     }),
//   });

//   const data = await response.json();

//   // هذا هو السطر المعدل لاستخراج النصوص فقط
//   const texts = data.top_documents.map((doc) => doc.text);

//   if (!texts || texts.length === 0) {
//     resultDiv.innerHTML = "<p>No results found.</p>";
//     return;
//   }

//   resultDiv.innerHTML = texts
//     .map(
//       (text, index) => `
//       <div class="card">
//         <h3>Result ${index + 1}</h3>
//         <p>${text}</p>
//       </div>
//     `
//     )
//     .join("");
// }

let currentPage = 1;
const resultsPerPage = 4;
let allTexts = [];

async function sendQuery() {
  const query = document.getElementById("query").value;
  const dataset = document.getElementById("dataset").value;
  const resultDiv = document.getElementById("result");
  const paginationDiv = document.getElementById("pagination");
  const api = document.getElementById("search_type").value;
  console.log(api);
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
