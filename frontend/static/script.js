async function sendQuery() {
  const query = document.getElementById("query").value;
  const dataset = document.getElementById("dataset").value;
  const resultDiv = document.getElementById("result");

  const response = await fetch("/process-text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: query,
      dataset_path: dataset,
    }),
  });

  const data = await response.json();
  const results = data.processed_text;

  if (!results || results.length === 0) {
    resultDiv.innerHTML = "<p>No results found.</p>";
    return;
  }

  resultDiv.innerHTML = results
    .map(
      (res, index) => `
      <div class="card">
        <h3>Result ${index + 1}</h3>
        <p><strong>Doc ID:</strong> ${res.doc_id}</p>
        <p><strong>Similarity Score:</strong> ${res.score}</p>
        <p>${res.text}</p>
      </div>
    `
    )
    .join("");
}
