// script.js
//Deepak-Chaturvedi/Stock-Signals

// --- STEP 1: Load SQLite DB from GitHub and initialize SQL.js ---
async function loadDatabase() {
  const sqlPromise = initSqlJs({
    locateFile: file => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.8.0/${file}`
  });

  // 👇 Replace with your actual GitHub username and repo name
  const dbUrl = "https://raw.githubusercontent.com/Deepak-Chaturvedi/Stock-Signals/main/data/stocks.db";

  const dataPromise = fetch(dbUrl).then(res => {
    if (!res.ok) throw new Error("Database not found or inaccessible");
    return res.arrayBuffer();
  });

  const [SQL, buf] = await Promise.all([sqlPromise, dataPromise]);
  const db = new SQL.Database(new Uint8Array(buf));

  // --- STEP 2: Query your specific table ---
  const tableName = "SIGNAL_ACCUMULATION_STEADY";

  // Get all column names
  const colQuery = db.exec(`PRAGMA table_info(${tableName});`);
  if (colQuery.length === 0) {
    alert(`Table ${tableName} not found in database.`);
    return;
  }

  const columns = colQuery[0].values.map(col => col[1]);

  // Fetch rows
  const dataQuery = db.exec(`SELECT * FROM ${tableName} LIMIT 1000;`);
  if (dataQuery.length === 0) {
    alert(`No data found in table ${tableName}`);
    return;
  }

  const rows = dataQuery[0].values.map(row => {
    const obj = {};
    columns.forEach((c, i) => obj[c] = row[i]);
    return obj;
  });

  // --- STEP 3: Build dropdown filters dynamically ---
  createFilters(rows, columns);

  // --- STEP 4: Initialize Tabulator Table ---
  renderTable(rows, columns);
}

// --- Filter creation ---
function createFilters(data, columns) {
  const filterDiv = document.getElementById("filters");
  filterDiv.innerHTML = ""; // clear old

  // Create dropdown for 'symbol' (if exists)
  if (columns.includes("symbol")) {
    const symbols = [...new Set(data.map(d => d.symbol))].filter(Boolean);
    const symbolSelect = document.createElement("select");
    symbolSelect.id = "symbolFilter";
    symbolSelect.innerHTML = `<option value="">All Symbols</option>` +
      symbols.map(s => `<option value="${s}">${s}</option>`).join("");
    filterDiv.appendChild(symbolSelect);
  }

  // Create date filter (if 'date' or 'trade_date' exists)
  const dateCol = columns.find(c => c.toLowerCase().includes("date"));
  if (dateCol) {
    const input = document.createElement("input");
    input.type = "date";
    input.id = "dateFilter";
    filterDiv.appendChild(input);
  }

  const btn = document.createElement("button");
  btn.textContent = "Apply Filters";
  btn.onclick = applyFilters;
  filterDiv.appendChild(btn);
}

let table; // global table reference

// --- Render table with Tabulator ---
function renderTable(data, columns) {
  table = new Tabulator("#table", {
    data: data,
    layout: "fitDataStretch",
    pagination: "local",
    paginationSize: 15,
    columns: columns.map(c => ({
      title: c,
      field: c,
      headerFilter: "input",
      sorter: "string",
      headerSort: true
    })),
  });
}

// --- Filter logic ---
function applyFilters() {
  const symbolVal = document.getElementById("symbolFilter")?.value || "";
  const dateVal = document.getElementById("dateFilter")?.value || "";

  table.clearFilter();

  if (symbolVal) {
    table.addFilter("symbol", "=", symbolVal);
  }
  if (dateVal) {
    const dateCol = table.getColumns().find(c => c.getField().toLowerCase().includes("date"));
    if (dateCol) {
      table.addFilter(dateCol.getField(), "=", dateVal);
    }
  }
}

// --- Auto load data on page load ---
window.addEventListener("DOMContentLoaded", loadDatabase);

