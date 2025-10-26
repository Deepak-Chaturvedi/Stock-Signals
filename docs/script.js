// script.js
// Deepak-Chaturvedi/Stock-Signals

// --- STEP 1: Load SQLite DB from GitHub and initialize SQL.js ---
async function loadDatabase() {
  const sqlPromise = initSqlJs({
    locateFile: file => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.8.0/${file}`
  });

  // Detect environment (localhost → development)
  const isDev = window.location.hostname === "localhost";
  const branch = isDev ? "development" : "main";

  // ✅ Dynamic DB path (auto-switches based on environment)
  const dbUrl = `https://raw.githubusercontent.com/Deepak-Chaturvedi/Stock-Signals/${branch}/data/stocks.db`;
  console.log("Using DB URL:", dbUrl);

  // Load database
  const dataPromise = fetch(dbUrl).then(res => {
    if (!res.ok) throw new Error("Database not found or inaccessible");
    return res.arrayBuffer();
  });

  const [SQL, buf] = await Promise.all([sqlPromise, dataPromise]);
  const db = new SQL.Database(new Uint8Array(buf));

  // --- STEP 2: Query your specific table ---
  const tableName = "SIGNAL_RETURNS"; // ✅ matches the new query source table

  // Check if table exists
  const colQuery = db.exec(`PRAGMA table_info(${tableName});`);
  if (colQuery.length === 0) {
    alert(`Table ${tableName} not found in database.`);
    return;
  }

  // ✅ Columns now match your query fields
  const columns = [
    "Symbol", 
    "Name",
    "Signal Type",
    "Signal Date",
    "Signal Price",
    "Current Price",
    "1 Week Return %",
    "2 Week Return %",
    "1 Month Return %",
    "3 Month Return %",
    "6 Month Return %",
    "1 Year Return %",
    "Return Since Signal %"
  ];

  // ✅ Execute your global query
  const dataQuery = db.exec(window.QUERY_SIGNAL_ACCUMULATION);

  if (dataQuery.length === 0) {
    alert(`No data found in table ${tableName}`);
    return;
  }

  if (dataQuery.length > 0) {
  console.log("Row count:", dataQuery[0].values.length);

  // Check if the last column is percentage values
  const allReturns = dataQuery[0].values.map(r => parseFloat((r.at(-1) || "0").toString().replace("%", "")));
  const maxReturn = Math.max(...allReturns);
  console.log("Max Return % from DB:", maxReturn);
  }


  // Map rows to objects for Tabulator
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
  filterDiv.innerHTML = ""; // clear old filters

  // ✅ Case-insensitive check for column names
  const symbolCol = columns.find(c => c.toLowerCase().includes("symbol"));
  if (symbolCol) {
    const symbols = [...new Set(data.map(d => d[symbolCol]))].filter(Boolean);
    const symbolSelect = document.createElement("select");
    symbolSelect.id = "symbolFilter";
    symbolSelect.innerHTML =
      `<option value="">All Symbols</option>` +
      symbols.map(s => `<option value="${s}">${s}</option>`).join("");
    filterDiv.appendChild(symbolSelect);
  }

  // ✅ Date filter for any date-like column
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

let table; // global reference

// --- Render table with Tabulator (fixed sorting) ---
function renderTable(data, columns) {
  table = new Tabulator("#table", {
    data: data,
    layout: "fitDataStretch",
    pagination: "local",
    paginationSize: 15,
    columns: columns.map(c => {
      let sorterType = "string";
      let sorterFunc = undefined;

      // Detect numeric or percentage columns
      if (c.toLowerCase().includes("price") || c.toLowerCase().includes("return")) {
        sorterType = "number";
        // Custom numeric sorter to handle "%" or stringified numbers
        sorterFunc = (a, b, aRow, bRow, column, dir, sorterParams) => {
          const toNum = val => {
            if (typeof val === "string") {
              // return parseFloat(val.replace("%", "").replace(",", "")) || 0;
              return parseFloat(val.replace(/[^\d.-]/g, "")) || 0; // remove ALL non-numeric chars
            }
            return val || 0;
          };
          return toNum(a) - toNum(b);
        };
      } 
      // Detect date columns
      else if (c.toLowerCase().includes("date")) {
        sorterType = "date";
      }

      return {
        title: c,
        field: c,
        headerFilter: "input",
        sorter: sorterFunc ? sorterFunc : sorterType,
        headerSort: true,
      };
    }),
  });

  // ✅ Default sort: latest Signal Date descending
  const dateCol = columns.find(c => c.toLowerCase().includes("date"));
  if (dateCol) {
    table.setSort([{ column: dateCol, dir: "desc" }]);
  }
}


// --- Filter logic ---
function applyFilters() {
  const symbolVal = document.getElementById("symbolFilter")?.value || "";
  const dateVal = document.getElementById("dateFilter")?.value || "";

  table.clearFilter();

  if (symbolVal) {
    const symbolCol = table.getColumns().find(c => c.getField().toLowerCase().includes("symbol"));
    if (symbolCol) table.addFilter(symbolCol.getField(), "=", symbolVal);
  }

  if (dateVal) {
    const dateCol = table.getColumns().find(c => c.getField().toLowerCase().includes("date"));
    if (dateCol) table.addFilter(dateCol.getField(), "=", dateVal);
  }
}

// --- Auto load data on page load ---
window.addEventListener("DOMContentLoaded", loadDatabase);
