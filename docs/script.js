// script.js

// ==========================================================
// ⚠️ SEBI DISCLAIMER
// ==========================================================
document.addEventListener("DOMContentLoaded", () => {
  const banner = document.createElement("div");

  banner.innerHTML = `
    <span style="flex: 1;">
      ⚠️ <strong>Disclaimer:</strong> For educational use only. 
      Consult a SEBI registered adviser before investing.
    </span>
    <button id="closeDisclaimer">✖</button>
  `;

  Object.assign(banner.style, {
    backgroundColor: "#f8d7da",
    color: "#721c24",
    padding: "10px",
    fontSize: "13px",
    position: "sticky",
    top: "0",
    zIndex: "9999",
    display: "flex",
    justifyContent: "center",
    alignItems: "center"
  });

  document.body.prepend(banner);

  document.getElementById("closeDisclaimer").onclick = () => {
    banner.style.display = "none";
  };
});

// ==========================================================
// 📅 Last Updated
// ==========================================================
function updateLastUpdatedFromDB(db) {
  try {
    const result = db.exec(`SELECT MAX(current_date) FROM STOCK_PRICES;`);
    const val = result?.[0]?.values?.[0]?.[0];
    if (!val) return;

    const formatted = new Date(val).toLocaleDateString("en-IN");
    document.getElementById("lastUpdated").textContent =
      `Data updated: ${formatted}`;
  } catch (err) {
    console.error(err);
  }
}

// ==========================================================
// 📥 Load DB
// ==========================================================
async function loadDatabase() {
  const SQL = await initSqlJs({
    locateFile: f => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.8.0/${f}`
  });

  // const branch = window.location.hostname === "localhost"
  //   ? "development"
  //   : "main";
  const branch = "hotfix-max-returns_by-period"  // for testing 


  const url = `https://raw.githubusercontent.com/Deepak-Chaturvedi/Stock-Signals/${branch}/data/stocks.db`;

  const res = await fetch(url);
  const buf = await res.arrayBuffer();

  const db = new SQL.Database(new Uint8Array(buf));

  updateLastUpdatedFromDB(db);

  const result = db.exec(window.QUERY_SIGNAL_ACCUMULATION);

  if (!result.length || !result[0].values.length) {
    console.error("No data returned from query");
    return;
  }

  const cols = result[0].columns;

  const rows = result[0].values.map(r => {
    let obj = {};
    cols.forEach((c, i) => obj[c] = r[i]);
    return obj;
  });

  console.log("Columns:", cols);
  console.log("Row count:", rows.length);

  createFilters(rows);
  renderTable(rows);
}

// ==========================================================
// 🔍 Filters
// ==========================================================
function createFilters(data) {
  const div = document.getElementById("filters");
  div.innerHTML = "";

  const symbols = [...new Set(data.map(d => d["Symbol"]))];

  const select = document.createElement("select");
  select.id = "symbolFilter";

  select.innerHTML =
    `<option value="">All Symbols</option>` +
    symbols.map(s => `<option value="${s}">${s}</option>`).join("");

  div.appendChild(select);

  const date = document.createElement("input");
  date.type = "date";
  date.id = "dateFilter";
  div.appendChild(date);

  const btn = document.createElement("button");
  btn.textContent = "Apply";
  btn.onclick = applyFilters;
  div.appendChild(btn);
}

let table;

// ==========================================================
// 📊 Table
// ==========================================================
function renderTable(data) {

  if (!data.length) {
    console.error("No data to render");
    return;
  }

  table = new Tabulator("#table", {
    data: data,
    layout: "fitDataStretch",
    height: "70vh",

    pagination: "local",
    paginationSize: 25,

    columns: Object.keys(data[0]).map(c => {

      let sorter = "string";

      if (c.includes("%") || c.includes("Price")) {
        sorter = (a, b) => {
          const num = v => parseFloat((v || "0").toString().replace(/[^\d.-]/g, "")) || 0;
          return num(a) - num(b);
        };
      }

      if (c === "Signal Date") sorter = "date";

      return {
        title: c,
        field: c,
        headerFilter: "input",
        sorter: sorter,

        formatter: cell => {
          const val = cell.getValue();

          if (typeof val === "string" && val.includes("%")) {
            const num = parseFloat(val.replace(/[^\d.-]/g, ""));
            if (num > 0) return `<span style="color:green;font-weight:600">${val}</span>`;
            if (num < 0) return `<span style="color:red;font-weight:600">${val}</span>`;
          }

          return val;
        }
      };
    }),

    rowFormatter: row => {
      const d = row.getData();

      const best = parsePercent(d["1M Best %"]);
      const dd = parsePercent(d["Max Drawdown %"]);

      if (best > 10 && dd > -5) {
        row.getElement().style.backgroundColor = "#e8f5e9";
      }
    }
  });

  table.setSort([{ column: "Signal Date", dir: "desc" }]);
}

// ==========================================================
// 🎯 Filters
// ==========================================================
function applyFilters() {
  const sym = document.getElementById("symbolFilter").value;
  const date = document.getElementById("dateFilter").value;

  table.clearFilter();

  if (sym) table.addFilter("Symbol", "=", sym);
  if (date) table.addFilter("Signal Date", "=", date);
}

// ==========================================================
// 🔥 Quick Filters (FIXED)
// ==========================================================
function parsePercent(val) {
  return parseFloat((val || "0").toString().replace(/[^\d.-]/g, "")) || 0;
}

function applyQuickFilter(type) {
  table.clearFilter();

  table.setFilter(data => {
    const best = parsePercent(data["1M Best %"]);
    const dd = parsePercent(data["Max Drawdown %"]);
    const current = parsePercent(data["Current Return %"]);

    if (type === "strong") return best > 10 && dd > -5;
    if (type === "momentum") return current > 5;
    if (type === "lowrisk") return dd > -5;
    return true;
  });
}

// button binding
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("#quickFilters button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#quickFilters button")
        .forEach(b => b.classList.remove("active"));

      btn.classList.add("active");

      applyQuickFilter(btn.dataset.filter);
    });
  });
});

// ==========================================================
// 📥 Export CSV
// ==========================================================
document.getElementById("downloadCSV")?.addEventListener("click", () => {
  if (!table) return;
  table.download("csv", "stock_signals.csv");
});

// ==========================================================
// 🚀 Init
// ==========================================================
window.addEventListener("DOMContentLoaded", loadDatabase);