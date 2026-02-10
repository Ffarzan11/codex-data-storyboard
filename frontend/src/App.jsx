import { useMemo, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  Legend,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

function toTitleCaseFromSnake(text) {
  if (!text) return "";
  return text
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function paragraphToBullets(text) {
  if (!text) return [];
  const parts = text
    .split(/(?<=[.!?])\s+/)
    .map((line) => line.trim())
    .filter(Boolean);

  const merged = [];
  for (const part of parts) {
    const isExample = /^example\b[:\s-]*/i.test(part);
    if (isExample && merged.length) {
      merged[merged.length - 1] = `${merged[merged.length - 1]} ${part}`;
    } else {
      merged.push(part);
    }
  }

  return merged;
}

function renderChart(chart) {
  if (!chart?.data?.length) return <p>No chart data available.</p>;

  if (chart.type === "bar") {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chart.data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d5dce8" />
          <XAxis dataKey={chart.x} tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Bar dataKey={chart.y} fill="#0ea5e9" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chart.type === "scatter") {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart>
          <CartesianGrid stroke="#d5dce8" />
          <XAxis type="number" dataKey={chart.x} name={chart.x} tick={{ fontSize: 12 }} />
          <YAxis type="number" dataKey={chart.y} name={chart.y} tick={{ fontSize: 12 }} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter data={chart.data} fill="#f97316" />
        </ScatterChart>
      </ResponsiveContainer>
    );
  }

  if (chart.type === "line") {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chart.data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d5dce8" />
          <XAxis dataKey={chart.x} tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey={chart.y}
            stroke="#06b6d4"
            strokeWidth={3}
            dot={{ r: 2 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  return <p>Unsupported chart type.</p>;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const metrics = useMemo(() => {
    if (!result) return [];
    return [
      { label: "Rows Before", value: result.cleaning.rows_before },
      { label: "Rows After", value: result.cleaning.rows_after },
      { label: "Duplicates Removed", value: result.cleaning.duplicates_removed },
      { label: "Fields Removed", value: result.cleaning.removed_columns },
      { label: "Missing Fixed", value: result.cleaning.missing_before - result.cleaning.missing_after },
    ];
  }, [result]);

  const uploadAndAnalyze = async () => {
    if (!file) {
      setError("Please choose a CSV file first.");
      return;
    }

    setError("");
    setLoading(true);
    setResult(null);

    const data = new FormData();
    data.append("file", file);

    try {
      const response = await axios.post(`${API_BASE}/analyze`, data, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch (e) {
      setError(e?.response?.data?.error || "Analysis failed. Please try another file.");
    } finally {
      setLoading(false);
    }
  };

  const downloadCleaned = async () => {
    if (!result?.analysis_id || !result?.downloads?.cleaned_csv) return;
    try {
      const response = await axios.get(`${API_BASE}${result.downloads.cleaned_csv}`, {
        responseType: "blob",
      });
      const url = URL.createObjectURL(new Blob([response.data], { type: "text/csv;charset=utf-8;" }));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `cleaned_${file?.name || "dataset.csv"}`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (_e) {
      setError("Unable to download cleaned CSV. Please re-run analysis.");
    }
  };

  return (
    <div className="page">
      <div className="glow glow-a" />
      <div className="glow glow-b" />
      <div className="float-dot dot-a" />
      <div className="float-dot dot-b" />
      <div className="float-dot dot-c" />

      <motion.header
        className="hero"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="hero-layout">
          <div>
            <h1>Data Storyboard Studio</h1>
            <p>
              Upload a CSV, auto-clean it, explore EDA visuals, and generate a clear data story with
              conclusions.
            </p>
          </div>
          <motion.div
            className="hero-visual"
            initial={{ opacity: 0, scale: 0.95, rotate: -1 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            <svg viewBox="0 0 380 210" role="img" aria-label="Data analysis illustration">
              <rect x="8" y="8" width="364" height="194" rx="18" fill="#f8fdff" stroke="#d7ebfa" />
              <rect x="26" y="28" width="158" height="74" rx="10" fill="#ecfeff" stroke="#a5f3fc" />
              <rect x="200" y="28" width="154" height="74" rx="10" fill="#fef9c3" stroke="#fde68a" />
              <rect x="26" y="116" width="328" height="66" rx="10" fill="#eef2ff" stroke="#c7d2fe" />

              <line x1="45" y1="164" x2="337" y2="164" stroke="#94a3b8" strokeWidth="2" />
              <circle cx="76" cy="156" r="5" fill="#06b6d4" />
              <circle cx="138" cy="145" r="5" fill="#06b6d4" />
              <circle cx="200" cy="150" r="5" fill="#06b6d4" />
              <circle cx="262" cy="135" r="5" fill="#06b6d4" />
              <circle cx="324" cy="126" r="5" fill="#06b6d4" />
              <polyline
                points="76,156 138,145 200,150 262,135 324,126"
                fill="none"
                stroke="#0284c7"
                strokeWidth="3"
                strokeLinecap="round"
              />

              <rect x="43" y="50" width="18" height="34" rx="4" fill="#22d3ee" />
              <rect x="68" y="42" width="18" height="42" rx="4" fill="#22d3ee" />
              <rect x="93" y="35" width="18" height="49" rx="4" fill="#22d3ee" />
              <rect x="118" y="57" width="18" height="27" rx="4" fill="#22d3ee" />
              <rect x="143" y="47" width="18" height="37" rx="4" fill="#22d3ee" />

              <circle cx="235" cy="63" r="16" fill="#f59e0b" opacity="0.25" />
              <circle cx="260" cy="63" r="16" fill="#f59e0b" opacity="0.45" />
              <circle cx="285" cy="63" r="16" fill="#f59e0b" opacity="0.65" />
              <circle cx="310" cy="63" r="16" fill="#f59e0b" opacity="0.85" />
            </svg>
          </motion.div>
        </div>
      </motion.header>

      <motion.section
        className="panel upload"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1, duration: 0.5 }}
      >
        <label className="file-picker">
          <span>{file ? file.name : "Choose CSV file"}</span>
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
        </label>

        <button type="button" disabled={loading} onClick={uploadAndAnalyze}>
          {loading ? "Analyzing..." : "Analyze & Build Storyboard"}
        </button>
        {result && (
          <button type="button" className="secondary-btn" onClick={downloadCleaned}>
            Download Cleaned Dataset
          </button>
        )}
      </motion.section>

      {error && <p className="error">{error}</p>}
      {loading && (
        <motion.div
          className="loading-strip"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
        >
          <span>Analyzing your data and crafting a story...</span>
          <div className="loading-bar" />
        </motion.div>
      )}

      {result && (
        <motion.main
          className="results"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
        >
          <section className="panel stats-grid">
            {metrics.map((item, idx) => (
              <motion.article
                className="stat-card"
                key={item.label}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.06 }}
                whileHover={{ y: -4, scale: 1.015 }}
              >
                <h3>{item.value}</h3>
                <p>{item.label}</p>
              </motion.article>
            ))}
          </section>

          <section className="panel">
            <h2>Interactive EDA</h2>
            <div className="chart-grid">
              {result.charts.map((chart, idx) => (
                <motion.article
                  className="chart-card"
                  key={chart.id}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 + idx * 0.08 }}
                  whileHover={{ y: -4 }}
                >
                  <h3>{chart.title}</h3>
                  {renderChart(chart)}
                </motion.article>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Your Data Story In Plain English</h2>
            <div className="story-grid">
              <article className="story-card">
                <h3>Field Guide</h3>
                <p>
                  We reviewed {result.profile?.rows ?? 0} website records and focused on the fields that matter
                  most.
                </p>
                <p>Below is a guide to what each key field means:</p>
                <ul>
                  {(result.field_analysis?.field_dictionary || []).slice(0, 8).map((item) => (
                    <li key={item.field}>
                      {toTitleCaseFromSnake(item.field)}:{" "}
                      {String(item.plain_meaning || "")
                        .replace(/^[^:]*:\s*/i, "")
                        .replace(/\s+Example value:.*$/i, "")
                        .trim()}
                      {item.example ? ` Example value: ${item.example}.` : ""}
                    </li>
                  ))}
                </ul>
              </article>
              <article className="story-card">
                <h3>What We Learned</h3>
                <ul>
                  {paragraphToBullets(result.simple_story?.what_we_learned).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </article>
              <article className="story-card">
                <h3>Final Takeaway</h3>
                <ul>
                  {paragraphToBullets(result.simple_story?.final_takeaway).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </article>
              <article className="story-card">
                <h3>Important Caveats</h3>
                <ul>
                  {paragraphToBullets(result.simple_story?.important_caveats).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </article>
            </div>
          </section>

          <section className="panel">
            <h2>Storyboard</h2>
            <div className="story-grid">
              <article className="story-card">
                <h3>Simple Summary</h3>
                <p>{result.storyboard.context}</p>
              </article>
              <article className="story-card">
                <h3>Key Fields We Focused On</h3>
                <div className="chips">
                  {result.storyboard.key_fields.map((field) => (
                    <span className="chip" key={field}>
                      {field}
                    </span>
                  ))}
                </div>
              </article>
              <article className="story-card">
                <h3>Removed Low-Value Fields</h3>
                <ul>
                  {(result.storyboard.removed_fields.length
                    ? result.storyboard.removed_fields
                    : ["No fields removed"]
                  ).map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </article>
              <article className="story-card">
                <h3>What Matters Most</h3>
                <ul>
                  {result.storyboard.key_findings.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </article>
              <article className="story-card">
                <h3>Top Relationships</h3>
                <ul>
                  {(result.field_analysis.top_relationships.length
                    ? result.field_analysis.top_relationships.map((r) => {
                        const strength = Math.round((r.score || 0) * 100);
                        if (r.type === "numeric_numeric") {
                          return `${r.field_a} ↔ ${r.field_b}: ${strength}% (Pearson ${r.pearson.toFixed(
                            2
                          )}, Spearman ${r.spearman.toFixed(2)})`;
                        }
                        if (r.type === "categorical_numeric") {
                          return `${r.field_a} → ${r.field_b}: ${strength}% effect (eta)`;
                        }
                        return `${r.field_a} ↔ ${r.field_b}: ${strength}% association (Cramer's V)`;
                      })
                    : ["No strong relationships found"]
                  ).map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              </article>
              <article className="story-card">
                <h3>Risks & Caveats</h3>
                <ul>
                  {(result.storyboard.risks.length ? result.storyboard.risks : ["No major caveats detected."]).map(
                    (r) => (
                      <li key={r}>{r}</li>
                    )
                  )}
                </ul>
              </article>
              <article className="story-card">
                <h3>Conclusion</h3>
                <p>{result.storyboard.conclusion}</p>
              </article>
              <article className="story-card">
                <h3>Recommended Actions</h3>
                <ul>
                  {result.storyboard.next_steps.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
              </article>
            </div>
          </section>

          <section className="panel">
            <h2>EDA Best Practices Check</h2>
            <div className="story-grid">
              <article className="story-card">
                <h3>Data Quality</h3>
                <ul>
                  {result.eda_best_practices.quality_checks.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </article>
              <article className="story-card">
                <h3>Distribution Diagnostics</h3>
                <ul>
                  {(result.eda_best_practices.distribution_checks.length
                    ? result.eda_best_practices.distribution_checks.map(
                        (d) =>
                          `${d.field}: outliers ${Math.round((d.outlier_rate || 0) * 100)}%, skew ${(
                            d.skew || 0
                          ).toFixed(2)}`
                      )
                    : ["No numeric distribution diagnostics available"]
                  ).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </article>
              <article className="story-card">
                <h3>Relationship Testing</h3>
                <ul>
                  {(result.eda_best_practices.relationship_checks.length
                    ? result.eda_best_practices.relationship_checks.map((r) => {
                        const strength = Math.round((r.score || 0) * 100);
                        if (r.type === "numeric_numeric") {
                          return `${r.field_a} & ${r.field_b}: ${strength}% (${r.direction})`;
                        }
                        if (r.type === "categorical_numeric") {
                          return `${r.field_a} impact on ${r.field_b}: ${strength}%`;
                        }
                        return `${r.field_a} and ${r.field_b}: ${strength}% categorical link`;
                      })
                    : ["No strong tested relationships"]
                  ).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </article>
            </div>
          </section>

          <section className="panel">
            <h2>Statistical Modeling Snapshot</h2>
            {result.modeling.status === "ok" ? (
              <div className="story-grid">
                {result.modeling.models.map((model) => (
                  <article className="story-card" key={model.target}>
                    <h3>Target: {model.target}</h3>
                    <ul>
                      <li>Rows used: {model.rows}</li>
                      <li>Train R²: {model.r2_train.toFixed(3)}</li>
                      <li>Adjusted Train R²: {model.adjusted_r2_train.toFixed(3)}</li>
                      <li>
                        Test R²:{" "}
                        {model.r2_test === null || model.r2_test === undefined
                          ? "N/A"
                          : model.r2_test.toFixed(3)}
                      </li>
                    </ul>
                    <p><strong>Top Predictors</strong></p>
                    <ul>
                      {model.coefficients.slice(0, 4).map((c) => (
                        <li key={c.field}>
                          {c.field}: {c.coefficient.toFixed(3)} ({c.impact_direction})
                        </li>
                      ))}
                    </ul>
                  </article>
                ))}
              </div>
            ) : (
              <p>{result.modeling.message || "Modeling could not be completed for this dataset."}</p>
            )}
          </section>

          <section className="panel">
            <h2>Cleaned Data Preview</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {result.preview.columns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.preview.rows.slice(0, 12).map((row, idx) => (
                    <tr key={`row-${idx}`}>
                      {result.preview.columns.map((col) => (
                        <td key={`${idx}-${col}`}>{String(row[col] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </motion.main>
      )}
    </div>
  );
}
