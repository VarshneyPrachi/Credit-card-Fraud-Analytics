
import React, { useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ShieldCheck,
  LayoutDashboard,
  ChartNoAxesCombined,
  BrainCircuit,
  ScanSearch,
  Activity,
  RefreshCw,
  Download,
  Database,
  CreditCard,
  Percent,
  AlertTriangle,
  Wifi,
  WifiOff,
  LoaderCircle,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
} from 'recharts';
import './style.css';

const API = (
  import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
).replace(/\/$/, '');

const NAV_ITEMS = [
  { name: 'Overview', Icon: LayoutDashboard },
  { name: 'Fraud analytics', Icon: ChartNoAxesCombined },
  { name: 'Model evaluation', Icon: BrainCircuit },
  { name: 'Risk checker', Icon: ScanSearch },
];

const MODEL_RESULTS = [
  {
    name: 'Logistic Regression',
    accuracy: 97.37,
    precision: 5.47,
    recall: 91.49,
    f1: 10.33,
    rocAuc: 98.04,
    averagePrecision: 77.96,
  },
  {
    name: 'Random Forest',
    accuracy: 99.96,
    precision: 93.98,
    recall: 82.98,
    f1: 88.14,
    rocAuc: 97.08,
    averagePrecision: 88.54,
  },
];

const FEATURE_NAMES = Array.from(
  { length: 28 },
  (_, index) => `V${index + 1}`
);

const EMPTY_FEATURES = () =>
  Object.fromEntries(FEATURE_NAMES.map((name) => [name, '']));

const EMPTY_METRICS = {
  total_transactions: null,
  confirmed_fraud: null,
  fraud_rate_percent: null,
};

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatNumber(value) {
  return number(value).toLocaleString('en-IN', {
    maximumFractionDigits: 2,
  });
}

function formatAmount(value) {
  return '$' + number(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatPercent(value) {
  return `${number(value).toFixed(2)}%`;
}

async function getJSON(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API}${path}`, {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    });
  } catch {
    throw new Error(
      `Cannot reach ${API}. Check that FastAPI is running on port 8000.`
    );
  }

  const raw = await response.text();
  let data = {};

  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(`The API returned invalid JSON (HTTP ${response.status}).`);
  }

  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg || JSON.stringify(item)).join('; ')
      : data.detail || data.message;

    throw new Error(
      detail || `API request failed with HTTP ${response.status}.`
    );
  }

  return data;
}

function StatCard({ label, value, Icon, color, description }) {
  return (
    <article className="card stat">
      <div className="stathead">
        {label}
        <span className={`icon ${color}`}>
          <Icon size={18} />
        </span>
      </div>
      <strong>{value}</strong>
      <small>{description}</small>
    </article>
  );
}

function StatusBadge({ connected }) {
  return (
    <span className="statuspill">
      {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
      {connected ? 'LIVE API' : 'API OFFLINE'}
    </span>
  );
}

function App() {
  const [page, setPage] = useState('Overview');

  const [summary, setSummary] = useState(EMPTY_METRICS);
  const [timeline, setTimeline] = useState([]);
  const [amountBuckets, setAmountBuckets] = useState([]);
  const [recentTransactions, setRecentTransactions] = useState([]);

  const [connection, setConnection] = useState('connecting');
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [endpointErrors, setEndpointErrors] = useState({});

  const [riskAmount, setRiskAmount] = useState('125.50');
  const [riskTime, setRiskTime] = useState('5400');
  const [riskModel, setRiskModel] = useState('Random Forest');
  const [threshold, setThreshold] = useState('0.5');
  const [features, setFeatures] = useState(EMPTY_FEATURES);
  const [riskResult, setRiskResult] = useState(null);
  const [riskLoading, setRiskLoading] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setConnection('connecting');
    setError('');

    const endpoints = [
      ['summary', '/api/dashboard/summary'],
      ['timeline', '/api/analytics/timeline'],
      ['amountBuckets', '/api/analytics/amount-buckets'],
      ['recentTransactions', '/api/transactions/recent'],
      ['health', '/api/health'],
    ];

    const results = await Promise.allSettled(
      endpoints.map(async ([key, path]) => ({
        key,
        data: await getJSON(path),
      }))
    );

    const failures = {};
    let summaryLoaded = false;
    let successfulRequests = 0;

    results.forEach((result, index) => {
      const key = endpoints[index][0];

      if (result.status === 'rejected') {
        failures[key] =
          result.reason?.message || 'Endpoint request failed.';
        return;
      }

      successfulRequests += 1;
      const data = result.value.data;

      if (key === 'summary') {
        setSummary((previous) => ({ ...previous, ...data }));
        summaryLoaded = true;
      }

      if (key === 'timeline') {
        setTimeline(Array.isArray(data.data) ? data.data : []);
      }

      if (key === 'amountBuckets') {
        setAmountBuckets(Array.isArray(data.data) ? data.data : []);
      }

      if (key === 'recentTransactions') {
        setRecentTransactions(Array.isArray(data.data) ? data.data : []);
      }
    });

    setEndpointErrors(failures);
    setConnection(summaryLoaded ? 'connected' : 'offline');

    if (!summaryLoaded) {
      setError(
        failures.summary ||
          'Dashboard summary could not be loaded. Check the backend.'
      );
    }

    if (successfulRequests > 0) {
      setLastUpdated(new Date());
    }

    setRefreshing(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function navigate(name) {
    setPage(name);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  const fraudRate = number(summary.fraud_rate_percent);
  const totalTransactions = number(summary.total_transactions);
  const confirmedFraud = number(summary.confirmed_fraud);

  const legitimateTransactions = Math.max(
    0,
    totalTransactions - confirmedFraud
  );

  const chartData = timeline.map((item) => ({
    hour: `${item.hour_bucket ?? '?'}h`,
    fraud: number(item.fraud_count),
    legitimate: number(item.legitimate_count),
  }));

  const hasLiveSummary = connection === 'connected';

  async function checkRisk(event) {
    event.preventDefault();
    setRiskResult(null);

    const missing = FEATURE_NAMES.filter(
      (name) =>
        features[name] === '' ||
        features[name] === null ||
        !Number.isFinite(Number(features[name]))
    );

    if (missing.length > 0) {
      setRiskResult({
        live: false,
        message: `Enter valid numerical values for all 28 features. Missing: ${missing.join(', ')}`,
      });
      return;
    }

    const amount = Number(riskAmount);
    const elapsed = Number(riskTime);
    const selectedThreshold = Number(threshold);

    if (
      !Number.isFinite(amount) ||
      amount < 0 ||
      !Number.isFinite(elapsed) ||
      elapsed < 0
    ) {
      setRiskResult({
        live: false,
        message: 'Amount and elapsed time must be valid non-negative numbers.',
      });
      return;
    }

    const payload = {
      Time: elapsed,
      Amount: amount,
      ...Object.fromEntries(
        FEATURE_NAMES.map((name) => [name, Number(features[name])])
      ),
      model_name: riskModel,
      threshold: selectedThreshold,
    };

    setRiskLoading(true);

    try {
      const result = await getJSON('/api/predict', {
        method: 'POST',
        body: JSON.stringify(payload),
      });

      if (
        result.fraud_probability === undefined ||
        result.predicted_class === undefined
      ) {
        throw new Error(
          'The prediction response is missing its probability or class.'
        );
      }

      setRiskResult({
        live: true,
        probability: number(result.fraud_probability),
        predictedClass: number(result.predicted_class),
        threshold: number(result.threshold ?? selectedThreshold),
        riskLevel: result.risk_level,
        source: result.source,
      });
    } catch (err) {
      setRiskResult({
        live: false,
        message: err.message || 'Prediction request failed.',
      });
    } finally {
      setRiskLoading(false);
    }
  }

  function exportSummary() {
    const report = {
      project: 'FraudGuard',
      exported_at: new Date().toISOString(),
      api_base: API,
      connection_status: connection,
      last_updated: lastUpdated?.toISOString() ?? null,
      summary,
      timeline,
      amount_buckets: amountBuckets,
      recent_transactions: recentTransactions,
      recorded_model_metrics: MODEL_RESULTS,
      note:
        'Model metrics are previously recorded evaluation results. Recent transaction labels are actual dataset labels, not predictions.',
    };

    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], {
        type: 'application/json',
      })
    );

    const link = document.createElement('a');
    link.href = url;
    link.download = 'fraudguard-summary.json';
    link.click();
    URL.revokeObjectURL(url);
  }

  const metrics = [
    {
      label: 'Total transactions',
      value: hasLiveSummary ? formatNumber(totalTransactions) : '—',
      Icon: CreditCard,
      color: 'blue',
      description: hasLiveSummary ? 'Loaded from MySQL API' : 'Awaiting live data',
    },
    {
      label: 'Confirmed fraud',
      value: hasLiveSummary ? formatNumber(confirmedFraud) : '—',
      Icon: AlertTriangle,
      color: 'pink',
      description: 'Transactions labelled as fraud',
    },
    {
      label: 'Fraud rate',
      value: hasLiveSummary ? `${fraudRate.toFixed(3)}%` : '—',
      Icon: Percent,
      color: 'teal',
      description: 'Reported dataset fraud prevalence',
    },
    {
      label: 'Random Forest precision',
      value: '93.98%',
      Icon: BrainCircuit,
      color: 'gold',
      description: 'Recorded held-out test result',
    },
  ];

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">
            <ShieldCheck size={23} />
          </span>
          <div>
            <b>FraudGuard</b>
            <small>INTELLIGENT ANALYTICS</small>
          </div>
        </div>

        <div>
          <label>WORKSPACE</label>
          {NAV_ITEMS.map(({ name, Icon }) => (
            <button
              key={name}
              className={`nav ${page === name ? 'active' : ''}`}
              onClick={() => navigate(name)}
            >
              <Icon size={17} />
              {name}
            </button>
          ))}
        </div>

        <div className="status">
          <i />
          {connection === 'connected'
            ? 'Live API connected'
            : connection === 'connecting'
              ? 'Connecting to API'
              : 'API unavailable'}
          <p>
            {connection === 'connected'
              ? 'Dashboard summary is coming from the backend.'
              : connection === 'connecting'
                ? 'Checking API and loading analytics.'
                : error || 'Start FastAPI and check the API URL.'}
          </p>
          {lastUpdated && (
            <small>Updated {lastUpdated.toLocaleTimeString()}</small>
          )}
        </div>
      </aside>

      <main>
        <header>
          <div>
            <small className="eyebrow">RISK INTELLIGENCE PLATFORM</small>
            <h1>
              {page === 'Overview'
                ? 'Credit card fraud intelligence'
                : page}
            </h1>
            <p>
              Monitor transactions, investigate risk patterns, and evaluate
              fraud detection models.
            </p>
          </div>

          <div className="actions">
            <StatusBadge connected={hasLiveSummary} />
            <button onClick={refresh} disabled={refreshing}>
              {refreshing ? (
                <LoaderCircle size={15} className="spin" />
              ) : (
                <RefreshCw size={15} />
              )}
              Refresh
            </button>
            <button className="primary" onClick={exportSummary}>
              <Download size={15} />
              Export
            </button>
          </div>
        </header>

        {error && <div className="notice">{error}</div>}

        {page === 'Overview' && (
          <>
            <section className="stats">
              {metrics.map(({ label, value, Icon, color, description }) => (
                <StatCard
                  key={label}
                  label={label}
                  value={value}
                  Icon={Icon}
                  color={color}
                  description={description}
                />
              ))}
            </section>

            <section className="grid">
              <article className="card">
                <div className="cardhead">
                  <div>
                    <h2>Fraud activity by elapsed time</h2>
                    <p>
                      {chartData.length
                        ? 'Live counts returned by the timeline endpoint'
                        : 'No live timeline data available'}
                    </p>
                  </div>
                  <Activity color="#22b8ff" />
                </div>

                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={270}>
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="fraudFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#22b8ff" stopOpacity={0.35} />
                          <stop offset="100%" stopColor="#22b8ff" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#20364b" />
                      <XAxis dataKey="hour" stroke="#8fa5bb" />
                      <YAxis stroke="#8fa5bb" />
                      <Tooltip
                        contentStyle={{
                          background: '#0d1d2e',
                          border: '1px solid #20364b',
                          borderRadius: 10,
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="fraud"
                        name="Fraud transactions"
                        stroke="#22b8ff"
                        strokeWidth={2.5}
                        fill="url(#fraudFill)"
                      />
                      <Area
                        type="monotone"
                        dataKey="legitimate"
                        name="Legitimate transactions"
                        stroke="#20e3d2"
                        strokeDasharray="4 4"
                        fill="none"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="footnote">
                    {endpointErrors.timeline || 'No timeline records returned.'}
                  </p>
                )}
                <p className="footnote">
                  Source: `/api/analytics/timeline`
                </p>
              </article>

              <article className="card">
                <h2>Model performance snapshot</h2>
                <p>Recorded held-out test metrics</p>
                {MODEL_RESULTS.map((model) => (
                  <div className="model" key={model.name}>
                    <small>{model.name.toUpperCase()}</small>
                    <strong>
                      {formatPercent(model.f1)} <em>F1-score</em>
                    </strong>
                    <p>
                      Precision {formatPercent(model.precision)} · Recall{' '}
                      {formatPercent(model.recall)}
                    </p>
                    <div className="meter">
                      <i
                        className={model.name === 'Logistic Regression' ? 'cyan' : ''}
                        style={{ width: `${model.f1}%` }}
                      />
                    </div>
                  </div>
                ))}
                <p className="footnote">
                  Metrics are saved project results, not recalculated by the
                  dashboard.
                </p>
              </article>
            </section>

            <section className="grid">
              <article className="card">
                <h2>Class imbalance</h2>
                <p>Based on live summary values</p>

                {hasLiveSummary ? (
                  <>
                    <div className="barline">
                      <span>Legitimate</span>
                      <div>
                        <i
                          style={{
                            width: `${totalTransactions
                              ? (legitimateTransactions / totalTransactions) * 100
                              : 0}%`,
                          }}
                        />
                      </div>
                      <b>{formatNumber(legitimateTransactions)}</b>
                    </div>
                    <div className="barline">
                      <span>Fraud</span>
                      <div>
                        <i
                          className="pinkbar"
                          style={{
                            width: `${totalTransactions
                              ? (confirmedFraud / totalTransactions) * 100
                              : 0}%`,
                            minWidth: confirmedFraud > 0 ? 4 : 0,
                          }}
                        />
                      </div>
                      <b>{formatNumber(confirmedFraud)}</b>
                    </div>
                  </>
                ) : (
                  <p className="footnote">
                    Connect the summary endpoint to show class counts.
                  </p>
                )}
              </article>

              <article className="card">
                <h2>Integration status</h2>
                <p>Current data sources</p>
                <div className="source">
                  <Database />
                  <div>
                    <b>MySQL · fraud_analytics</b>
                    <small>
                      {hasLiveSummary
                        ? 'Summary retrieved from the API'
                        : 'Waiting for the database summary'}
                    </small>
                  </div>
                  {hasLiveSummary ? (
                    <CheckCircle2 color="#20e3d2" />
                  ) : (
                    <XCircle color="#ff6384" />
                  )}
                </div>
                <div className="source">
                  <BrainCircuit />
                  <div>
                    <b>Saved ML models</b>
                    <small>Evaluation metrics from recorded results</small>
                  </div>
                  <span className="badge">Recorded</span>
                </div>
              </article>
            </section>
          </>
        )}

        {page === 'Fraud analytics' && (
          <>
            {endpointErrors.amountBuckets && (
              <div className="notice">{endpointErrors.amountBuckets}</div>
            )}

            <section className="card">
              <div className="cardhead">
                <div>
                  <h2>Transaction amount distribution</h2>
                  <p>Live transaction and fraud counts from MySQL</p>
                </div>
                <Database color="#22b8ff" />
              </div>

              {amountBuckets.length > 0 ? (
                <>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={amountBuckets}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#20364b" />
                      <XAxis dataKey="amount_bucket" stroke="#8fa5bb" />
                      <YAxis stroke="#8fa5bb" />
                      <Tooltip
                        contentStyle={{
                          background: '#0d1d2e',
                          border: '1px solid #20364b',
                          borderRadius: 10,
                        }}
                      />
                      <Bar
                        dataKey="total_transactions"
                        name="Total transactions"
                        fill="#22b8ff"
                        radius={[5, 5, 0, 0]}
                      />
                      <Bar
                        dataKey="fraud_count"
                        name="Fraud transactions"
                        fill="#ff6384"
                        radius={[5, 5, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>

                  <div className="tablewrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Amount range</th>
                          <th>Total transactions</th>
                          <th>Fraud transactions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {amountBuckets.map((item, index) => (
                          <tr key={`${item.amount_bucket}-${index}`}>
                            <td>{item.amount_bucket}</td>
                            <td>{formatNumber(item.total_transactions)}</td>
                            <td>{formatNumber(item.fraud_count)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <p className="footnote">
                  No live amount data loaded. {endpointErrors.amountBuckets || ''}
                </p>
              )}
            </section>

            <section className="card" style={{ marginTop: 16 }}>
              <div className="cardhead">
                <div>
                  <h2>Recent transactions</h2>
                  <p>Latest records returned by the database</p>
                </div>
                <Activity color="#20e3d2" />
              </div>

              <div className="tablewrap">
                <table>
                  <thead>
                    <tr>
                      <th>Transaction ID</th>
                      <th>Elapsed time</th>
                      <th>Amount</th>
                      <th>Actual label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentTransactions.map((item, index) => (
                      <tr key={`${item.transaction_id}-${index}`}>
                        <td>{item.transaction_id}</td>
                        <td>{formatNumber(item.elapsed_seconds)} sec</td>
                        <td>{formatAmount(item.amount)}</td>
                        <td>
                          <span
                            className={`badge ${
                              Number(item.fraud_label) === 0 ? 'safe' : ''
                            }`}
                          >
                            {Number(item.fraud_label) === 1
                              ? 'Confirmed fraud'
                              : 'Legitimate'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {recentTransactions.length === 0 && (
                  <p className="footnote">
                    {endpointErrors.recentTransactions ||
                      'No recent transactions returned.'}
                  </p>
                )}
              </div>
            </section>
          </>
        )}

        {page === 'Model evaluation' && (
          <section className="card">
            <h2>Model evaluation</h2>
            <p>Recorded held-out test-set metrics</p>
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-score</th>
                    <th>ROC-AUC</th>
                    <th>Avg precision</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_RESULTS.map((model) => (
                    <tr key={model.name}>
                      <td>{model.name}</td>
                      <td>{formatPercent(model.accuracy)}</td>
                      <td>{formatPercent(model.precision)}</td>
                      <td>{formatPercent(model.recall)}</td>
                      <td>{formatPercent(model.f1)}</td>
                      <td>{formatPercent(model.rocAuc)}</td>
                      <td>{formatPercent(model.averagePrecision)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="footnote">
              These are recorded evaluation results, not metrics recalculated
              from live transactions.
            </p>
          </section>
        )}

        {page === 'Risk checker' && (
          <section className="grid">
            <article className="card">
              <h2>Transaction risk checker</h2>
              <p>
                Submit the complete feature vector to the trained model.
              </p>

              <form onSubmit={checkRisk}>
                <div className="field">
                  <label htmlFor="riskAmount">Transaction amount</label>
                  <input
                    id="riskAmount"
                    type="number"
                    min="0"
                    step="any"
                    required
                    value={riskAmount}
                    onChange={(e) => setRiskAmount(e.target.value)}
                  />
                </div>

                <div className="field" style={{ marginTop: 14 }}>
                  <label htmlFor="riskTime">Time (elapsed seconds)</label>
                  <input
                    id="riskTime"
                    type="number"
                    min="0"
                    step="any"
                    required
                    value={riskTime}
                    onChange={(e) => setRiskTime(e.target.value)}
                  />
                </div>

                <div className="field" style={{ marginTop: 14 }}>
                  <label htmlFor="riskModel">Model</label>
                  <select
                    id="riskModel"
                    value={riskModel}
                    onChange={(e) => setRiskModel(e.target.value)}
                  >
                    <option>Random Forest</option>
                    <option>Logistic Regression</option>
                  </select>
                </div>

                <div className="field" style={{ marginTop: 14 }}>
                  <label htmlFor="threshold">Decision threshold</label>
                  <select
                    id="threshold"
                    value={threshold}
                    onChange={(e) => setThreshold(e.target.value)}
                  >
                    <option value="0.5">0.50 (default)</option>
                    <option value="0.25">0.25 (higher sensitivity)</option>
                    <option value="0.75">0.75 (fewer alerts)</option>
                  </select>
                </div>

                <details style={{ marginTop: 18 }}>
                  <summary style={{ cursor: 'pointer', marginBottom: 12 }}>
                    Model features V1–V28 (required)
                  </summary>
                  <p className="footnote">
                    Enter the actual values from a compatible transaction
                    record. The zero defaults below are placeholders only.
                    Predictions made with placeholder values are not
                    representative of a real transaction.
                  </p>
                  <div className="feature-grid">
                    {FEATURE_NAMES.map((name) => (
                      <div className="field" key={name}>
                        <label htmlFor={`feature-${name}`}>{name}</label>
                        <input
                          id={`feature-${name}`}
                          type="number"
                          step="any"
                          required
                          value={features[name]}
                          placeholder="Required"
                          onChange={(e) =>
                            setFeatures((previous) => ({
                              ...previous,
                              [name]: e.target.value,
                            }))
                          }
                        />
                      </div>
                    ))}
                  </div>
                </details>

                <button
                  type="submit"
                  className="primary"
                  disabled={riskLoading}
                  style={{ marginTop: 18 }}
                >
                  {riskLoading ? 'Checking model…' : 'Check transaction risk'}
                </button>
              </form>

              {riskResult && (
                <div className="notice" style={{ marginTop: 16 }}>
                  {riskResult.live ? (
                    <>
                      <h3>
                        Fraud probability:{' '}
                        {(riskResult.probability * 100).toFixed(2)}%
                      </h3>
                      <p>
                        {riskResult.predictedClass === 1
                          ? 'The backend model flagged this transaction.'
                          : 'The backend model classified this transaction below the selected threshold.'}
                      </p>
                      <p>
                        Decision threshold:{' '}
                        {(riskResult.threshold * 100).toFixed(0)}%
                      </p>
                      {riskResult.riskLevel && (
                        <p>Risk level: {riskResult.riskLevel}</p>
                      )}
                      {riskResult.source && <p>Source: {riskResult.source}</p>}
                    </>
                  ) : (
                    <p>{riskResult.message}</p>
                  )}
                </div>
              )}

              <p className="footnote">
                This form sends Time, Amount, and V1–V28. Use the same
                preprocessing and feature values used during model training.
              </p>
            </article>

            <article className="card">
              <h2>Prediction pipeline</h2>
              <p>Requirements for reliable model inference</p>
              <div className="source">
                <ScanSearch />
                <div>
                  <b>1. Validate features</b>
                  <small>Check the exact feature schema and input ranges.</small>
                </div>
              </div>
              <div className="source">
                <BrainCircuit />
                <div>
                  <b>2. Load the saved model</b>
                  <small>
                    Apply the same preprocessing used during training.
                  </small>
                </div>
              </div>
              <div className="source">
                <ShieldCheck />
                <div>
                  <b>3. Return a risk result</b>
                  <small>Probability, threshold, class, and limitations.</small>
                </div>
              </div>
              <button
                type="button"
                onClick={async () => {
                  try {
                    const health = await getJSON('/api/health');
                    setRiskResult({
                      live: false,
                      message: `Health endpoint responded: ${JSON.stringify(health)}`,
                    });
                  } catch (err) {
                    setRiskResult({ live: false, message: err.message });
                  }
                }}
                style={{ marginTop: 18 }}
              >
                Test backend health
              </button>
            </article>
          </section>
        )}

        <footer>
          FraudGuard · Portfolio application
          <span>API: {API}</span>
        </footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
