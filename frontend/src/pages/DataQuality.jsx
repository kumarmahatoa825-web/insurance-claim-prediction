import React, { useEffect } from 'react';
import { getDataQuality } from '../services/api';
import { useApi } from '../hooks/useApi';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ErrorMessage from '../components/ui/ErrorMessage';
import { AlertTriangle, CheckCircle, Info, BarChart2 } from 'lucide-react';

const ICON_MAP = { warning: AlertTriangle, info: Info, ok: CheckCircle };
const COLOR_MAP = {
  warning: 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200',
  info: 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 border-blue-200',
  ok: 'text-green-600 bg-green-50 dark:bg-green-900/20 border-green-200',
};

function WarningCard({ item }) {
  const Icon = ICON_MAP[item.level] || Info;
  const color = COLOR_MAP[item.level] || COLOR_MAP.info;
  return (
    <div className={`flex items-start gap-3 p-3 rounded-lg border ${color}`}>
      <Icon className="w-4 h-4 mt-0.5 flex-shrink-0" />
      <p className="text-sm">{item.message}</p>
    </div>
  );
}

function StatBadge({ value, max, label, color = '#6366f1' }) {
  const pct = max ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
        <span>{label}</span><span>{value} ({pct.toFixed(1)}%)</span>
      </div>
      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className="h-2 rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

const DataQuality = () => {
  const { data: report, loading, error, execute: fetchData } = useApi(getDataQuality);

  useEffect(() => { fetchData(); }, [fetchData]);

  if (loading) return <LoadingSpinner size="lg" className="mt-20" />;
  if (error) return <ErrorMessage message={error} onRetry={fetchData} />;
  if (!report) return null;

  const summary = report.summary || {};
  const warnings = report.warnings || [];
  const colStats = report.column_stats || {};
  const dist = report.class_distribution || {};
  const outliers = report.outliers || {};

  const numericalCols = Object.entries(colStats).filter(([, s]) => s.type === 'numerical');
  const categoricalCols = Object.entries(colStats).filter(([, s]) => s.type === 'categorical');

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Data Quality</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Comprehensive quality report for the training dataset.
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Rows', value: summary.total_rows?.toLocaleString() || '—' },
          { label: 'Total Columns', value: summary.total_columns || '—' },
          { label: 'Missing Values', value: summary.total_missing || 0, warn: summary.total_missing > 0 },
          { label: 'Duplicates', value: summary.total_duplicates || 0, warn: summary.total_duplicates > 0 },
        ].map(s => (
          <div key={s.label} className={`rounded-lg p-4 shadow text-center ${s.warn && s.value > 0 ? 'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200' : 'bg-white dark:bg-gray-800'}`}>
            <p className={`text-2xl font-bold ${s.warn && s.value > 0 ? 'text-yellow-600' : 'text-gray-900 dark:text-white'}`}>{s.value}</p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Warnings */}
      {warnings.length > 0 ? (
        <div className="space-y-2">
          <h2 className="font-semibold text-gray-900 dark:text-white text-sm">Quality Warnings ({warnings.length})</h2>
          {warnings.map((w, i) => <WarningCard key={i} item={w} />)}
        </div>
      ) : (
        <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 rounded-lg">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <p className="text-sm text-green-700 dark:text-green-400">No significant data quality issues detected.</p>
        </div>
      )}

      {/* Class distribution */}
      {Object.keys(dist).length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
          <h2 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-indigo-500" /> Target Class Distribution
          </h2>
          <div className="space-y-3">
            {Object.entries(dist.counts || {}).map(([cls, count]) => (
              <StatBadge
                key={cls}
                label={cls === '0' ? 'No Claim (0)' : 'Claim (1)'}
                value={count}
                max={dist.total || 1}
                color={cls === '0' ? '#22c55e' : '#ef4444'}
              />
            ))}
          </div>
          {dist.imbalance_ratio && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-3">
              Imbalance ratio: {dist.imbalance_ratio.toFixed(2)}:1 —
              {dist.is_imbalanced ? ' Class imbalance detected. Models use class weighting.' : ' Classes are relatively balanced.'}
            </p>
          )}
        </div>
      )}

      {/* Outlier summary */}
      {Object.keys(outliers).length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
          <h2 className="font-semibold text-gray-900 dark:text-white mb-3">Outlier Summary (IQR Method)</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  {['Column', 'IQR Outliers', 'Z-Score Outliers', 'Q1', 'Q3', 'IQR'].map(h => (
                    <th key={h} className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {Object.entries(outliers).filter(([, v]) => v.iqr_outliers > 0).map(([col, v]) => (
                  <tr key={col} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="px-3 py-2 font-mono text-xs text-gray-700 dark:text-gray-300">{col}</td>
                    <td className="px-3 py-2 text-center font-medium text-orange-600 dark:text-orange-400">{v.iqr_outliers}</td>
                    <td className="px-3 py-2 text-center text-gray-600 dark:text-gray-400">{v.z_score_outliers}</td>
                    <td className="px-3 py-2 text-center text-gray-600 dark:text-gray-400">{v.q1?.toFixed(2)}</td>
                    <td className="px-3 py-2 text-center text-gray-600 dark:text-gray-400">{v.q3?.toFixed(2)}</td>
                    <td className="px-3 py-2 text-center text-gray-600 dark:text-gray-400">{v.iqr?.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Numerical column stats */}
      {numericalCols.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
          <h2 className="font-semibold text-gray-900 dark:text-white mb-3">Numerical Column Statistics</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  {['Column', 'Min', 'Max', 'Mean', 'Median', 'Std', 'Missing'].map(h => (
                    <th key={h} className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {numericalCols.map(([col, s]) => (
                  <tr key={col} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="px-3 py-2 font-mono text-xs text-gray-700 dark:text-gray-300">{col}</td>
                    {[s.min, s.max, s.mean, s.median, s.std].map((v, i) => (
                      <td key={i} className="px-3 py-2 text-gray-600 dark:text-gray-400 text-xs font-mono">
                        {v !== null && v !== undefined ? (typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(2) : v) : '—'}
                      </td>
                    ))}
                    <td className="px-3 py-2 text-center">
                      <span className={`text-xs font-medium ${s.missing > 0 ? 'text-red-600' : 'text-green-600'}`}>{s.missing}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Categorical columns */}
      {categoricalCols.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
          <h2 className="font-semibold text-gray-900 dark:text-white mb-3">Categorical Column Stats</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {categoricalCols.map(([col, s]) => (
              <div key={col} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2 font-mono">{col}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                  {s.unique} unique values | {s.missing} missing
                </p>
                <div className="space-y-1">
                  {Object.entries(s.value_counts || {}).slice(0, 5).map(([val, cnt]) => (
                    <div key={val} className="flex justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400 truncate max-w-28">{val}</span>
                      <span className="text-xs font-medium text-indigo-600 dark:text-indigo-400">{cnt}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DataQuality;
