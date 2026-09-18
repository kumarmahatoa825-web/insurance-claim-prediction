import React, { useEffect, useState } from 'react';
import { getModelComparison } from '../services/api';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ErrorMessage from '../components/ui/ErrorMessage';
import { useApi } from '../hooks/useApi';
import { Star, Clock } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, ReferenceLine
} from 'recharts';

const COLORS = ['#6366f1', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#0ea5e9'];
const METRICS = ['pr_auc', 'f1', 'roc_auc', 'brier_score', 'recall', 'precision'];
const METRIC_LABELS = {
  pr_auc: 'PR-AUC',
  f1: 'F1 Score',
  roc_auc: 'ROC-AUC',
  brier_score: 'Brier Score',
  recall: 'Recall',
  precision: 'Precision',
};

function MetricCell({ metric, value, isBest }) {
  let formatted = '—';
  if (value != null && Number.isFinite(value)) {
    if (metric === 'brier_score') {
      formatted = value.toFixed(4);
    } else {
      formatted = `${(value * 100).toFixed(1)}%`;
    }
  }
  return (
    <td className={`px-4 py-3 text-sm text-center font-mono ${isBest ? 'text-green-600 dark:text-green-400 font-bold' : 'text-gray-700 dark:text-gray-300'}`}>
      {formatted}
    </td>
  );
}

const ModelComparison = () => {
  const { data, loading, error, execute: fetchData } = useApi(getModelComparison);
  const [sortMetric, setSortMetric] = useState('pr_auc');

  useEffect(() => { fetchData(); }, [fetchData]);

  if (loading) return <LoadingSpinner size="lg" className="mt-20" />;
  if (error) return <ErrorMessage message={error} onRetry={fetchData} />;
  if (!data) return null;

  const models = Object.entries(data).map(([name, info]) => ({ name, ...info }));
  const sortedModels = [...models].sort((a, b) => {
    const valA = a.cv_mean?.[sortMetric] ?? (sortMetric === 'brier_score' ? 999 : -999);
    const valB = b.cv_mean?.[sortMetric] ?? (sortMetric === 'brier_score' ? 999 : -999);
    return sortMetric === 'brier_score' ? valA - valB : valB - valA;
  });
  const bestName = sortedModels[0]?.name;

  // Bar chart data (exclude brier_score from percentage bar chart or scale it)
  const barMetrics = ['pr_auc', 'f1', 'roc_auc', 'recall', 'precision'];
  const barData = barMetrics.map(metric => {
    const row = { metric: METRIC_LABELS[metric] };
    models.forEach(m => { row[m.name] = parseFloat(((m.cv_mean?.[metric] || 0) * 100).toFixed(1)); });
    return row;
  });

  // ROC curve data (normalize thresholds to same length for recharts)
  const rocData = [];
  const rocModels = models.filter(m => m.roc_curve?.fpr);
  if (rocModels.length > 0) {
    const len = Math.min(...rocModels.map(m => m.roc_curve.fpr.length));
    const step = Math.max(1, Math.floor(len / 50));
    for (let i = 0; i < len; i += step) {
      const point = { fpr: parseFloat(rocModels[0].roc_curve.fpr[i].toFixed(3)) };
      rocModels.forEach(m => {
        point[m.name] = parseFloat(m.roc_curve.tpr[i]?.toFixed(3) ?? 0);
      });
      rocData.push(point);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Comparison</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Architectures evaluated using 5-fold Stratified Out-Of-Fold Cross-Validation on 46,873 development records with strict leak-free preprocessing and cross-fitted probability calibration.
        </p>
      </div>

      {/* Sort selector */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm text-gray-600 dark:text-gray-400">Sort by:</span>
        <div className="flex gap-2 flex-wrap">
          {METRICS.map(m => (
            <button key={m} onClick={() => setSortMetric(m)}
              className={`px-3 py-1 text-xs rounded-full font-medium transition-colors ${
                sortMetric === m ? 'bg-indigo-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200'
              }`}>
              {METRIC_LABELS[m]} {m === 'brier_score' ? '(Lowest)' : '(Highest)'}
            </button>
          ))}
        </div>
      </div>

      {/* Leaderboard table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <h2 className="font-semibold text-gray-900 dark:text-white">Model Leaderboard (5-Fold CV on Development Split)</h2>
          <span className="text-xs text-gray-500 dark:text-gray-400">Primary Selection Metric: <strong>PR-AUC &ge; 0.100</strong></span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Model Architecture</th>
                {METRICS.map(m => (
                  <th key={m} className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase cursor-pointer hover:text-indigo-600"
                    onClick={() => setSortMetric(m)}>
                    {METRIC_LABELS[m]} {sortMetric === m ? (m === 'brier_score' ? '▲' : '▼') : ''}
                  </th>
                ))}
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {sortedModels.map((m, i) => {
                const isChamp = m.is_champion === true;
                return (
                  <tr key={m.name} className={`hover:bg-gray-50 dark:hover:bg-gray-700/50 ${isChamp ? 'bg-indigo-50/60 dark:bg-indigo-950/20' : ''}`}>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {isChamp && <Star className="w-4 h-4 text-amber-500 fill-amber-400 shrink-0" />}
                        <span className={`text-sm font-medium ${isChamp ? 'text-indigo-900 dark:text-indigo-200 font-bold' : 'text-gray-900 dark:text-white'}`}>
                          {m.name}
                        </span>
                      </div>
                    </td>
                    {METRICS.map(metric => (
                      <MetricCell key={metric} metric={metric} value={m.cv_mean?.[metric]}
                        isBest={i === 0 && sortMetric === metric} />
                    ))}
                    <td className="px-4 py-3 text-center text-xs">
                      {isChamp ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300">
                          Active Champion
                        </span>
                      ) : (
                        <span className="text-gray-400 dark:text-gray-500 font-mono text-[11px]">Evaluated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bar chart comparison */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
        <h2 className="font-semibold text-gray-900 dark:text-white mb-4">Metric Comparison (CV Mean)</h2>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={barData} margin={{ left: 0, right: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
            <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Legend />
            {models.map((m, i) => (
              <Bar key={m.name} dataKey={m.name} fill={COLORS[i % COLORS.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ROC Curves */}
      {rocData.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
          <h2 className="font-semibold text-gray-900 dark:text-white mb-1">ROC Curves (Test Set)</h2>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
            Higher and more to the upper-left is better. Diagonal line = random classifier.
          </p>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="fpr" tickFormatter={v => v.toFixed(2)} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fontSize: 12 }} />
              <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip formatter={(v) => v?.toFixed(3)} />
              <Legend />
              <ReferenceLine x={0} y={0} stroke="gray" strokeDasharray="5 5" />
              {rocModels.map((m, i) => (
                <Line key={m.name} type="monotone" dataKey={m.name}
                  stroke={COLORS[i % COLORS.length]}
                  dot={false} strokeWidth={2}
                  name={`${m.name} (AUC=${m.roc_curve?.auc?.toFixed(3)})`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* SMOTE info */}
      <div className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-indigo-800 dark:text-indigo-300 mb-2">Class Imbalance Handling</h3>
        <p className="text-sm text-indigo-700 dark:text-indigo-400">
          All models were trained with <code className="bg-indigo-100 dark:bg-indigo-800 px-1 rounded">class_weight='balanced'</code> where supported.
          SMOTE oversampling was also evaluated on the best model — applied only to training folds, never to the test set.
        </p>
      </div>
    </div>
  );
};

export default ModelComparison;
