import React, { useEffect } from 'react';
import { getAnalytics, getModelInfo } from '../services/api';
import { useApi } from '../hooks/useApi';
import StatCard from '../components/ui/StatCard';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ErrorMessage from '../components/ui/ErrorMessage';
import { formatMetric } from '../utils/formatters';
import { Activity, Users, AlertCircle, Percent, Award, Target } from 'lucide-react';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, Legend, ResponsiveContainer, AreaChart, Area
} from 'recharts';

const RISK_COLORS = { Low: '#22c55e', Moderate: '#eab308', High: '#f97316', Critical: '#ef4444' };
const CLAIM_COLORS = ['#ef4444', '#22c55e'];

const Dashboard = () => {
  const { data: analytics, loading: analyticsLoading, error: analyticsError, execute: fetchAnalytics } = useApi(getAnalytics);
  const { data: modelInfo, loading: modelLoading, error: modelError, execute: fetchModelInfo } = useApi(getModelInfo);

  useEffect(() => {
    fetchAnalytics();
    fetchModelInfo();
  }, [fetchAnalytics, fetchModelInfo]);

  if (analyticsLoading || modelLoading) return <LoadingSpinner size="lg" className="mt-20" />;
  if (analyticsError) return <ErrorMessage message="Backend not reachable. Start the FastAPI server and retrain the model." onRetry={() => { fetchAnalytics(); fetchModelInfo(); }} />;
  if (!analytics) return null;

  // Claim distribution as recharts-friendly array
  const claimDistData = analytics.claim_distribution
    ? Object.entries(analytics.claim_distribution).map(([name, value]) => ({ name, value }))
    : [];

  // Risk category distribution
  const riskDistData = analytics.risk_category_distribution
    ? Object.entries(analytics.risk_category_distribution).map(([category, count]) => ({ category, count }))
    : [];

  // Probability histogram
  const histData = analytics.probability_histogram || [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <StatCard title="Total Policies Analyzed" value={(analytics.total_analyzed || 0).toLocaleString()} icon={Users} />
        <StatCard title="High Risk Predictions" value={(analytics.high_risk_count || 0).toLocaleString()} icon={AlertCircle} color="red" />
        <StatCard title="Avg Claim Probability" value={`${((analytics.avg_claim_probability || 0) * 100).toFixed(1)}%`} icon={Percent} />
        <StatCard title="Actual Claim Rate" value={`${((analytics.actual_claim_rate || 0) * 100).toFixed(1)}%`} icon={Activity} />
        <StatCard title="Best Model" value={modelInfo?.model_name || analytics.best_model || 'Unknown'} icon={Award} color="indigo" />
        <StatCard title="Model F1 Score" value={(modelInfo?.test_metrics?.f1 || analytics.best_f1 || 0).toFixed(3)} icon={Target} color="green" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* Claim Distribution Pie */}
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Claim Distribution</h3>
          {claimDistData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={claimDistData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {claimDistData.map((_, i) => <Cell key={i} fill={CLAIM_COLORS[i % 2]} />)}
                </Pie>
                <RechartsTooltip formatter={(v) => v.toLocaleString()} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : <p className="text-sm text-gray-400 text-center py-10">No data</p>}
        </div>

        {/* Risk Category Bar */}
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Risk Category Counts</h3>
          {riskDistData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={riskDistData}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <RechartsTooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {riskDistData.map((entry) => (
                    <Cell key={entry.category} fill={RISK_COLORS[entry.category] || '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <p className="text-sm text-gray-400 text-center py-10">No data</p>}
        </div>

        {/* Probability Histogram */}
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Probability Histogram</h3>
            <span className="text-[10px] uppercase font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/40 px-2 py-0.5 rounded border border-indigo-200 dark:border-indigo-800">
              Sampled Estimation
            </span>
          </div>
          {histData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={histData}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <RechartsTooltip />
                <Area type="monotone" dataKey="count" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <p className="text-sm text-gray-400 text-center py-10">Run predictions to see histogram</p>}
        </div>
      </div>

      {/* Model Overview card */}
      {modelInfo && (
        <div className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg p-5">
          <h3 className="text-sm font-semibold text-indigo-800 dark:text-indigo-300 mb-3">Active Model</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            {[
              { label: 'Model', value: modelInfo.model_name || '—' },
              { label: 'Version', value: modelInfo.version || '—' },
              { label: 'ROC-AUC', value: formatMetric(modelInfo.test_metrics?.roc_auc) },
              { label: 'Optimal Threshold', value: (modelInfo.default_threshold ?? modelInfo.optimal_threshold) != null ? (modelInfo.default_threshold ?? modelInfo.optimal_threshold).toFixed(2) : '—' },
            ].map(s => (
              <div key={s.label}>
                <p className="text-base font-bold text-indigo-800 dark:text-indigo-200">{s.value}</p>
                <p className="text-xs text-indigo-600 dark:text-indigo-400 mt-0.5">{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
        <p className="text-xs text-amber-800 dark:text-amber-300">
          ⚠️ <strong>Disclaimer:</strong> InsureAI is an educational AI/ML project. Predictions are probabilistic estimates
          and should not be used as the sole basis for real insurance decisions. Analytical recommendations shown are
          not automatic insurance determinations.
        </p>
      </div>
    </div>
  );
};

export default Dashboard;
