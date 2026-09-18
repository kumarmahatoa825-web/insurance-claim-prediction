import React, { useEffect } from 'react';
import { getAnalytics } from '../services/api';
import { useApi } from '../hooks/useApi';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ErrorMessage from '../components/ui/ErrorMessage';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area
} from 'recharts';

const COLORS = ['#6366f1', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6'];
const RISK_COLORS = { Low: '#22c55e', Moderate: '#eab308', High: '#f97316', Critical: '#ef4444' };

function ChartCard({ title, subtitle, children, tall }) {
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-5 ${tall ? 'lg:col-span-2' : ''}`}>
      <h3 className="font-semibold text-gray-900 dark:text-white text-sm">{title}</h3>
      {subtitle && <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 mb-3">{subtitle}</p>}
      <div className="mt-3">
        {children}
      </div>
    </div>
  );
}

const Analytics = () => {
  const { data, loading, error, execute: fetchData } = useApi(getAnalytics);

  useEffect(() => { fetchData(); }, [fetchData]);

  if (loading) return <LoadingSpinner size="lg" className="mt-20" />;
  if (error) return <ErrorMessage message={error} onRetry={fetchData} />;
  if (!data) return null;

  const claimDistData = data.claim_distribution
    ? Object.entries(data.claim_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const riskDistData = data.risk_category_distribution
    ? Object.entries(data.risk_category_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const ageData = data.claims_by_age_group
    ? Object.entries(data.claims_by_age_group).map(([group, claims]) => ({ group, claims }))
    : [];

  const policyTypeData = data.claims_by_policy_type
    ? Object.entries(data.claims_by_policy_type).map(([type, claims]) => ({ type, claims }))
    : [];

  const locationData = data.claims_by_location
    ? Object.entries(data.claims_by_location).map(([location, claims]) => ({ location, claims }))
    : [];

  const occupationData = data.claims_by_occupation
    ? Object.entries(data.claims_by_occupation).map(([occupation, claims]) => ({ occupation, claims }))
    : [];

  const incomeData = data.claims_by_income_range
    ? Object.entries(data.claims_by_income_range).map(([range, claims]) => ({ range, claims }))
    : [];

  const featureImportance = data.feature_importance || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Dataset-level analytics computed from {data.total_analyzed?.toLocaleString()} records.
        </p>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Policies', value: data.total_analyzed?.toLocaleString() || '—' },
          { label: 'Actual Claim Rate', value: `${((data.actual_claim_rate || 0) * 100).toFixed(1)}%` },
          { label: 'Best Model', value: data.best_model || '—' },
          { label: 'Best F1 Score', value: (data.best_f1 || 0).toFixed(3) },
        ].map(s => (
          <div key={s.label} className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow text-center">
            <p className="text-xl font-bold text-gray-900 dark:text-white">{s.value}</p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {/* Claim Distribution */}
        <ChartCard title="Claim vs No Claim" subtitle="Target class distribution in dataset">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={claimDistData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {claimDistData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Pie>
              <Tooltip formatter={(v) => v.toLocaleString()} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Risk Distribution */}
        <ChartCard title="Risk Category Distribution" subtitle="Based on AI Risk Score thresholds">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={riskDistData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                {riskDistData.map((entry) => <Cell key={entry.name} fill={RISK_COLORS[entry.name] || '#6366f1'} />)}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Probability Histogram */}
        {data.probability_histogram?.length > 0 && (
          <ChartCard title="Claim Probability Distribution" subtitle="From sampled predictions">
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={data.probability_histogram}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Area type="monotone" dataKey="count" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {/* Claims by Age Group */}
        <ChartCard title="Claims by Age Group">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={ageData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="group" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="claims" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Claims by Policy Type */}
        <ChartCard title="Claims by Policy Type">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={policyTypeData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="type" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="claims" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Claims by Location */}
        <ChartCard title="Claims by Location">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={locationData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="location" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="claims" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Claims by Occupation */}
        <ChartCard title="Claims by Occupation">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={occupationData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="occupation" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="claims" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Claims by Income */}
        <ChartCard title="Claims by Income Range" subtitle="INR income brackets">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={incomeData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="range" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="claims" fill="#14b8a6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Feature Importance */}
        {featureImportance.length > 0 && (
          <ChartCard title="Top Feature Importance" subtitle="Model-derived feature weights" tall>
            <ResponsiveContainer width="100%" height={featureImportance.length * 28 + 40}>
              <BarChart data={featureImportance.slice(0, 12)} layout="vertical" margin={{ left: 130 }}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="feature" type="category" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => v.toFixed(4)} />
                <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
      </div>
    </div>
  );
};

export default Analytics;
