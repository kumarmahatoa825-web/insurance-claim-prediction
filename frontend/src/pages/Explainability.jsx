import React, { useEffect, useState } from 'react';
import { getModelInfo, explain } from '../services/api';
import { useApi } from '../hooks/useApi';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ErrorMessage from '../components/ui/ErrorMessage';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine
} from 'recharts';
import { Info, Brain, Sparkles, Sliders, ShieldCheck, AlertCircle } from 'lucide-react';
import RiskBadge from '../components/ui/RiskBadge';

import { DEFAULT_VEHICLE_RECORD } from '../config/vehicleSchema';

export default function Explainability() {
  const { data: modelInfo, loading, error, execute: fetchInfo } = useApi(getModelInfo);
  const [localInput, setLocalInput] = useState({ ...DEFAULT_VEHICLE_RECORD });
  const [shapResult, setShapResult] = useState(null);
  const [shapLoading, setShapLoading] = useState(false);
  const [shapError, setShapError] = useState(null);

  useEffect(() => {
    fetchInfo();
    runInitialExplain();
  }, [fetchInfo]);

  const runInitialExplain = async () => {
    try {
      const payload = { ...DEFAULT_VEHICLE_RECORD, store_prediction: false };
      delete payload.threshold;
      const res = await explain(payload);
      setShapResult(res.data);
    } catch {
      // initial load silent fallback
    }
  };

  const handleInputChange = (field, value) => {
    setLocalInput(prev => ({ ...prev, [field]: value }));
  };

  const runExplain = async () => {
    setShapLoading(true);
    setShapError(null);
    try {
      const payload = { ...localInput, store_prediction: false };
      delete payload.threshold;
      const res = await explain(payload);
      setShapResult(res.data);
    } catch (err) {
      setShapError(err?.response?.data?.detail || 'SHAP explanation computation failed.');
    } finally {
      setShapLoading(false);
    }
  };

  if (loading) return <LoadingSpinner size="lg" className="mt-20" />;
  if (error) return <ErrorMessage message={error} onRetry={fetchInfo} />;

  const featureImportances = modelInfo?.feature_importance || modelInfo?.feature_importances || [];
  const topGlobalFeatures = featureImportances.slice(0, 15);

  // Local SHAP factors
  const localFactors = shapResult?.explanation?.top_risk_factors || shapResult?.prediction?.top_risk_factors || [];
  const localProtective = shapResult?.explanation?.protective_factors || shapResult?.prediction?.protective_factors || [];

  const allShapFactors = [
    ...localFactors.map(f => ({ ...f, fill: '#ef4444' })),
    ...localProtective.map(f => ({
      ...f,
      fill: '#22c55e',
      contribution: -Math.abs(f.contribution || 0),
    })),
  ]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 14);

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Page Title */}
      <div>
        <div className="flex items-center gap-2.5">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Explainable AI (SHAP)</h1>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300">
            TreeExplainer Engine
          </span>
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Rigorous game-theoretic feature attributions for vehicle insurance risk, globally across the population and locally per policyholder.
        </p>
      </div>

      {/* SHAP Info Banner with Causality Disclaimer */}
      <div className="bg-indigo-50 dark:bg-indigo-950/20 border border-indigo-200 dark:border-indigo-800/80 rounded-xl p-5 flex gap-4">
        <Brain className="w-6 h-6 text-indigo-600 dark:text-indigo-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h3 className="text-sm font-bold text-indigo-900 dark:text-indigo-300">
            Shapley Additive exPlanations (SHAP) Methodology
          </h3>
          <p className="text-xs text-indigo-800 dark:text-indigo-300 leading-relaxed">
            SHAP distributes fair credit for the model's claim probability among input vehicle and policyholder features based on cooperative game theory. Positive SHAP values (red) increase the estimated risk, while negative values (green) act as protective factors.
          </p>
          <div className="pt-2 flex items-center gap-1.5 text-[11px] font-semibold text-indigo-700 dark:text-indigo-400">
            <AlertCircle className="w-3.5 h-3.5" />
            <span>Important: SHAP values quantify model attribution and statistical associations in training data; they are not causal proof of real-world risk.</span>
          </div>
        </div>
      </div>

      {/* Global Feature Importance */}
      {topGlobalFeatures.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h2 className="text-base font-bold text-gray-900 dark:text-white">Global Model Feature Importance (Mean Gini / MDI)</h2>
              <p className="text-xs text-gray-500">
                Mean Decrease in Impurity (Gini importance) measuring overall feature split contributions across the ensemble. Global importance reflects aggregate model reliance across the entire 58,592 policyholder dataset, whereas local SHAP waterfall values below quantify directional risk increases or decreases for an individual policyholder.
              </p>
            </div>
            <span className="text-xs font-mono bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-2.5 py-1 rounded">
              {modelInfo?.model_name || 'Champion Random Forest'}
            </span>
          </div>

          <div className="h-96 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={topGlobalFeatures}
                layout="vertical"
                margin={{ left: 140, right: 30, top: 10, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  width={140}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <Tooltip
                  formatter={(v) => [Number(v).toFixed(4), 'Importance']}
                  labelFormatter={(l) => `Feature: ${l}`}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {topGlobalFeatures.map((_, i) => (
                    <Cell
                      key={i}
                      fill={`hsl(${230 - i * 6}, 75%, ${50 + i * 2}%)`}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Local Prediction Explanation Demo */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm">
        <h2 className="text-base font-bold text-gray-900 dark:text-white mb-1 flex items-center gap-2">
          <Sliders className="w-5 h-5 text-indigo-600" /> Local Policyholder Explanation
        </h2>
        <p className="text-xs text-gray-500 mb-5">
          Select or customize a policyholder vehicle profile to compute an individual SHAP waterfall explanation.
        </p>

        {/* Real Vehicle Form Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Policy Tenure (normalized)
            </label>
            <input
              type="number"
              step="0.01"
              value={localInput.policy_tenure}
              onChange={e => handleInputChange('policy_tenure', parseFloat(e.target.value) || 0)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-white"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Age of Policyholder
            </label>
            <input
              type="number"
              step="0.01"
              value={localInput.age_of_policyholder}
              onChange={e => handleInputChange('age_of_policyholder', parseFloat(e.target.value) || 0)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-white"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Age of Car (normalized)
            </label>
            <input
              type="number"
              step="0.01"
              value={localInput.age_of_car}
              onChange={e => handleInputChange('age_of_car', parseFloat(e.target.value) || 0)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-white"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Area Cluster
            </label>
            <select
              value={localInput.area_cluster}
              onChange={e => handleInputChange('area_cluster', e.target.value)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-white"
            >
              {AREA_CLUSTERS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Fuel Type
            </label>
            <select
              value={localInput.fuel_type}
              onChange={e => handleInputChange('fuel_type', e.target.value)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-white"
            >
              {['Petrol', 'Diesel', 'CNG'].map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Vehicle Segment
            </label>
            <select
              value={localInput.segment}
              onChange={e => handleInputChange('segment', e.target.value)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-white"
            >
              {['A', 'B1', 'B2', 'C1', 'C2', 'Utility'].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Airbags (Count)
            </label>
            <input
              type="number"
              min="1"
              max="6"
              value={localInput.airbags}
              onChange={e => handleInputChange('airbags', parseInt(e.target.value) || 2)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-white"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              NCAP Safety Rating (0-5)
            </label>
            <select
              value={localInput.ncap_rating}
              onChange={e => handleInputChange('ncap_rating', parseInt(e.target.value) || 0)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-white"
            >
              {[0, 1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n} Stars</option>)}
            </select>
          </div>
        </div>

        <button
          onClick={runExplain}
          disabled={shapLoading}
          className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-medium rounded-lg transition-colors text-sm shadow-sm"
        >
          {shapLoading ? <LoadingSpinner size="small" /> : <Sparkles className="w-4 h-4" />} Explain This Policy
        </button>

        {shapError && <p className="text-sm text-red-600 mt-3">{shapError}</p>}

        {/* Prediction Summary & Waterfall */}
        {shapResult && (
          <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700 space-y-6">
            {shapResult.available === false ? (
              <div className="p-4 rounded-xl border border-amber-200 bg-amber-50 dark:bg-amber-950/20 text-amber-800 dark:text-amber-300 flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold">⚠ SHAP explanation unavailable</p>
                  <p className="text-xs">{shapResult.reason || 'Local SHAP attribution is uninitialized or unavailable for this model.'}</p>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="flex flex-wrap items-center gap-6 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-xl border border-gray-100 dark:border-gray-800">
                  <div>
                    <span className="text-xs text-gray-500">Predicted Claim Probability</span>
                    <p className="text-2xl font-black text-indigo-600 dark:text-indigo-400">
                      {((shapResult.prediction?.claim_probability || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="h-10 w-px bg-gray-200 dark:bg-gray-700" />
                  <div>
                    <span className="text-xs text-gray-500">AI Risk Score (Project Scale)</span>
                    <p className="text-2xl font-black text-gray-900 dark:text-white">
                      {shapResult.prediction?.risk_score || 0}/100
                    </p>
                  </div>
                  <div className="h-10 w-px bg-gray-200 dark:bg-gray-700" />
                  <div>
                    <span className="text-xs text-gray-500">Risk Classification</span>
                    <div className="mt-1">
                      <RiskBadge category={shapResult.prediction?.risk_category || 'Low'} />
                    </div>
                  </div>
                  <div className="h-10 w-px bg-gray-200 dark:bg-gray-700" />
                  <div>
                    <span className="text-xs text-gray-500">Decision Threshold</span>
                    <p className="text-base font-mono font-bold text-gray-700 dark:text-gray-300 mt-1">
                      {shapResult.prediction?.threshold_used != null ? Number(shapResult.prediction.threshold_used).toFixed(2) : '—'}
                    </p>
                  </div>
                </div>

                {shapResult.attribution_note && (
                  <p className="text-xs text-gray-500 italic bg-gray-50 dark:bg-gray-800/60 p-2.5 rounded-lg border border-gray-100 dark:border-gray-700">
                    ℹ️ {shapResult.attribution_note}
                  </p>
                )}
              </div>
            )}

            {/* SHAP Waterfall Chart */}
            {allShapFactors.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-bold text-gray-900 dark:text-white">
                    Feature Contributions for this Policy (SHAP Values)
                  </h3>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="flex items-center gap-1.5 text-red-600 font-semibold">
                      <span className="w-3 h-3 rounded bg-red-500" /> Increases Claim Probability
                    </span>
                    <span className="flex items-center gap-1.5 text-green-600 font-semibold">
                      <span className="w-3 h-3 rounded bg-green-500" /> Protective (Decreases Risk)
                    </span>
                  </div>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={allShapFactors}
                      layout="vertical"
                      margin={{ left: 160, right: 40, top: 10, bottom: 10 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis type="number" tick={{ fontSize: 11 }} />
                      <YAxis
                        dataKey="feature"
                        type="category"
                        width={160}
                        tick={{ fontSize: 11, fill: '#64748b' }}
                      />
                      <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                      <ReferenceLine x={0} stroke="#6b7280" strokeWidth={1.5} />
                      <Bar dataKey="contribution" radius={3}>
                        {allShapFactors.map((f, i) => (
                          <Cell key={i} fill={f.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
