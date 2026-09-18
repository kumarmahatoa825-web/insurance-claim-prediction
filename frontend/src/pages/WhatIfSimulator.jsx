import React, { useState, useEffect } from 'react';
import { predict, getDataMode } from '../services/api';
import { DEFAULT_VEHICLE_RECORD, VEHICLE_SCHEMA } from '../config/vehicleSchema';
import { getRiskColor, getRiskBgColor } from '../utils/riskUtils';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import Disclaimer from '../components/ui/Disclaimer';
import RiskBadge from '../components/ui/RiskBadge';
import toast from 'react-hot-toast';
import {
  RefreshCw, ArrowRight, TrendingDown, TrendingUp, ShieldAlert,
  Car, Sliders, CheckCircle2, AlertCircle, Info, Sparkles
} from 'lucide-react';

const INITIAL_SCENARIO = {
  ...DEFAULT_VEHICLE_RECORD,
  store_prediction: false,
};

export default function WhatIfSimulator() {
  const [dataMode, setDataMode] = useState('real');
  const [activeThreshold, setActiveThreshold] = useState(0.080);
  const [original, setOriginal] = useState({ ...INITIAL_SCENARIO });
  const [modified, setModified] = useState({ ...INITIAL_SCENARIO });
  const [loading, setLoading] = useState(false);
  const [origResult, setOrigResult] = useState(null);
  const [modResult, setModResult] = useState(null);

  useEffect(() => {
    getDataMode()
      .then(res => {
        if (res.data?.data_mode) setDataMode(res.data.data_mode);
        if (res.data?.default_threshold != null) setActiveThreshold(res.data.default_threshold);
      })
      .catch(() => {});
    runSimulation(INITIAL_SCENARIO, INITIAL_SCENARIO);
  }, []);

  const runSimulation = async (origInput, modInput) => {
    setLoading(true);
    try {
      const cleanOrig = { ...origInput, store_prediction: false };
      delete cleanOrig.threshold;
      const cleanMod = { ...modInput, store_prediction: false };
      delete cleanMod.threshold;
      const [resOrig, resMod] = await Promise.all([
        predict(cleanOrig),
        predict(cleanMod),
      ]);
      setOrigResult(resOrig.data);
      setModResult(resMod.data);
    } catch (err) {
      toast.error('Simulation calculation failed. Please check inputs.');
    } finally {
      setLoading(false);
    }
  };

  const handleModifiedChange = (field, value) => {
    const updated = { ...modified, [field]: value };
    setModified(updated);
  };

  const handleReset = () => {
    setModified({ ...original });
    runSimulation(original, original);
    toast.success('Simulation reset to original scenario.');
  };

  // Detect changed fields between original and modified
  const changedFields = Object.keys(modified).filter(k => {
    if (k === 'store_prediction' || k === 'threshold') return false;
    return modified[k] !== original[k];
  });

  const probDiff = modResult && origResult
    ? ((modResult.claim_probability - origResult.claim_probability) * 100).toFixed(2)
    : 0;
  const scoreDiff = modResult && origResult ? modResult.risk_score - origResult.risk_score : 0;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">What-If Risk Simulator</h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300">
              Vehicle Insurance &bull; Active Threshold: {Number(activeThreshold).toFixed(3)}
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Simulate parameter changes on policyholder, vehicle safety, and performance attributes to evaluate risk sensitivity.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            <RefreshCw className="w-4 h-4" /> Reset
          </button>
          <button
            onClick={() => runSimulation(original, modified)}
            disabled={loading}
            className="flex items-center gap-1.5 px-4 py-2 text-sm font-semibold bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg shadow-sm transition disabled:opacity-50"
          >
            {loading ? <LoadingSpinner size="small" /> : <Sparkles className="w-4 h-4" />} Run Simulation
          </button>
        </div>
      </div>

      {/* Comparison Results Card */}
      {origResult && modResult && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Original Card */}
          <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
            <span className="text-xs font-bold text-gray-400 tracking-wider uppercase">Baseline Scenario</span>
            <div className="mt-3 flex items-baseline justify-between">
              <span className="text-3xl font-extrabold text-gray-900 dark:text-white">
                {(origResult.claim_probability * 100).toFixed(1)}%
              </span>
              <RiskBadge category={origResult.risk_category} />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Risk Score: <strong className="text-gray-700 dark:text-gray-300">{origResult.risk_score}/100</strong> (Project Scale)
            </p>
            <p className="text-xs text-gray-500 mt-0.5">Status: {origResult.prediction}</p>
          </div>

          {/* Delta / Impact Card */}
          <div className={`p-5 rounded-xl border shadow-sm flex flex-col justify-center text-center ${
            probDiff < 0
              ? 'bg-green-50/70 border-green-200 dark:bg-green-950/20 dark:border-green-800'
              : probDiff > 0
              ? 'bg-amber-50/70 border-amber-200 dark:bg-amber-950/20 dark:border-amber-800'
              : 'bg-gray-50 border-gray-200 dark:bg-gray-800/50 dark:border-gray-700'
          }`}>
            <span className="text-xs font-bold text-gray-500 uppercase">Simulated Impact</span>
            <div className="flex items-center justify-center gap-2 mt-2">
              {probDiff < 0 ? (
                <TrendingDown className="w-6 h-6 text-green-600 dark:text-green-400" />
              ) : probDiff > 0 ? (
                <TrendingUp className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              ) : null}
              <span className={`text-3xl font-black ${
                probDiff < 0
                  ? 'text-green-600 dark:text-green-400'
                  : probDiff > 0
                  ? 'text-amber-600 dark:text-amber-400'
                  : 'text-gray-700 dark:text-gray-300'
              }`}>
                {probDiff > 0 ? `+${probDiff}%` : `${probDiff}%`}
              </span>
            </div>
            <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mt-1">
              {probDiff < 0
                ? `Risk probability decreased by ${Math.abs(probDiff)} percentage points`
                : probDiff > 0
                ? `Risk probability increased by ${probDiff} percentage points`
                : 'No net probability change'}
            </p>
            {changedFields.length > 0 && (
              <span className="mt-2 inline-block text-[11px] font-semibold text-indigo-600 dark:text-indigo-400">
                {changedFields.length} attribute{changedFields.length > 1 ? 's' : ''} modified
              </span>
            )}
          </div>

          {/* Modified Card */}
          <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-indigo-200 dark:border-indigo-800/60 shadow-sm ring-1 ring-indigo-500/20">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400 tracking-wider uppercase">Modified Scenario</span>
              <span className="text-[10px] bg-indigo-50 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 px-2 py-0.5 rounded font-mono font-medium">Active</span>
            </div>
            <div className="mt-3 flex items-baseline justify-between">
              <span className="text-3xl font-extrabold text-indigo-600 dark:text-indigo-400">
                {(modResult.claim_probability * 100).toFixed(1)}%
              </span>
              <RiskBadge category={modResult.risk_category} />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Risk Score: <strong className="text-gray-700 dark:text-gray-300">{modResult.risk_score}/100</strong> (Project Scale)
            </p>
            <p className="text-xs text-gray-500 mt-0.5">Status: {modResult.prediction}</p>
          </div>
        </div>
      )}

      {/* Changed Fields Summary */}
      {changedFields.length > 0 && (
        <div className="bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2 text-amber-800 dark:text-amber-300 font-semibold text-sm">
            <AlertCircle className="w-4 h-4 text-amber-600" />
            <span>Modified Parameters from Baseline:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {changedFields.map(f => (
              <span key={f} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-white dark:bg-gray-800 border border-amber-300 dark:border-amber-700 text-amber-900 dark:text-amber-200">
                <span className="font-semibold">{f.replace(/_/g, ' ')}:</span>
                <span className="text-gray-500 line-through">{String(original[f])}</span>
                <ArrowRight className="w-3 h-3 text-amber-600" />
                <span className="font-bold text-indigo-600 dark:text-indigo-400">{String(modified[f])}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Simulation Controls Form */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm">
        <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Sliders className="w-5 h-5 text-indigo-600" /> Adjust Parameters to Simulate Risk
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {/* Policy Tenure */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Policy Tenure (normalized): <span className="text-indigo-600 font-bold">{modified.policy_tenure}</span>
            </label>
            <input
              type="range"
              min={VEHICLE_SCHEMA.numericBounds.policy_tenure.min}
              max={VEHICLE_SCHEMA.numericBounds.policy_tenure.max}
              step={VEHICLE_SCHEMA.numericBounds.policy_tenure.step}
              value={modified.policy_tenure}
              onChange={e => handleModifiedChange('policy_tenure', parseFloat(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <span className="text-[10px] text-gray-400">Baseline: {original.policy_tenure}</span>
          </div>

          {/* Age of Policyholder */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Age of Policyholder: <span className="text-indigo-600 font-bold">{modified.age_of_policyholder}</span>
            </label>
            <input
              type="range"
              min={VEHICLE_SCHEMA.numericBounds.age_of_policyholder.min}
              max={VEHICLE_SCHEMA.numericBounds.age_of_policyholder.max}
              step={VEHICLE_SCHEMA.numericBounds.age_of_policyholder.step}
              value={modified.age_of_policyholder}
              onChange={e => handleModifiedChange('age_of_policyholder', parseFloat(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <span className="text-[10px] text-gray-400">Baseline: {original.age_of_policyholder}</span>
          </div>

          {/* Age of Car */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Age of Car: <span className="text-indigo-600 font-bold">{modified.age_of_car}</span>
            </label>
            <input
              type="range"
              min={VEHICLE_SCHEMA.numericBounds.age_of_car.min}
              max={VEHICLE_SCHEMA.numericBounds.age_of_car.max}
              step={VEHICLE_SCHEMA.numericBounds.age_of_car.step}
              value={modified.age_of_car}
              onChange={e => handleModifiedChange('age_of_car', parseFloat(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <span className="text-[10px] text-gray-400">Baseline: {original.age_of_car}</span>
          </div>

          {/* Airbags */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              Airbags: <span className="text-indigo-600 font-bold">{modified.airbags}</span>
            </label>
            <input
              type="range"
              min={VEHICLE_SCHEMA.numericBounds.airbags.min}
              max={VEHICLE_SCHEMA.numericBounds.airbags.max}
              step={VEHICLE_SCHEMA.numericBounds.airbags.step}
              value={modified.airbags}
              onChange={e => handleModifiedChange('airbags', parseInt(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <span className="text-[10px] text-gray-400">Baseline: {original.airbags}</span>
          </div>

          {/* NCAP Safety Rating */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
              NCAP Safety Rating (0-5 stars)
            </label>
            <select
              value={modified.ncap_rating}
              onChange={e => handleModifiedChange('ncap_rating', parseInt(e.target.value))}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2"
            >
              {[0, 1, 2, 3, 4, 5].map(r => (
                <option key={r} value={r}>{r} Star{r !== 1 ? 's' : ''}</option>
              ))}
            </select>
          </div>

          {/* Fuel Type */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Fuel Type</label>
            <select
              value={modified.fuel_type}
              onChange={e => handleModifiedChange('fuel_type', e.target.value)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2"
            >
              {VEHICLE_SCHEMA.fuelTypes.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>

          {/* Vehicle Segment */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Vehicle Segment</label>
            <select
              value={modified.segment}
              onChange={e => handleModifiedChange('segment', e.target.value)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2"
            >
              {VEHICLE_SCHEMA.segments.map(s => (
                <option key={s} value={s}>Segment {s}</option>
              ))}
            </select>
          </div>

          {/* Transmission Type */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Transmission</label>
            <select
              value={modified.transmission_type}
              onChange={e => handleModifiedChange('transmission_type', e.target.value)}
              className="w-full text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2"
            >
              {VEHICLE_SCHEMA.transmissionTypes.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Safety Features Toggles */}
        <div className="mt-6 pt-5 border-t border-gray-200 dark:border-gray-700">
          <label className="block text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-wider mb-3">
            Safety Technology Interventions
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {[
              { field: 'is_esc', label: 'Electronic Stability (ESC)' },
              { field: 'is_tpms', label: 'Tire Pressure (TPMS)' },
              { field: 'is_brake_assist', label: 'Brake Assist' },
              { field: 'is_parking_sensors', label: 'Parking Sensors' },
              { field: 'is_parking_camera', label: 'Parking Camera' },
            ].map(item => {
              const isYes = modified[item.field] === 'Yes';
              return (
                <button
                  key={item.field}
                  type="button"
                  onClick={() => handleModifiedChange(item.field, isYes ? 'No' : 'Yes')}
                  className={`flex items-center justify-between p-3 rounded-lg border text-left text-xs font-medium transition ${
                    isYes
                      ? 'bg-indigo-50 border-indigo-400 text-indigo-900 dark:bg-indigo-950/40 dark:border-indigo-700 dark:text-indigo-200'
                      : 'bg-gray-50 border-gray-200 text-gray-600 dark:bg-gray-800/40 dark:border-gray-700 dark:text-gray-400'
                  }`}
                >
                  <span>{item.label}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    isYes ? 'bg-indigo-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}>
                    {modified[item.field]}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Analytical Caution Disclaimer (Requirement 245) */}
      <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4 flex items-start gap-3">
        <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 shrink-0 mt-0.5" />
        <div className="text-xs text-blue-800 dark:text-blue-300 space-y-1">
          <p className="font-semibold">Causal Inference Notice (Requirement 245):</p>
          <p>
            What-If risk simulations are model-based predictive estimations under changed feature values. They reflect correlational patterns learned from historical vehicle insurance claims and do not guarantee causal real-world outcomes.
          </p>
        </div>
      </div>

      <Disclaimer />
    </div>
  );
}
