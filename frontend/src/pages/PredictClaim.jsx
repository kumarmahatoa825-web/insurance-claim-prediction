import React, { useState, useEffect } from 'react';
import { predict, getDataMode, getModelInfo } from '../services/api';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import Disclaimer from '../components/ui/Disclaimer';
import RiskBadge from '../components/ui/RiskBadge';
import toast from 'react-hot-toast';
import { AlertCircle, CheckCircle, Car, Shield, Gauge } from 'lucide-react';
import { VEHICLE_SCHEMA, DEFAULT_VEHICLE_RECORD } from '../config/vehicleSchema';

// ── Real dataset default values ──────────────────────────────────────────────
const REAL_DEFAULTS = {
  ...DEFAULT_VEHICLE_RECORD,
};

// ── Synthetic (old) dataset defaults ─────────────────────────────────────────
const SYNTHETIC_DEFAULTS = {
  policy_id: 'POL-2024-001',
  age: 35, gender: 'Male', income: 600000, occupation: 'Salaried',
  location: 'Urban', policy_type: 'Health', policy_tenure: 5,
  premium: 25000, coverage_amount: 1000000, vehicle_age: 2,
  num_dependents: 2, previous_claims: 1, payment_history: 0.85,
  renewal_count: 4, health_score: 0.75,
  store_prediction: true,
};

// ── Canonical vehicle form sections ──────────────────────────────────────────
const REAL_FORM_SECTIONS = [
  {
    title: 'Policy Info',
    icon: Shield,
    fields: [
      { key: 'policy_tenure', label: 'Policy Tenure (0–1, normalized)', type: 'range', min: 0, max: 1, step: 0.01 },
      { key: 'area_cluster', label: 'Area Cluster', type: 'select', opts: VEHICLE_SCHEMA.areaClusters },
      { key: 'population_density', label: 'Population Density', type: 'number', min: 100, max: 100000 },
    ],
  },
  {
    title: 'Policyholder Info',
    icon: Shield,
    fields: [
      { key: 'age_of_policyholder', label: 'Age of Policyholder (0–1, normalized)', type: 'range', min: 0, max: 1, step: 0.01 },
      { key: 'ncap_rating', label: 'NCAP Safety Rating (0–5)', type: 'select', opts: [0, 1, 2, 3, 4, 5] },
    ],
  },
  {
    title: 'Vehicle Info',
    icon: Car,
    fields: [
      { key: 'age_of_car', label: 'Age of Car (0–1, normalized)', type: 'range', min: 0, max: 1, step: 0.01 },
      { key: 'make', label: 'Manufacturer Code (Make 1-5)', type: 'select', opts: VEHICLE_SCHEMA.makes },
      { key: 'segment', label: 'Vehicle Segment', type: 'select', opts: VEHICLE_SCHEMA.segments },
      { key: 'model', label: 'Vehicle Model', type: 'select', opts: VEHICLE_SCHEMA.models },
      { key: 'fuel_type', label: 'Fuel Type', type: 'select', opts: VEHICLE_SCHEMA.fuelTypes },
      { key: 'transmission_type', label: 'Transmission', type: 'select', opts: VEHICLE_SCHEMA.transmissionTypes },
      { key: 'gear_box', label: 'Gear Box Speeds', type: 'number', min: 4, max: 8 },
      { key: 'cylinder', label: 'Cylinders', type: 'number', min: 2, max: 8 },
      { key: 'displacement', label: 'Engine Displacement (cc)', type: 'number', min: 500, max: 3000 },
      { key: 'airbags', label: 'Number of Airbags', type: 'number', min: 1, max: 10 },
      { key: 'turning_radius', label: 'Turning Radius (m)', type: 'number', min: 3, max: 10, step: 0.1 },
      { key: 'gross_weight', label: 'Gross Vehicle Weight (kg)', type: 'number', min: 500, max: 5000 },
      { key: 'length', label: 'Length (mm)', type: 'number', min: 2000, max: 6000 },
      { key: 'width', label: 'Width (mm)', type: 'number', min: 1000, max: 3000 },
      { key: 'height', label: 'Height (mm)', type: 'number', min: 1000, max: 3000 },
      { key: 'rear_brakes_type', label: 'Rear Brakes', type: 'select', opts: VEHICLE_SCHEMA.rearBrakesTypes },
      { key: 'steering_type', label: 'Steering Type', type: 'select', opts: VEHICLE_SCHEMA.steeringTypes },
    ],
  },
  {
    title: 'Performance & Engine',
    icon: Gauge,
    fields: [
      { key: 'engine_type', label: 'Engine Specification', type: 'select', opts: VEHICLE_SCHEMA.engineTypes },
      { key: 'max_torque', label: 'Max Torque (e.g. 100Nm@4000rpm)', type: 'text' },
      { key: 'max_power', label: 'Max Power (e.g. 80bhp@5500rpm)', type: 'text' },
    ],
  },
  {
    title: 'Safety Features',
    icon: Shield,
    fields: VEHICLE_SCHEMA.binaryFields.map(k => ({
      key: k,
      label: k.replace('is_', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      type: 'toggle',
    })),
  },
];

const SYNTHETIC_FORM_SECTIONS = [
  { title: 'Personal Info', icon: Shield, fields: [
    { key: 'age', label: 'Age', type: 'number', min: 18, max: 100 },
    { key: 'gender', label: 'Gender', type: 'select', opts: ['Male','Female','Other'] },
    { key: 'income', label: 'Annual Income (₹)', type: 'number', min: 0 },
    { key: 'occupation', label: 'Occupation', type: 'select', opts: ['Salaried','Self-Employed','Business','Retired','Student'] },
    { key: 'location', label: 'Location', type: 'select', opts: ['Urban','Suburban','Rural'] },
    { key: 'num_dependents', label: 'Number of Dependents', type: 'number', min: 0, max: 10 },
    { key: 'health_score', label: 'Health Score (0–1)', type: 'range', min: 0, max: 1, step: 0.01 },
  ]},
  { title: 'Policy Info', icon: Shield, fields: [
    { key: 'policy_type', label: 'Policy Type', type: 'select', opts: ['Health','Vehicle','Property','Life','Travel'] },
    { key: 'policy_tenure', label: 'Policy Tenure (years)', type: 'number', min: 1, max: 30 },
    { key: 'premium', label: 'Premium (₹)', type: 'number', min: 0 },
    { key: 'coverage_amount', label: 'Coverage Amount (₹)', type: 'number', min: 0 },
    { key: 'vehicle_age', label: 'Vehicle Age (years)', type: 'number', min: 0, max: 30 },
    { key: 'renewal_count', label: 'Renewals', type: 'number', min: 0, max: 30 },
  ]},
  { title: 'History', icon: Shield, fields: [
    { key: 'previous_claims', label: 'Previous Claims', type: 'number', min: 0, max: 20 },
    { key: 'payment_history', label: 'Payment History (0–1)', type: 'range', min: 0, max: 1, step: 0.01 },
  ]},
];

// ── Input field renderer ──────────────────────────────────────────────────────
function FieldInput({ field, value, onChange }) {
  if (field.type === 'select') {
    return (
      <select value={value} onChange={e => onChange(field.key, e.target.value)}
        className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-2 py-1.5 text-sm">
        {field.opts.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    );
  }
  if (field.type === 'range') {
    return (
      <div>
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>{field.min}</span>
          <span className="font-semibold text-indigo-600">{Number(value).toFixed(2)}</span>
          <span>{field.max}</span>
        </div>
        <input type="range" min={field.min} max={field.max} step={field.step || 0.01} value={value}
          onChange={e => onChange(field.key, parseFloat(e.target.value))}
          className="w-full accent-indigo-600" />
      </div>
    );
  }
  if (field.type === 'toggle') {
    return (
      <button type="button" onClick={() => onChange(field.key, value === 'Yes' ? 'No' : 'Yes')}
        className={`w-16 h-8 rounded-full transition-colors flex items-center px-1 ${
          value === 'Yes' ? 'bg-green-500' : 'bg-gray-300 dark:bg-gray-600'
        }`}>
        <span className={`w-6 h-6 bg-white rounded-full shadow transition-transform ${value === 'Yes' ? 'translate-x-8' : ''}`} />
      </button>
    );
  }
  if (field.type === 'text') {
    return (
      <input type="text" value={value}
        onChange={e => onChange(field.key, e.target.value)}
        placeholder={field.placeholder}
        className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-2 py-1.5 text-sm" />
    );
  }
  return (
    <input type="number" value={value} min={field.min} max={field.max} step={field.step || 1}
      onChange={e => onChange(field.key, parseFloat(e.target.value) || 0)}
      className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-2 py-1.5 text-sm" />
  );
}

function FormSection({ section, values, onChange }) {
  const Icon = section.icon;
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2 border-b pb-2 border-gray-200 dark:border-gray-700">
        <Icon className="w-4 h-4 text-indigo-500" /> {section.title}
      </h3>
      <div className={section.title === 'Safety Features' ? 'grid grid-cols-2 gap-3' : 'grid grid-cols-1 gap-3'}>
        {section.fields.map(field => (
          <div key={field.key}>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
              {field.label}
            </label>
            {field.type === 'toggle' ? (
              <div className="flex items-center gap-2">
                <FieldInput field={field} value={values[field.key] ?? 'No'} onChange={onChange} />
                <span className={`text-xs font-medium ${values[field.key] === 'Yes' ? 'text-green-600' : 'text-gray-400'}`}>
                  {values[field.key] ?? 'No'}
                </span>
              </div>
            ) : (
              <FieldInput field={field} value={values[field.key] ?? ''} onChange={onChange} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultPanel({ result, dataMode }) {
  if (!result) return null;
  const prob = result.claim_probability;
  const pct = Math.round(prob * 100);
  const riskColors = {
    Low: 'text-green-600', Moderate: 'text-yellow-600',
    High: 'text-orange-500', Critical: 'text-red-600',
  };
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 space-y-4 sticky top-4">
      <h3 className="font-semibold text-gray-900 dark:text-white text-base">Prediction Result</h3>

      {/* Probability gauge */}
      <div className="text-center py-4">
        <p className={`text-6xl font-bold ${riskColors[result.risk_category] || 'text-gray-900 dark:text-white'}`}>
          {pct}%
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Claim Probability</p>
        <div className="mt-3 flex justify-center">
          <RiskBadge category={result.risk_category} />
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className="h-3 rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            backgroundColor: prob >= 0.76 ? '#ef4444' : prob >= 0.51 ? '#f97316' : prob >= 0.26 ? '#eab308' : '#22c55e',
          }} />
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 text-center">
          <p className="text-lg font-bold text-gray-900 dark:text-white">{result.risk_score}/100</p>
          <p className="text-xs text-gray-500">AI Risk Score*</p>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 text-center">
          <p className="text-sm font-bold text-gray-900 dark:text-white">{result.prediction}</p>
          <p className="text-xs text-gray-500">Decision @{(result.threshold_used*100).toFixed(0)}%</p>
        </div>
      </div>

      <p className="text-xs text-gray-500 italic">*AI Risk Score — Project Scale only. Not an official insurance rating.</p>
      <p className="text-xs text-gray-500">Model: {result.model_name} | v{result.model_version}</p>
      <p className="text-xs text-gray-400">Dataset: {dataMode === 'real' ? 'Real Vehicle Insurance' : 'Synthetic'}</p>

      {/* Recommendations */}
      {result.recommendations?.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase">Recommendations</h4>
          {result.recommendations.map((r, i) => (
            <div key={i} className="flex items-start gap-2">
              <span className="text-indigo-500 mt-0.5">•</span>
              <p className="text-xs text-gray-600 dark:text-gray-400">{r}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const PredictClaim = () => {
  const [dataMode, setDataMode] = useState('real');
  const [activeThreshold, setActiveThreshold] = useState(null);
  const [values, setValues] = useState({ ...REAL_DEFAULTS });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modeLoading, setModeLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getDataMode().catch(() => ({ data: { data_mode: 'real' } })),
      getModelInfo().catch(() => ({ data: {} })),
    ]).then(([modeRes, infoRes]) => {
      const mode = modeRes.data?.data_mode || 'real';
      const thresh = infoRes.data?.default_threshold ?? null;
      setDataMode(mode);
      setActiveThreshold(thresh);
      setValues({
        ...(mode === 'real' ? REAL_DEFAULTS : SYNTHETIC_DEFAULTS),
        threshold: thresh,
      });
    }).finally(() => setModeLoading(false));
  }, []);

  const handleChange = (key, value) => {
    setValues(prev => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const payload = {
        ...values,
        store_prediction: true,
      };
      delete payload.threshold;
      const res = await predict(payload);
      setResult(res.data);
      toast.success('Prediction complete!');
    } catch (err) {
      const detail = err?.response?.data?.detail;
      toast.error(typeof detail === 'string' ? detail : 'Prediction failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const sections = dataMode === 'real' ? REAL_FORM_SECTIONS : SYNTHETIC_FORM_SECTIONS;

  if (modeLoading) return <LoadingSpinner size="lg" className="mt-20" />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Predict Claim</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {dataMode === 'real'
              ? 'Enter vehicle insurance policy details to get a claim prediction.'
              : 'Enter policyholder details to get a claim prediction.'}
          </p>
        </div>
        <div className={`px-3 py-1.5 rounded-full text-xs font-semibold ${
          dataMode === 'real'
            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
            : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
        }`}>
          {dataMode === 'real' ? '🚗 Real Vehicle Dataset' : '🧪 Synthetic Dataset'}
        </div>
      </div>

      {dataMode === 'real' && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 flex items-start gap-3">
          <Car className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-blue-800 dark:text-blue-300">Real Dataset Mode</p>
            <p className="text-sm text-blue-700 dark:text-blue-400 mt-1">
              Model trained on <strong>58,592 real vehicle insurance records</strong>. Calibrated decision threshold is <strong>{activeThreshold != null ? Number(activeThreshold).toFixed(2) : '—'}</strong>
              (optimized on out-of-fold probabilities to match empirical prevalence).
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        <div className="xl:col-span-3 space-y-4">
          {/* Policy ID */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Policy ID (optional)</label>
            <input type="text" value={values.policy_id || ''} onChange={e => handleChange('policy_id', e.target.value)}
              placeholder="e.g. POL-2024-001"
              className="w-full max-w-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-2 py-1.5 text-sm" />
          </div>

          {sections.map(section => (
            <FormSection key={section.title} section={section} values={values} onChange={handleChange} />
          ))}

          {/* Governed Decision Threshold (Read-Only) */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 border-l-4 border-indigo-500">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Authoritative Decision Threshold</span>
                <p className="text-sm font-medium text-gray-900 dark:text-white mt-0.5">
                  Locked at <strong className="text-indigo-600 dark:text-indigo-400 text-base">{activeThreshold != null ? Number(activeThreshold).toFixed(3) : '0.080'}</strong>
                </p>
              </div>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900/40 dark:text-indigo-300">
                Single Source of Truth
              </span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Threshold is governed strictly by the active Champion model release (<code className="text-xs">metadata.json</code>). Client overrides are disabled for production integrity. Probabilities &ge; threshold are classified as "Likely to Claim".
            </p>
          </div>

          <button onClick={handleSubmit} disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white font-semibold rounded-lg transition-colors text-base">
            {loading ? <><LoadingSpinner size="sm" /> Predicting...</> : 'Run Prediction'}
          </button>
        </div>

        {/* Results panel */}
        <div className="xl:col-span-1">
          <ResultPanel result={result} dataMode={dataMode} />
        </div>
      </div>

      <Disclaimer />
    </div>
  );
};

export default PredictClaim;
