import React, { useState, useRef, useEffect } from 'react';
import { batchPredict, getDataMode } from '../services/api';
import { getRiskColor } from '../utils/riskUtils';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import RiskBadge from '../components/ui/RiskBadge';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, Download, AlertCircle, CheckCircle, FileText, Car } from 'lucide-react';
import toast from 'react-hot-toast';

const SYNTHETIC_COLS = [
  'age', 'gender', 'income', 'occupation', 'location', 'policy_type',
  'policy_tenure', 'premium', 'coverage_amount', 'vehicle_age',
  'num_dependents', 'previous_claims', 'payment_history', 'renewal_count', 'health_score'
];

const REAL_COLS = [
  'policy_id', 'policy_tenure', 'age_of_car', 'age_of_policyholder', 'area_cluster',
  'population_density', 'make', 'segment', 'model', 'fuel_type',
  'max_torque', 'max_power', 'engine_type', 'airbags', 'is_esc',
  'displacement', 'cylinder', 'transmission_type', 'gear_box', 'ncap_rating'
];

const RISK_COLORS = { Low: '#22c55e', Moderate: '#eab308', High: '#f97316', Critical: '#ef4444' };

const SYNTHETIC_SAMPLE = `age,gender,income,occupation,location,policy_type,policy_tenure,premium,coverage_amount,vehicle_age,num_dependents,previous_claims,payment_history,renewal_count,health_score
35,Male,600000,Salaried,Urban,Health,5,25000,1000000,0,2,1,0.85,4,0.75
45,Female,800000,Business,Suburban,Vehicle,8,40000,500000,5,3,2,0.70,7,0.65
28,Male,350000,Student,Rural,Life,2,15000,2000000,0,0,0,0.95,1,0.90
60,Female,1200000,Retired,Urban,Health,15,60000,3000000,0,1,3,0.60,14,0.50`;

const REAL_SAMPLE = `policy_id,policy_tenure,age_of_car,age_of_policyholder,area_cluster,population_density,make,segment,model,fuel_type,max_torque,max_power,engine_type,airbags,is_esc,is_adjustable_steering,is_tpms,is_parking_sensors,is_parking_camera,rear_brakes_type,displacement,cylinder,transmission_type,gear_box,steering_type,turning_radius,length,width,height,gross_weight,is_front_fog_lights,is_rear_window_wiper,is_rear_window_washer,is_rear_window_defogger,is_brake_assist,is_power_door_locks,is_central_locking,is_power_steering,is_driver_seat_height_adjustable,is_day_night_rear_view_mirror,is_ecw,is_speed_alert,ncap_rating
DEMO001,0.52,0.05,0.64,C1,4990,1,A,M1,CNG,60Nm@3500rpm,40.36bhp@6000rpm,F8D Petrol Engine,2,No,No,No,Yes,No,Drum,796,3,Manual,5,Power,4.6,3445,1515,1475,1185,No,No,No,No,No,No,No,Yes,No,No,No,Yes,0
DEMO002,0.67,0.02,0.38,C2,27003,1,A,M1,CNG,60Nm@3500rpm,40.36bhp@6000rpm,F8D Petrol Engine,2,No,No,No,Yes,No,Drum,796,3,Manual,5,Power,4.6,3445,1515,1475,1185,No,No,No,No,No,No,No,Yes,No,No,No,Yes,0
DEMO003,0.84,0.02,0.38,C3,4076,1,A,M1,CNG,60Nm@3500rpm,40.36bhp@6000rpm,F8D Petrol Engine,2,No,No,No,Yes,No,Drum,796,3,Manual,5,Power,4.6,3445,1515,1475,1185,No,No,No,No,No,No,No,Yes,No,No,No,Yes,0`;



const BatchPrediction = () => {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [errors, setErrors] = useState([]);
  const [dataMode, setDataMode] = useState('real');
  const fileRef = useRef();

  useEffect(() => {
    getDataMode().then(res => setDataMode(res.data?.data_mode || 'real')).catch(() => {});
  }, []);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.name.endsWith('.csv')) { setFile(f); setErrors([]); }
    else toast.error('Please upload a CSV file.');
  };

  const handleFileSelect = (e) => {
    const f = e.target.files[0];
    if (f) { setFile(f); setErrors([]); }
  };

  const handleSubmit = async () => {
    if (!file) { toast.error('Please select a CSV file first.'); return; }
    setLoading(true);
    setResults(null);
    setErrors([]);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await batchPredict(formData);
      setResults(res.data);
      toast.success(`Predicted ${res.data.total} policies successfully!`);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      if (typeof detail === 'object' && Array.isArray(detail?.errors)) {
        setErrors(detail.errors);
      } else if (typeof detail === 'string') {
        setErrors([detail]);
      } else {
        setErrors(['Batch prediction failed. Check CSV columns match the vehicle schema, contains no duplicate policy IDs, and excludes is_claim.']);
      }
      toast.error('Batch prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const downloadSample = () => {
    const csv = dataMode === 'real' ? REAL_SAMPLE : SYNTHETIC_SAMPLE;
    const fname = dataMode === 'real' ? 'sample_real_batch.csv' : 'sample_synthetic_batch.csv';
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = fname;
    a.click();
  };

  const downloadResults = () => {
    if (!results?.predictions) return;
    const headers = ['policy_id', 'claim_probability', 'non_claim_probability', 'risk_score', 'risk_category', 'prediction', 'model_version'];
    const rows = results.predictions.map(p => headers.map(h => p[h] ?? '').join(','));
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'prediction_results.csv';
    a.click();
  };

  const riskPieData = results?.risk_distribution
    ? Object.entries(results.risk_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const requiredCols = dataMode === 'real' ? REAL_COLS : SYNTHETIC_COLS;



  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Batch Prediction</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Upload a CSV file to predict multiple policies at once.
            {dataMode === 'real' && ' Uses real vehicle insurance dataset schema.'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`px-3 py-1.5 rounded-full text-xs font-semibold ${
            dataMode === 'real'
              ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
              : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
          }`}>
            {dataMode === 'real' ? '🚗 Real Dataset' : '🧪 Synthetic'}
          </div>
          <button onClick={downloadSample} className="flex items-center gap-2 px-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300">
            <Download className="w-4 h-4" /> Sample CSV
          </button>
        </div>
      </div>

      {/* Required columns */}
      <div className={`border rounded-lg p-4 ${
        dataMode === 'real'
          ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
          : 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
      }`}>
        <p className={`text-sm font-medium mb-2 ${dataMode === 'real' ? 'text-green-800 dark:text-green-300' : 'text-blue-800 dark:text-blue-300'}`}>
          {dataMode === 'real' ? 'Required CSV Columns (Vehicle Insurance Schema):' : 'Required CSV Columns:'}
        </p>
        <div className="flex flex-wrap gap-2">
          {requiredCols.map(col => (
            <span key={col} className={`px-2 py-0.5 text-xs rounded font-mono ${
              dataMode === 'real'
                ? 'bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-200'
                : 'bg-blue-100 dark:bg-blue-800 text-blue-700 dark:text-blue-200'
            }`}>{col}</span>
          ))}
          {dataMode === 'real' && <span className="text-xs text-green-600 dark:text-green-400 italic self-center">+ 23 more columns (see sample CSV)</span>}
        </div>
      </div>



      {/* Upload area */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
          dragging ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20' : 'border-gray-300 dark:border-gray-600 hover:border-indigo-400'
        }`}
      >
        <input ref={fileRef} type="file" accept=".csv" onChange={handleFileSelect} className="hidden" />
        <Upload className="w-10 h-10 mx-auto text-gray-400 mb-3" />
        {file ? (
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-white flex items-center justify-center gap-2">
              <FileText className="w-4 h-4 text-indigo-500" /> {file.name}
            </p>
            <p className="text-xs text-gray-500 mt-1">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
        ) : (
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Drag & drop CSV here or click to browse</p>
            <p className="text-xs text-gray-400 mt-1">Maximum file size: 15 MB</p>
          </div>
        )}
      </div>

      {errors.length > 0 && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 rounded-lg p-4">
          <p className="text-sm font-medium text-red-800 dark:text-red-300 flex items-center gap-2 mb-2">
            <AlertCircle className="w-4 h-4" /> Validation Errors
          </p>
          <ul className="text-sm text-red-700 dark:text-red-400 list-disc list-inside space-y-1">
            {errors.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </div>
      )}

      <div className="flex justify-center">
        <button
          onClick={handleSubmit}
          disabled={loading || !file}
          className="flex items-center gap-2 px-8 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white font-semibold rounded-lg transition-colors"
        >
          {loading ? <><LoadingSpinner size="sm" /> Processing...</> : <><Upload className="w-4 h-4" /> Run Batch Prediction</>}
        </button>
      </div>

      {results && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Policies', value: results.total },
              { label: 'High Risk', value: (results.risk_distribution?.High || 0) + (results.risk_distribution?.Critical || 0) },
              { label: 'Low Risk', value: results.risk_distribution?.Low || 0 },
              { label: 'Moderate Risk', value: results.risk_distribution?.Moderate || 0 },
            ].map(s => (
              <div key={s.label} className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow text-center">
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{s.value}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{s.label}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Risk distribution chart */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
              <h3 className="text-base font-semibold text-gray-900 dark:text-white mb-4">Risk Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={riskPieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {riskPieData.map((entry) => (
                      <Cell key={entry.name} fill={RISK_COLORS[entry.name] || '#6366f1'} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Warnings */}
            {results.warnings?.length > 0 && (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-5">
                <h3 className="text-base font-semibold text-yellow-800 dark:text-yellow-300 mb-3 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" /> Data Warnings
                </h3>
                <ul className="space-y-2">
                  {results.warnings.map((w, i) => <li key={i} className="text-sm text-yellow-700 dark:text-yellow-400">{w}</li>)}
                </ul>
              </div>
            )}
          </div>

          {/* Results table */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold text-gray-900 dark:text-white">Prediction Results ({results.total})</h3>
              <button onClick={downloadResults} className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors">
                <Download className="w-4 h-4" /> Download CSV
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    {['Policy ID', 'Claim Probability', 'Risk Score', 'Risk Category', 'Prediction'].map(h => (
                      <th key={h} className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {results.predictions.slice(0, 100).map((p, i) => (
                    <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                      <td className="px-4 py-3 font-mono text-xs text-gray-600 dark:text-gray-300">{p.policy_id || `row_${i+1}`}</td>
                      <td className="px-4 py-3 font-semibold text-gray-900 dark:text-white">{(p.claim_probability * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3 text-gray-700 dark:text-gray-300">{p.risk_score}/100</td>
                      <td className="px-4 py-3"><RiskBadge category={p.risk_category} /></td>
                      <td className="px-4 py-3 text-xs text-gray-600 dark:text-gray-300">{p.prediction}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {results.total > 100 && (
                <p className="text-center text-sm text-gray-500 p-4">Showing first 100 of {results.total} results. Download CSV for full results.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchPrediction;
