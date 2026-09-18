import React, { useState, useEffect } from 'react';
import { getRegistry, approveCandidate, promoteCandidate, rollbackChampion, compareCandidate } from '../services/api';
import { formatMetric } from '../utils/formatters';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ErrorMessage from '../components/ui/ErrorMessage';
import toast from 'react-hot-toast';
import {
  Award, History, ArrowUpRight, RotateCcw, CheckCircle2,
  XCircle, AlertCircle, Shield, GitBranch, Cpu, Calendar, Activity
} from 'lucide-react';

export default function ModelRegistry() {
  const [registry, setRegistry] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [comparisonModal, setComparisonModal] = useState(null);

  const fetchRegistry = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getRegistry();
      setRegistry(res.data);
    } catch (err) {
      setError('Failed to load Model Registry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRegistry();
  }, []);

  const handleApprove = async (version) => {
    setActionLoading(true);
    try {
      await approveCandidate(version);
      toast.success(`Candidate ${version} approved for promotion!`);
      fetchRegistry();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Approval failed.');
    } finally {
      setActionLoading(false);
    }
  };

  const handlePromote = async (version) => {
    if (!window.confirm(`Are you sure you want to promote candidate ${version} to Champion?`)) return;
    setActionLoading(true);
    try {
      await promoteCandidate(version);
      toast.success(`Candidate ${version} promoted to Champion successfully!`);
      fetchRegistry();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Promotion failed.');
    } finally {
      setActionLoading(false);
    }
  };

  const handleRollback = async () => {
    if (!window.confirm('Roll back to the previous Champion model?')) return;
    setActionLoading(true);
    try {
      await rollbackChampion();
      toast.success('Successfully rolled back to previous Champion!');
      fetchRegistry();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Rollback failed.');
    } finally {
      setActionLoading(false);
    }
  };

  const handleCompare = async (version) => {
    try {
      const res = await compareCandidate(version);
      setComparisonModal(res.data);
    } catch (err) {
      toast.error('Failed to compare candidate.');
    }
  };

  if (loading) return <LoadingSpinner size="lg" className="mt-20" />;
  if (error && !registry) return <ErrorMessage message={error} onRetry={fetchRegistry} />;

  const champion = registry?.champion;
  const candidates = registry?.candidates || [];
  const history = registry?.history || [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Registry & Governance</h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300">
              Champion/Challenger Framework
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Track ML model lifecycle, validate candidates against strict promotion gates, and manage zero-downtime rollbacks.
          </p>
        </div>
        {history.length > 0 && (
          <button
            onClick={handleRollback}
            disabled={actionLoading}
            className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium border border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/30 rounded-lg hover:bg-amber-100 dark:hover:bg-amber-900/40 transition disabled:opacity-50"
          >
            <RotateCcw className="w-4 h-4" /> Rollback to Previous Champion
          </button>
        )}
      </div>

      {/* Active Champion Card */}
      {champion ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl border-2 border-indigo-200 dark:border-indigo-800/80 p-6 shadow-sm ring-2 ring-indigo-500/10">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-gray-100 dark:border-gray-700">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-indigo-50 dark:bg-indigo-900/50 rounded-xl">
                <Award className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-bold text-gray-900 dark:text-white">{champion.model_name}</h2>
                  <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-green-100 text-green-800 dark:bg-green-950/40 dark:text-green-300">
                    ACTIVE CHAMPION
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-0.5">Version: {champion.version} • Registered: {new Date(champion.registered_at).toLocaleString()}</p>
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs text-gray-400">Decision Threshold:</span>
              <p className="text-xl font-mono font-bold text-indigo-600 dark:text-indigo-400">{champion.threshold != null ? champion.threshold.toFixed(2) : '—'}</p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 mt-5">
            <div className="text-center p-2.5 bg-gray-50 dark:bg-gray-900/40 rounded-lg">
              <span className="text-[11px] text-gray-400">CV F1 (Mean)</span>
              <p className="text-base font-black text-gray-900 dark:text-white mt-1">{formatMetric(champion.metrics?.cv_f1_mean)}</p>
            </div>
            <div className="text-center p-2.5 bg-gray-50 dark:bg-gray-900/40 rounded-lg">
              <span className="text-[11px] text-gray-400">CV ROC-AUC</span>
              <p className="text-base font-black text-indigo-600 dark:text-indigo-400 mt-1">{formatMetric(champion.metrics?.cv_roc_auc_mean)}</p>
            </div>
            <div className="text-center p-2.5 bg-gray-50 dark:bg-gray-900/40 rounded-lg">
              <span className="text-[11px] text-gray-400">CV PR-AUC</span>
              <p className="text-base font-black text-blue-600 dark:text-blue-400 mt-1">{formatMetric(champion.metrics?.cv_pr_auc_mean)}</p>
            </div>
            <div className="text-center p-2.5 bg-gray-50 dark:bg-gray-900/40 rounded-lg">
              <span className="text-[11px] text-gray-400">Test ROC-AUC</span>
              <p className="text-base font-black text-yellow-600 dark:text-yellow-400 mt-1">{formatMetric(champion.metrics?.test_roc_auc)}</p>
            </div>
            <div className="text-center p-2.5 bg-gray-50 dark:bg-gray-900/40 rounded-lg">
              <span className="text-[11px] text-gray-400">Test PR-AUC</span>
              <p className="text-base font-black text-purple-600 dark:text-purple-400 mt-1">{formatMetric(champion.metrics?.test_pr_auc)}</p>
            </div>
            <div className="text-center p-2.5 bg-gray-50 dark:bg-gray-900/40 rounded-lg">
              <span className="text-[11px] text-gray-400">Test F1</span>
              <p className="text-base font-black text-green-600 dark:text-green-400 mt-1">{formatMetric(champion.metrics?.test_f1)}</p>
            </div>
            <div className="text-center p-2.5 bg-gray-50 dark:bg-gray-900/40 rounded-lg">
              <span className="text-[11px] text-gray-400">Brier Score</span>
              <p className="text-base font-black text-teal-600 dark:text-teal-400 mt-1">{formatMetric(champion.metrics?.brier_score, 4)}</p>
            </div>
            <div className="text-center p-2.5 bg-gray-50 dark:bg-gray-900/40 rounded-lg">
              <span className="text-[11px] text-gray-400">Features</span>
              <p className="text-base font-black text-gray-700 dark:text-gray-300 mt-1">{champion.feature_count ?? '—'}</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-8 text-center bg-gray-50 dark:bg-gray-800 rounded-xl border border-dashed border-gray-300">
          <p className="text-gray-500">No active Champion model registered yet.</p>
        </div>
      )}

      {/* Candidate Models Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm">
        <h2 className="text-base font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-indigo-600" /> Candidate Models (Challengers)
        </h2>
        {candidates.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-50 dark:bg-gray-700/50 text-xs uppercase text-gray-500">
                <tr>
                  <th className="px-4 py-3">Version</th>
                  <th className="px-4 py-3">Algorithm</th>
                  <th className="px-4 py-3">Test ROC-AUC</th>
                  <th className="px-4 py-3">Test F1</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                {candidates.map(cand => (
                  <tr key={cand.version}>
                    <td className="px-4 py-3 font-mono font-medium text-gray-900 dark:text-white">{cand.version}</td>
                    <td className="px-4 py-3 text-gray-600 dark:text-gray-300">{cand.model_name}</td>
                    <td className="px-4 py-3 font-semibold text-indigo-600">{formatMetric(cand.metrics?.test_roc_auc)}</td>
                    <td className="px-4 py-3 font-semibold text-green-600">{formatMetric(cand.metrics?.test_f1)}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-xs border ${
                        cand.approval_status === 'approved'
                          ? 'bg-green-50 text-green-700 border-green-200 dark:bg-green-950/30 dark:text-green-300'
                          : 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/30 dark:text-amber-300'
                      }`}>
                        {cand.approval_status || 'Pending'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right space-x-2">
                      <button
                        onClick={() => handleCompare(cand.version)}
                        className="px-2.5 py-1 text-xs font-semibold border rounded text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700"
                      >
                        Compare
                      </button>
                      {cand.approval_status !== 'approved' ? (
                        <button
                          onClick={() => handleApprove(cand.version)}
                          disabled={actionLoading}
                          className="px-2.5 py-1 text-xs font-semibold bg-amber-500 hover:bg-amber-600 text-white rounded disabled:opacity-50"
                        >
                          Approve
                        </button>
                      ) : (
                        <button
                          onClick={() => handlePromote(cand.version)}
                          disabled={actionLoading}
                          className="px-2.5 py-1 text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white rounded disabled:opacity-50"
                        >
                          Promote to Champion
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-gray-500">No candidate models in queue. Subsequent retrains will populate this list as challengers.</p>
        )}
      </div>

      {/* Audit History */}
      {history.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm">
          <h2 className="text-base font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <History className="w-4 h-4 text-gray-500" /> Model Audit & Archival History
          </h2>
          <div className="space-y-3">
            {history.map((hist, idx) => (
              <div key={idx} className="flex items-center justify-between p-3.5 rounded-lg border border-gray-100 dark:border-gray-700 text-xs">
                <div>
                  <span className="font-semibold text-gray-900 dark:text-white font-mono">{hist.version}</span>
                  <span className="ml-2 text-gray-500">({hist.model_name})</span>
                  <p className="text-gray-400 mt-0.5">Archived: {new Date(hist.archived_at || hist.registered_at).toLocaleString()} • Status: {hist.status}</p>
                </div>
                <div className="flex items-center gap-4 text-right">
                  <div>
                    <span className="text-gray-400">ROC-AUC:</span>
                    <span className="ml-1 font-mono font-bold text-gray-700 dark:text-gray-300">{formatMetric(hist.metrics?.test_roc_auc)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">F1:</span>
                    <span className="ml-1 font-mono font-bold text-green-600">{formatMetric(hist.metrics?.test_f1)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Comparison Modal */}
      {comparisonModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl max-w-lg w-full p-6 shadow-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">Champion vs Challenger Comparison</h3>
            <p className="text-xs text-gray-500 mb-4">
              Comparing Candidate <strong>{comparisonModal.candidate_version}</strong> with Active Champion <strong>{comparisonModal.champion_version}</strong>.
            </p>

            <div className="space-y-2 mb-6">
              {Object.entries(comparisonModal.checks || {}).map(([check, passed]) => (
                <div key={check} className="flex items-center justify-between text-xs p-2 rounded bg-gray-50 dark:bg-gray-900/50">
                  <span className="text-gray-700 dark:text-gray-300 capitalize">{check.replace(/_/g, ' ')}</span>
                  <span className={`px-2 py-0.5 rounded font-semibold ${passed ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {passed ? 'PASS' : 'FAIL'}
                  </span>
                </div>
              ))}
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-gray-100 dark:border-gray-700">
              <span className={`text-xs font-bold uppercase ${
                comparisonModal.recommendation === 'promote' ? 'text-green-600' : 'text-amber-600'
              }`}>
                Recommendation: {comparisonModal.recommendation}
              </span>
              <button
                onClick={() => setComparisonModal(null)}
                className="px-4 py-1.5 text-xs font-medium border rounded-lg hover:bg-gray-50"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
