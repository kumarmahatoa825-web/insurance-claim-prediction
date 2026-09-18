import React, { useEffect, useState, useCallback } from 'react';
import { getHistory, deleteHistory, exportHistory } from '../services/api';
import RiskBadge from '../components/ui/RiskBadge';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ErrorMessage from '../components/ui/ErrorMessage';
import EmptyState from '../components/ui/EmptyState';
import { Trash2, Download, Search, Filter, ChevronLeft, ChevronRight } from 'lucide-react';
import toast from 'react-hot-toast';

const RISK_CATS = ['', 'Low', 'Moderate', 'High', 'Critical'];
const SORT_COLS = [
  { key: 'timestamp', label: 'Date' },
  { key: 'claim_probability', label: 'Probability' },
  { key: 'risk_score', label: 'Risk Score' },
];

const PredictionHistory = () => {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState('');
  const [sortBy, setSortBy] = useState('timestamp');
  const [sortOrder, setSortOrder] = useState('desc');
  const [deletingId, setDeletingId] = useState(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getHistory({ page, page_size: 15, search, risk_category: riskFilter, sort_by: sortBy, sort_order: sortOrder });
      setItems(res.data.items || []);
      setTotal(res.data.total || 0);
      setPages(res.data.pages || 1);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to load history.');
    } finally {
      setLoading(false);
    }
  }, [page, search, riskFilter, sortBy, sortOrder]);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this prediction record?')) return;
    setDeletingId(id);
    try {
      await deleteHistory(id);
      toast.success('Prediction deleted.');
      fetchHistory();
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Delete failed.');
    } finally {
      setDeletingId(null);
    }
  };

  const handleExport = async () => {
    try {
      const res = await exportHistory();
      const url = URL.createObjectURL(new Blob([res.data], { type: 'text/csv' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = 'prediction_history.csv';
      a.click();
      URL.revokeObjectURL(url);
      toast.success('History exported!');
    } catch {
      toast.error('Export failed.');
    }
  };

  const handleSort = (col) => {
    if (sortBy === col) setSortOrder(o => o === 'desc' ? 'asc' : 'desc');
    else { setSortBy(col); setSortOrder('desc'); }
    setPage(1);
  };

  const SortIcon = ({ col }) => sortBy === col ? (sortOrder === 'desc' ? ' ▼' : ' ▲') : '';

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Prediction History</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{total} predictions stored locally.</p>
        </div>
        <button onClick={handleExport} className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors">
          <Download className="w-4 h-4" /> Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text" placeholder="Search by policy ID..."
            value={search} onChange={e => { setSearch(e.target.value); setPage(1); }}
            className="w-full pl-9 pr-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm focus:ring-2 focus:ring-indigo-500"
          />
        </div>
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <select
            value={riskFilter} onChange={e => { setRiskFilter(e.target.value); setPage(1); }}
            className="pl-9 pr-8 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
          >
            <option value="">All Risk Levels</option>
            {RISK_CATS.filter(Boolean).map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {loading && <LoadingSpinner size="md" />}
      {error && <ErrorMessage message={error} onRetry={fetchHistory} />}

      {!loading && !error && items.length === 0 && (
        <EmptyState
          title="No predictions yet"
          description="Go to 'Predict Claim' to make your first prediction. It will appear here."
          icon="📋"
        />
      )}

      {!loading && items.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  {[
                    { key: 'timestamp', label: 'Date/Time' },
                    { key: null, label: 'Policy ID' },
                    { key: 'claim_probability', label: 'Probability' },
                    { key: 'risk_score', label: 'Risk Score' },
                    { key: null, label: 'Risk Category' },
                    { key: null, label: 'Prediction' },
                    { key: null, label: 'Model' },
                    { key: null, label: '' },
                  ].map((col, i) => (
                    <th key={i}
                      onClick={col.key ? () => handleSort(col.key) : undefined}
                      className={`px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase ${col.key ? 'cursor-pointer hover:text-indigo-600' : ''}`}>
                      {col.label}<SortIcon col={col.key} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {items.map(item => (
                  <tr key={item.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="px-4 py-3 text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
                      {item.timestamp ? new Date(item.timestamp).toLocaleString() : '—'}
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-gray-600 dark:text-gray-300">{item.policy_id || '—'}</td>
                    <td className="px-4 py-3 font-semibold text-gray-900 dark:text-white">
                      {(item.claim_probability * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-gray-700 dark:text-gray-300">{item.risk_score}/100</td>
                    <td className="px-4 py-3"><RiskBadge category={item.risk_category} /></td>
                    <td className="px-4 py-3 text-xs text-gray-600 dark:text-gray-300">{item.prediction}</td>
                    <td className="px-4 py-3 text-xs text-gray-500 dark:text-gray-400">{item.model_version}</td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleDelete(item.id)}
                        disabled={deletingId === item.id}
                        className="p-1.5 text-red-500 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                        title="Delete"
                      >
                        {deletingId === item.id ? <LoadingSpinner size="sm" /> : <Trash2 className="w-4 h-4" />}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {pages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-500">Page {page} of {pages} ({total} total)</p>
              <div className="flex gap-2">
                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                  className="p-2 rounded-lg border border-gray-300 dark:border-gray-600 disabled:opacity-50 hover:bg-gray-50 dark:hover:bg-gray-700">
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page === pages}
                  className="p-2 rounded-lg border border-gray-300 dark:border-gray-600 disabled:opacity-50 hover:bg-gray-50 dark:hover:bg-gray-700">
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PredictionHistory;
