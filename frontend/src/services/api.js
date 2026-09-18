import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 60000,  // 60s for batch ops
});

export const healthCheck = () => api.get('/health');
export const predict = (data) => api.post('/predict', data);
export const explain = (data) => api.post('/explain', data);
export const batchPredict = (formData) => api.post('/batch-predict', formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
  timeout: 120000,
});
export const getModelInfo = () => api.get('/model-info');
export const getAnalytics = () => api.get('/analytics', { timeout: 60000 });
export const getModelComparison = () => api.get('/model-comparison');
export const getDataQuality = () => api.get('/data-quality', { timeout: 60000 });
export const getFairness = () => api.get('/fairness');
export const getHistory = (params) => api.get('/history', { params });
export const deleteHistory = (id) => api.delete(`/history/${id}`);
export const exportHistory = () => api.get('/history/export', { responseType: 'blob' });
export const getThresholdAnalysis = () => api.get('/threshold-analysis');
export const getCostAnalysis = (data) => api.post('/cost-analysis', data);
export const getCalibration = () => api.get('/calibration');
export const retrain = (formData) => api.post('/retrain', formData);
export const confirmRetrain = () => api.post('/retrain/confirm');
export const getDataMode = () => api.get('/data-mode');
export const batchPredictRealTest = () => api.post('/batch-predict-real-test', {}, { timeout: 120000 });
export const getMonitoring = () => api.get('/monitoring');
export const getRegistry = () => api.get('/registry');
export const approveCandidate = (version) => api.post(`/registry/approve/${version}`);
export const promoteCandidate = (version) => api.post(`/registry/promote/${version}`);
export const rollbackChampion = () => api.post('/registry/rollback');
export const compareCandidate = (version) => api.get(`/registry/compare/${version}`);

export default api;

