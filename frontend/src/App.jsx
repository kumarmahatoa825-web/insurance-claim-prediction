import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout/Layout';

import Dashboard from './pages/Dashboard';
import PredictClaim from './pages/PredictClaim';
import WhatIfSimulator from './pages/WhatIfSimulator';
import BatchPrediction from './pages/BatchPrediction';
import Analytics from './pages/Analytics';
import ModelComparison from './pages/ModelComparison';
import Explainability from './pages/Explainability';
import PredictionHistory from './pages/PredictionHistory';
import DataQuality from './pages/DataQuality';
import ModelMonitoring from './pages/ModelMonitoring';
import ModelRegistry from './pages/ModelRegistry';
import About from './pages/About';

function App() {
  return (
    <Router>
      <Toaster position="top-right" />
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="predict" element={<PredictClaim />} />
          <Route path="simulator" element={<WhatIfSimulator />} />
          <Route path="batch" element={<BatchPrediction />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="models" element={<ModelComparison />} />
          <Route path="explain" element={<Explainability />} />
          <Route path="history" element={<PredictionHistory />} />
          <Route path="data-quality" element={<DataQuality />} />
          <Route path="monitoring" element={<ModelMonitoring />} />
          <Route path="registry" element={<ModelRegistry />} />
          <Route path="about" element={<About />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
