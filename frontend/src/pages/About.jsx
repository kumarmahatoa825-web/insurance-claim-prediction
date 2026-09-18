import React from 'react';
import Disclaimer from '../components/ui/Disclaimer';

const About = () => {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">About InsureAI</h1>
      
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow space-y-4">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Project Overview</h2>
        <p className="text-gray-700 dark:text-gray-300">
          InsureAI is a modern analytics dashboard designed to assist insurance professionals in assessing claim probabilities using advanced machine learning models.
        </p>
        
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mt-6">Technology Stack</h2>
        <ul className="list-disc pl-5 text-gray-700 dark:text-gray-300 space-y-2">
          <li><strong>Frontend:</strong> React 18, Vite, Tailwind CSS, Recharts</li>
          <li><strong>Backend:</strong> FastAPI (Python), scikit-learn, XGBoost</li>
          <li><strong>Explainability:</strong> SHAP (SHapley Additive exPlanations)</li>
        </ul>
        
        <div className="mt-8">
          <Disclaimer />
        </div>
      </div>
    </div>
  );
};

export default About;
