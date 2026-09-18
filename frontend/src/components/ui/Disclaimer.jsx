import { AlertTriangle } from 'lucide-react';

const Disclaimer = () => {
  return (
    <div className="mt-8 bg-blue-50 dark:bg-blue-900/30 border-l-4 border-blue-500 p-4 rounded-r-md">
      <div className="flex">
        <div className="flex-shrink-0">
          <AlertTriangle className="h-5 w-5 text-blue-400" aria-hidden="true" />
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300">
            Responsible AI Disclaimer
          </h3>
          <div className="mt-2 text-sm text-blue-700 dark:text-blue-200">
            <p>
              This tool provides AI-assisted predictions based on historical patterns. It should be used to support, not replace, human decision-making. 
              Always review high-risk flags and consult established policies before finalizing a claim assessment.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Disclaimer;
