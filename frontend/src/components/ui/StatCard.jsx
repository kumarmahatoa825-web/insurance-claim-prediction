import React from 'react';

const StatCard = ({ title, value, icon: Icon, description, trend, trendValue }) => {
  return (
    <div className="bg-white dark:bg-gray-800 overflow-hidden shadow rounded-lg">
      <div className="p-5">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            {Icon && <Icon className="h-6 w-6 text-gray-400 dark:text-gray-500" aria-hidden="true" />}
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">{title}</dt>
              <dd>
                <div className="text-lg font-medium text-gray-900 dark:text-white">{value}</div>
              </dd>
            </dl>
          </div>
        </div>
      </div>
      {(description || trend) && (
        <div className="bg-gray-50 dark:bg-gray-700/50 px-5 py-3">
          <div className="text-sm text-gray-500 dark:text-gray-400 flex justify-between">
            <span>{description}</span>
            {trend && (
              <span className={trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
                {trendValue}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default StatCard;
