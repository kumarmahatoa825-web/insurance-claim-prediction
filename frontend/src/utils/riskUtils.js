export const getRiskCategory = (score) => {
  if (score < 25) return 'Low';
  if (score < 50) return 'Moderate';
  if (score < 75) return 'High';
  return 'Critical';
};

export const getRiskColor = (category) => {
  switch (category?.toLowerCase()) {
    case 'low': return 'text-green-600 dark:text-green-400';
    case 'moderate': return 'text-yellow-600 dark:text-yellow-400';
    case 'high': return 'text-orange-600 dark:text-orange-400';
    case 'critical': return 'text-red-600 dark:text-red-400';
    default: return 'text-gray-600 dark:text-gray-400';
  }
};

export const getRiskBgColor = (category) => {
  switch (category?.toLowerCase()) {
    case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
    case 'moderate': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
    case 'high': return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400';
    case 'critical': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
    default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400';
  }
};

// Alias for backwards compatibility
export const getRiskBg = getRiskBgColor;

