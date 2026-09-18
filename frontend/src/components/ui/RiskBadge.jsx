import clsx from 'clsx';
import { getRiskColor, getRiskBgColor } from '../../utils/riskUtils';

const RiskBadge = ({ category, className }) => {
  if (!category) return null;
  
  return (
    <span className={clsx(
      "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
      getRiskBgColor(category),
      className
    )}>
      {category} Risk
    </span>
  );
};

export default RiskBadge;
