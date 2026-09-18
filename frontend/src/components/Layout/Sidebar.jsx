import { Link, useLocation } from 'react-router-dom';
import clsx from 'clsx';
import { 
  Home, 
  Search, 
  Gamepad2, 
  Package, 
  BarChart2, 
  Award, 
  HelpCircle, 
  Clock, 
  Database, 
  Activity, 
  GitBranch,
  Info,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

const navItems = [
  { name: 'Dashboard', path: '/', icon: Home },
  { name: 'Predict Claim', path: '/predict', icon: Search },
  { name: 'What-If Simulator', path: '/simulator', icon: Gamepad2 },
  { name: 'Batch Prediction', path: '/batch', icon: Package },
  { name: 'Analytics', path: '/analytics', icon: BarChart2 },
  { name: 'Model Comparison', path: '/models', icon: Award },
  { name: 'Explainability', path: '/explain', icon: HelpCircle },
  { name: 'History', path: '/history', icon: Clock },
  { name: 'Data Quality', path: '/data-quality', icon: Database },
  { name: 'Model Monitoring', path: '/monitoring', icon: Activity },
  { name: 'Model Registry', path: '/registry', icon: GitBranch },
  { name: 'About', path: '/about', icon: Info },
];

const Sidebar = ({ isOpen, setIsOpen }) => {
  const location = useLocation();

  return (
    <aside className={clsx(
      "bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col transition-all duration-300",
      isOpen ? "w-64" : "w-20"
    )}>
      <div className="h-16 flex items-center justify-between px-4 border-b border-gray-200 dark:border-gray-700">
        {isOpen && <span className="font-bold text-xl text-indigo-600 dark:text-indigo-400">InsureAI</span>}
        <button 
          onClick={() => setIsOpen(!isOpen)}
          className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500"
        >
          {isOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} className="mx-auto" />}
        </button>
      </div>
      <nav className="flex-1 overflow-y-auto py-4 space-y-1 px-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          return (
            <Link
              key={item.name}
              to={item.path}
              className={clsx(
                "flex items-center px-3 py-2.5 rounded-md transition-colors",
                isActive 
                  ? "bg-indigo-50 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300" 
                  : "text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700",
                !isOpen && "justify-center"
              )}
              title={!isOpen ? item.name : undefined}
            >
              <Icon size={20} className={clsx("shrink-0", isOpen && "mr-3", isActive ? "text-indigo-600 dark:text-indigo-400" : "text-gray-400")} />
              {isOpen && <span className="truncate">{item.name}</span>}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
};

export default Sidebar;
