import { useState } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell
} from 'recharts';

const activityColors = {
  Walking: '#8884d8',
  'Walking Upstairs': '#82ca9d',
  'Walking Downstairs': '#ffc658',
  Sitting: '#ff7300',
  Standing: '#0088fe',
  Laying: '#00C49F'
};

export default function HARCharts() {
  const [activeModel, setActiveModel] = useState('CNN');

  const accuracyData = [
    { epoch: 1, CNN: 0.75, LSTM: 0.72 },
    { epoch: 5, CNN: 0.82, LSTM: 0.78 },
    { epoch: 10, CNN: 0.87, LSTM: 0.83 },
    { epoch: 15, CNN: 0.89, LSTM: 0.86 },
    { epoch: 20, CNN: 0.91, LSTM: 0.88 },
    { epoch: 25, CNN: 0.92, LSTM: 0.89 },
    { epoch: 30, CNN: 0.923, LSTM: 0.905 }
  ];

  const activityDistribution = [
    { activity: 'Walking', samples: 1722 },
    { activity: 'Walking Upstairs', samples: 1544 },
    { activity: 'Walking Downstairs', samples: 1406 },
    { activity: 'Sitting', samples: 1777 },
    { activity: 'Standing', samples: 1906 },
    { activity: 'Laying', samples: 1944 }
  ];

  const modelPerformance = [
    { metric: 'Accuracy', CNN: 92.3, LSTM: 90.5 },
    { metric: 'Precision', CNN: 91.8, LSTM: 89.9 },
    { metric: 'Recall', CNN: 92.1, LSTM: 90.2 },
    { metric: 'F1-Score', CNN: 91.9, LSTM: 90.0 }
  ];

  return (
    <div className="space-y-8">
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Training Accuracy Over Epochs</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={accuracyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="CNN" stroke="#8884d8" strokeWidth={2} />
            <Line type="monotone" dataKey="LSTM" stroke="#82ca9d" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Activity Distribution in Dataset</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={activityDistribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="activity" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="samples">
              {activityDistribution.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={Object.values(activityColors)[index]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Model Performance Comparison</h3>
        <div className="flex space-x-4 mb-4">
          <button
            className={`px-4 py-2 rounded ${activeModel === 'CNN' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setActiveModel('CNN')}
          >
            CNN
          </button>
          <button
            className={`px-4 py-2 rounded ${activeModel === 'LSTM' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setActiveModel('LSTM')}
          >
            LSTM
          </button>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={modelPerformance}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis domain={[80, 100]} />
            <Tooltip />
            <Legend />
            <Bar dataKey={activeModel} fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Activity Distribution (Pie Chart)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={activityDistribution}
              dataKey="samples"
              nameKey="activity"
              cx="50%"
              cy="50%"
              outerRadius={100}
              label
            >
              {activityDistribution.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={Object.values(activityColors)[index]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
