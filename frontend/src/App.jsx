
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const App = () => {
  const [liveData, setLiveData] = useState(null);
  const [aqiHistory, setAqiHistory] = useState([]);
  const [alertThreshold, setAlertThreshold] = useState(150);
  const [showAlert, setShowAlert] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/live-data');
        const data = await response.json();
        setLiveData(data);
        setAqiHistory(prev => [...prev.slice(-9), { name: new Date().toLocaleTimeString(), aqi: data.aqi }]);
        if (data.aqi > alertThreshold) {
          setShowAlert(true);
        } else {
          setShowAlert(false);
        }
      } catch (error) {
        console.error("Error fetching live data:", error);
      }
    };

    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [alertThreshold]);

  const getAqiColor = (aqi) => {
    if (aqi <= 50) return '#4CAF50'; // Green
    if (aqi <= 100) return '#FFEB3B'; // Yellow
    if (aqi <= 150) return '#FF9800'; // Orange
    if (aqi <= 200) return '#F44336'; // Red
    if (aqi <= 300) return '#9C27B0'; // Purple
    return '#795548'; // Brown
  };

  if (!liveData) {
    return <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">Loading...</div>;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-blue-900 text-white p-8 font-sans">
      <header className="flex justify-between items-center mb-8">
        <h1 className="text-4xl font-bold">AQI.now</h1>
        <div className="text-right">
          <p className="text-xl">{liveData.location}</p>
          <p className="text-sm">{new Date().toLocaleString()}</p>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main AQI Display */}
        <motion.div 
          className="lg:col-span-2 bg-white/10 backdrop-blur-md rounded-xl p-8 flex flex-col items-center justify-center text-center"
          initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        >
          <p className="text-2xl mb-2">Current AQI</p>
          <h2 className="text-8xl font-bold" style={{ color: getAqiColor(liveData.aqi) }}>{liveData.aqi}</h2>
          <p className="text-3xl mt-2" style={{ color: getAqiColor(liveData.aqi) }}>{liveData.category}</p>
        </motion.div>

        {/* Weather Card */}
        <motion.div 
          className="bg-white/10 backdrop-blur-md rounded-xl p-8 flex flex-col justify-center"
          initial={{ opacity: 0, x: 50 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.2 }}
        >
          <p className="text-2xl mb-4">Environment</p>
          <div className="text-xl">Temperature: {liveData.temperature}°C</div>
          <div className="text-xl">Humidity: {liveData.humidity}%</div>
        </motion.div>

        {/* Pollutants Details */}
        <div className="lg:col-span-3 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {Object.entries(liveData.pollutants).map(([key, value]) => (
            <motion.div 
              key={key} 
              className="bg-white/10 backdrop-blur-md rounded-xl p-4 text-center"
              initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.3, delay: 0.1 * Object.keys(liveData.pollutants).indexOf(key) }}
            >
              <p className="font-bold text-lg">{key}</p>
              <p className="text-2xl">{value}</p>
            </motion.div>
          ))}
        </div>

        {/* AQI Trend Graph */}
        <motion.div 
          className="lg:col-span-3 bg-white/10 backdrop-blur-md rounded-xl p-8"
          initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.4 }}
        >
          <h3 className="text-2xl mb-4">AQI Trend (Last 10 Readings)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={aqiHistory}>
              <defs>
                <linearGradient id="colorAqi" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={getAqiColor(liveData.aqi)} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={getAqiColor(liveData.aqi)} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="name" stroke="#FFFFFF" />
              <YAxis stroke="#FFFFFF" />
              <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
              <Area type="monotone" dataKey="aqi" stroke={getAqiColor(liveData.aqi)} fillOpacity={1} fill="url(#colorAqi)" />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* AI Prediction & Alert System */}
        <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* AI Prediction Panel */}
          <AIPredictionPanel />

          {/* Alert System */}
          <AlertSystem 
            alertThreshold={alertThreshold} 
            setAlertThreshold={setAlertThreshold} 
            showAlert={showAlert} 
            currentAqi={liveData.aqi}
          />
        </div>

      </main>
    </div>
  );
};

const AIPredictionPanel = () => {
  const [input, setInput] = useState({ PM2_5: '', PM10: '', NO2: '', CO: '', SO2: '', O3: '' });
  const [prediction, setPrediction] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error("Error predicting AQI:", error);
    }
  };

  return (
    <motion.div 
      className="bg-white/10 backdrop-blur-md rounded-xl p-8"
      initial={{ opacity: 0, x: -50 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.6 }}
    >
      <h3 className="text-2xl mb-4">AI Prediction Panel</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        {Object.keys(input).map(key => (
          <input 
            key={key} 
            type="number" 
            placeholder={key} 
            className="bg-white/20 p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
            onChange={e => setInput({...input, [key]: parseFloat(e.target.value) || 0})}
          />
        ))}
      </div>
      <button 
        onClick={handlePredict} 
        className="w-full bg-blue-500 hover:bg-blue-600 p-3 rounded-md transition-colors"
      >
        Predict AQI
      </button>
      {prediction && (
        <div className="mt-4 text-center">
          <p className="text-lg">Predicted AQI: <span className="font-bold text-2xl">{prediction.predicted_aqi}</span></p>
          <p className="text-lg">Category: <span className="font-bold">{prediction.category}</span></p>
        </div>
      )}
    </motion.div>
  );
};

const AlertSystem = ({ alertThreshold, setAlertThreshold, showAlert, currentAqi }) => {
  const handleSendAlert = async () => {
    try {
      await fetch('http://127.0.0.1:5000/send-alert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ aqi: currentAqi, threshold: alertThreshold }),
      });
      alert('Alert sent!');
    } catch (error) {
      console.error("Error sending alert:", error);
    }
  };

  return (
    <motion.div 
      className="bg-white/10 backdrop-blur-md rounded-xl p-8"
      initial={{ opacity: 0, x: 50 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.8 }}
    >
      <h3 className="text-2xl mb-4">Alert System</h3>
      <div className="flex items-center mb-4">
        <label htmlFor="threshold" className="mr-4">AQI Threshold:</label>
        <input 
          id="threshold" 
          type="number" 
          value={alertThreshold} 
          onChange={e => setAlertThreshold(parseInt(e.target.value, 10))}
          className="bg-white/20 p-2 rounded-md w-24 focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>
      {showAlert && (
        <motion.div 
          className="bg-red-500/80 p-4 rounded-md text-center mb-4"
          initial={{ scale: 0.9 }} animate={{ scale: 1 }} transition={{ type: 'spring' }}
        >
          <p className="font-bold text-lg">WARNING: High AQI Detected!</p>
          <button 
            onClick={handleSendAlert} 
            className="mt-2 bg-white text-red-500 px-4 py-1 rounded-md hover:bg-gray-200 transition-colors"
          >
            Send Manual Alert
          </button>
        </motion.div>
      )}
      <div className="flex items-center justify-center">
        <p className="mr-4">Live Status:</p>
        <motion.div 
          className="w-8 h-8 rounded-full border-2 border-white"
          animate={{ backgroundColor: showAlert ? '#F44336' : '#4CAF50' }}
          transition={{ duration: 0.5 }}
        />
      </div>
    </motion.div>
  );
};

export default App;
