import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area, CartesianGrid 
} from 'recharts';
import { 
  Wind, Thermometer, Droplets, AlertTriangle, 
  ChevronRight, Brain, Bell, LogOut, User, ShieldAlert, Activity, Mail
} from 'lucide-react';
import { auth, googleProvider } from './firebase';
import { 
  signInWithPopup, 
  signOut, 
  onAuthStateChanged,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  updateProfile
} from 'firebase/auth';

const App = () => {
  const [user, setUser] = useState(null);
  const [liveData, setLiveData] = useState(null);
  const [aqiHistory, setAqiHistory] = useState([]);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [loading, setLoading] = useState(true);

  // Auth Listener
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        // Sync user with backend
        fetch('http://127.0.0.1:5000/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            uid: currentUser.uid,
            name: currentUser.displayName || currentUser.email.split('@')[0],
            email: currentUser.email
          })
        });
      } else {
        setUser(null);
      }
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // Data Fetching
  useEffect(() => {
    if (!user) return;

    const fetchData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/live-data');
        const data = await response.json();
        setLiveData(data);
        
        setAqiHistory(prev => {
          const now = new Date();
          const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          return [...prev.slice(-29), { name: timeStr, aqi: data.aqi }];
        });
      } catch (error) {
        console.error("Fetch error:", error);
      }
    };

    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, [user]);

  const handleGoogleLogin = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (error) {
      alert("Google login failed. Please check your Firebase config in firebase.js: " + error.message);
    }
  };

  const handleEmailLogin = async (email, password) => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (error) {
      alert("Login failed: " + error.message);
    }
  };

  const handleEmailSignup = async (name, email, password) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(userCredential.user, { displayName: name });
      // Force user state update
      setUser({ ...userCredential.user, displayName: name });
    } catch (error) {
      alert("Signup failed: " + error.message);
    }
  };

  const handleLogout = () => signOut(auth);

  const toggleAlerts = async () => {
    const newState = !alertsEnabled;
    setAlertsEnabled(newState);
    try {
      await fetch('http://127.0.0.1:5000/toggle-alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid: user.uid, enabled: newState })
      });
    } catch (error) {
      console.error("Toggle alerts error:", error);
    }
  };

  if (loading) return <LoadingScreen />;
  if (!user) return (
    <LoginScreen 
      onGoogleLogin={handleGoogleLogin} 
      onEmailLogin={handleEmailLogin}
      onEmailSignup={handleEmailSignup}
    />
  );

  return (
    <div className="min-h-screen bg-[#05070a] text-zinc-100 font-sans overflow-x-hidden selection:bg-emerald-500/30">
      <AnimatedBackground aqi={liveData?.aqi || 0} />
      
      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 px-8 py-4 bg-black/20 backdrop-blur-xl border-b border-white/5 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Wind className="text-black w-6 h-6" />
          </div>
          <span className="text-2xl font-black tracking-tighter">AERO<span className="text-emerald-500">GUARD</span></span>
        </div>

        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-3 px-4 py-2 bg-white/5 rounded-full border border-white/10">
            <img src={user.photoURL} alt="avatar" className="w-8 h-8 rounded-full border border-emerald-500/50" />
            <span className="text-sm font-bold">{user.displayName}</span>
          </div>
          <button onClick={handleLogout} className="p-2.5 hover:bg-red-500/10 hover:text-red-500 rounded-full transition-all border border-white/5">
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </nav>

      <main className="pt-32 pb-20 px-8 max-w-[1600px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
        
        {/* Main AQI Gauge */}
        <section className="lg:col-span-8 space-y-8">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative overflow-hidden bg-white/5 backdrop-blur-3xl border border-white/10 rounded-[40px] p-12 shadow-2xl"
          >
            <div className="flex justify-between items-start">
              <div className="space-y-4">
                <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-emerald-400 text-xs font-black tracking-widest uppercase">
                  <Activity className="w-3 h-3 animate-pulse" /> Live Sensor Feed
                </div>
                <h1 className="text-6xl font-black tracking-tight">Air Quality <br/>Index</h1>
              </div>
              <div className="text-right">
                <div className={`text-8xl font-black ${getAqiTextColor(liveData?.aqi)}`}>
                  {liveData?.aqi || 0}
                </div>
                <div className="text-zinc-500 font-bold uppercase tracking-widest text-sm mt-2">Predicted AQI</div>
              </div>
            </div>

            <div className="mt-16 grid grid-cols-3 gap-8">
              <SensorCard icon={Wind} label="Gas (MQ135)" value={liveData?.gas_mq || 0} unit="ppm" color="emerald" />
              <SensorCard icon={Thermometer} label="Temperature" value={liveData?.temperature || 0} unit="°C" color="orange" />
              <SensorCard icon={Droplets} label="Humidity" value={liveData?.humidity || 0} unit="%" color="blue" />
            </div>

            <div className="mt-12">
              <div className="flex justify-between text-xs font-black uppercase tracking-tighter mb-4 text-zinc-500">
                <span>Good</span>
                <span>Moderate</span>
                <span>Unhealthy</span>
                <span>Severe</span>
                <span>Hazardous</span>
              </div>
              <div className="h-4 w-full bg-white/5 rounded-full overflow-hidden p-1 border border-white/5">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min((liveData?.aqi / 350) * 100, 100)}%` }}
                  className={`h-full rounded-full bg-gradient-to-r from-emerald-500 via-yellow-400 to-red-600 shadow-lg`}
                />
              </div>
            </div>
          </motion.div>

          {/* Trend Chart */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white/5 backdrop-blur-3xl border border-white/10 rounded-[40px] p-10"
          >
            <div className="flex justify-between items-center mb-10">
              <h3 className="text-2xl font-black">Historical Trend</h3>
              <div className="flex gap-2">
                <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Real-time Sync</span>
              </div>
            </div>
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={aqiHistory}>
                  <defs>
                    <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="name" stroke="#52525b" fontSize={10} axisLine={false} tickLine={false} dy={10} />
                  <YAxis stroke="#52525b" fontSize={10} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f1115', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px' }}
                    itemStyle={{ color: '#10b981', fontWeight: 'bold' }}
                  />
                  <Area type="monotone" dataKey="aqi" stroke="#10b981" strokeWidth={4} fill="url(#chartGradient)" animationDuration={1500} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        </section>

        {/* Sidebar */}
        <aside className="lg:col-span-4 space-y-8">
          {/* Alert Panel */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white/5 backdrop-blur-3xl border border-white/10 rounded-[40px] p-8"
          >
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <Mail className="w-6 h-6 text-emerald-400" />
                <h3 className="text-xl font-black uppercase tracking-tighter">Email Alerts</h3>
              </div>
              <button 
                onClick={toggleAlerts}
                className={`w-14 h-8 rounded-full p-1 transition-all ${alertsEnabled ? 'bg-emerald-500' : 'bg-zinc-700'}`}
              >
                <motion.div 
                  animate={{ x: alertsEnabled ? 24 : 0 }}
                  className="w-6 h-6 bg-white rounded-full shadow-lg"
                />
              </button>
            </div>
            <p className="text-sm text-zinc-500 font-medium leading-relaxed mb-6">
              Get instant notifications at <span className="text-zinc-300 font-bold">{user.email}</span> when air quality becomes hazardous.
            </p>
            <div className="space-y-3">
              <ThresholdIndicator label="Moderate" value="100" active={liveData?.aqi > 100} />
              <ThresholdIndicator label="Unhealthy" value="150" active={liveData?.aqi > 150} />
              <ThresholdIndicator label="Severe" value="250" active={liveData?.aqi > 250} />
            </div>
          </motion.div>

          {/* AI Predictor Form */}
          <AIPredictorCard />

          {/* Health Status */}
          <motion.div className="bg-gradient-to-br from-emerald-500/10 to-blue-500/10 border border-white/10 rounded-[40px] p-8">
            <h3 className="text-xl font-black mb-6 uppercase tracking-tighter">Health Advisory</h3>
            <div className="flex gap-4">
              <div className="w-12 h-12 bg-white/5 rounded-2xl flex items-center justify-center shrink-0">
                <ShieldAlert className="w-6 h-6 text-emerald-400" />
              </div>
              <p className="text-sm text-zinc-400 leading-relaxed italic">
                "{getHealthAdvice(liveData?.aqi)}"
              </p>
            </div>
          </motion.div>
        </aside>
      </main>

      {/* Floating Particles */}
      <Particles count={20} aqi={liveData?.aqi || 0} />
    </div>
  );
};

// Helper Components
const SensorCard = ({ icon: Icon, label, value, unit, color }) => (
  <div className="bg-white/5 border border-white/5 p-6 rounded-[32px] hover:bg-white/10 transition-all group">
    <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 bg-${color}-500/10 text-${color}-400 group-hover:scale-110 transition-transform`}>
      <Icon size={20} />
    </div>
    <div className="text-2xl font-black tracking-tighter">{value}<span className="text-xs text-zinc-500 ml-1 font-bold">{unit}</span></div>
    <div className="text-[10px] text-zinc-500 font-black uppercase tracking-widest mt-1">{label}</div>
  </div>
);

const ThresholdIndicator = ({ label, value, active }) => (
  <div className={`flex justify-between items-center p-4 rounded-2xl border transition-all ${active ? 'bg-red-500/10 border-red-500/20' : 'bg-white/5 border-white/5'}`}>
    <span className={`text-xs font-black uppercase ${active ? 'text-red-400' : 'text-zinc-500'}`}>{label} Alert</span>
    <span className={`text-xs font-bold ${active ? 'text-red-400' : 'text-zinc-600'}`}>{value} AQI</span>
  </div>
);

const AIPredictorCard = () => {
  const [input, setInput] = useState({ gas: '', temp: '', humidity: '' });
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
      });
      setRes(await response.json());
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div className="bg-white/5 border border-white/10 rounded-[40px] p-8">
      <div className="flex items-center gap-3 mb-8">
        <Brain className="w-6 h-6 text-purple-400" />
        <h3 className="text-xl font-black uppercase tracking-tighter">AI Forecaster</h3>
      </div>
      <div className="space-y-4">
        {['gas', 'temp', 'humidity'].map(k => (
          <input 
            key={k}
            type="number" 
            placeholder={`Enter ${k}...`}
            className="w-full bg-white/5 border border-white/5 p-4 rounded-2xl text-sm font-bold focus:outline-none focus:border-emerald-500/50 transition-all"
            onChange={e => setInput({...input, [k]: e.target.value})}
          />
        ))}
        <button 
          onClick={handlePredict}
          className="w-full bg-emerald-500 text-black font-black py-4 rounded-2xl hover:bg-emerald-400 transition-all flex items-center justify-center gap-2"
        >
          {loading ? "Processing..." : "Run Forecast"} <ChevronRight size={18} />
        </button>
      </div>
      {res && (
        <div className="mt-6 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl text-center">
          <div className="text-3xl font-black text-emerald-400">{res.predicted_aqi}</div>
          <div className="text-[10px] font-black uppercase text-zinc-500 tracking-widest">{res.category}</div>
        </div>
      )}
    </motion.div>
  );
};

const AnimatedBackground = ({ aqi }) => {
  const getGradient = () => {
    if (aqi <= 50) return 'from-emerald-950/20 to-[#05070a]';
    if (aqi <= 150) return 'from-yellow-900/10 to-[#05070a]';
    return 'from-red-950/20 to-[#05070a]';
  };
  return <div className={`fixed inset-0 bg-gradient-to-b ${getGradient()} transition-all duration-1000`} />;
};

const Particles = ({ count, aqi }) => (
  <div className="fixed inset-0 pointer-events-none z-0">
    {[...Array(count)].map((_, i) => (
      <motion.div
        key={i}
        className={`absolute rounded-full blur-xl ${aqi > 150 ? 'bg-red-500/5' : 'bg-emerald-500/5'}`}
        initial={{ x: Math.random() * 100 + "%", y: Math.random() * 100 + "%", width: 200, height: 200 }}
        animate={{ y: ["0%", "100%"], x: [null, (Math.random() * 20 - 10) + "%"] }}
        transition={{ duration: 20 + Math.random() * 20, repeat: Infinity, ease: "linear" }}
      />
    ))}
  </div>
);

const LoginScreen = ({ onGoogleLogin, onEmailLogin, onEmailSignup }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ name: '', email: '', password: '' });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isLogin) {
      onEmailLogin(formData.email, formData.password);
    } else {
      onEmailSignup(formData.name, formData.email, formData.password);
    }
  };

  return (
    <div className="min-h-screen bg-[#05070a] flex items-center justify-center p-8 overflow-hidden relative">
      <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 via-transparent to-blue-500/10 blur-3xl" />
      
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-md w-full bg-white/5 backdrop-blur-3xl border border-white/10 rounded-[48px] p-10 text-center relative z-10 shadow-2xl"
      >
        <div className="w-16 h-16 bg-emerald-500 rounded-[20px] flex items-center justify-center mx-auto mb-8 shadow-xl shadow-emerald-500/20">
          <Wind className="text-black w-8 h-8" />
        </div>
        
        <h1 className="text-3xl font-black mb-2 tracking-tighter">AERO<span className="text-emerald-500">GUARD</span></h1>
        <p className="text-zinc-500 text-sm font-medium mb-8">Access your environmental intelligence dashboard.</p>

        <form onSubmit={handleSubmit} className="space-y-4 text-left">
          {!isLogin && (
            <div className="space-y-2">
              <label className="text-xs font-black uppercase text-zinc-500 ml-2">Full Name</label>
              <input 
                type="text" 
                required
                placeholder="John Doe"
                className="w-full bg-white/5 border border-white/5 p-4 rounded-2xl text-sm font-bold focus:outline-none focus:border-emerald-500/50 transition-all"
                onChange={e => setFormData({...formData, name: e.target.value})}
              />
            </div>
          )}
          <div className="space-y-2">
            <label className="text-xs font-black uppercase text-zinc-500 ml-2">Gmail Address</label>
            <input 
              type="email" 
              required
              placeholder="name@gmail.com"
              className="w-full bg-white/5 border border-white/5 p-4 rounded-2xl text-sm font-bold focus:outline-none focus:border-emerald-500/50 transition-all"
              onChange={e => setFormData({...formData, email: e.target.value})}
            />
          </div>
          <div className="space-y-2">
            <label className="text-xs font-black uppercase text-zinc-500 ml-2">Password</label>
            <input 
              type="password" 
              required
              placeholder="••••••••"
              className="w-full bg-white/5 border border-white/5 p-4 rounded-2xl text-sm font-bold focus:outline-none focus:border-emerald-500/50 transition-all"
              onChange={e => setFormData({...formData, password: e.target.value})}
            />
          </div>

          <button 
            type="submit"
            className="w-full bg-emerald-500 text-black font-black py-4 rounded-2xl hover:bg-emerald-400 transition-all shadow-lg mt-4"
          >
            {isLogin ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <div className="my-8 flex items-center gap-4 text-zinc-600">
          <div className="h-px bg-white/5 flex-1" />
          <span className="text-[10px] font-black uppercase tracking-widest">OR</span>
          <div className="h-px bg-white/5 flex-1" />
        </div>

        <button 
          onClick={onGoogleLogin}
          className="w-full bg-white text-black font-black py-4 rounded-2xl hover:bg-zinc-200 transition-all flex items-center justify-center gap-3 shadow-xl mb-6"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-1 .67-2.28 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" />
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.66l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
          </svg>
          Continue with Google
        </button>

        <button 
          onClick={() => setIsLogin(!isLogin)}
          className="text-xs font-bold text-zinc-500 hover:text-emerald-400 transition-colors"
        >
          {isLogin ? "Don't have an account? Sign Up" : "Already have an account? Sign In"}
        </button>
      </motion.div>
    </div>
  );
};

const LoadingScreen = () => (
  <div className="min-h-screen bg-[#05070a] flex items-center justify-center">
    <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }} className="w-12 h-12 border-4 border-emerald-500 border-t-transparent rounded-full" />
  </div>
);

// Utility functions
const getAqiTextColor = (aqi) => {
  if (aqi <= 50) return 'text-emerald-500';
  if (aqi <= 100) return 'text-yellow-400';
  if (aqi <= 150) return 'text-orange-500';
  return 'text-red-500';
};

const getHealthAdvice = (aqi) => {
  if (aqi <= 50) return "Air quality is great! Perfect day for outdoor activities.";
  if (aqi <= 100) return "Moderate air quality. Sensitive individuals should limit prolonged exertion.";
  if (aqi <= 150) return "Unhealthy for sensitive groups. Consider staying indoors.";
  return "Hazardous conditions. Keep windows closed and avoid all outdoor physical activity.";
};

export default App;
