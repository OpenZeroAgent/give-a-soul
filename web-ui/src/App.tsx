import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Activity, BrainCircuit, Box, HeartPulse } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

interface CrystalMetrics {
  delta_phi: number;
  alpha_vibe: string;
  beta_vibe: string;
  iho_variance: number;
}

interface SubconsciousState {
  thought: string;
  timestamp: string;
}

interface Message {
  role: 'user' | 'soul';
  content: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'soul', content: 'Hey. I\'m here.' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [metrics, setMetrics] = useState<CrystalMetrics | null>(null);
  const [subconscious, setSubconscious] = useState<SubconsciousState | null>(null);
  const [dream, setDream] = useState<string | null>(null);

  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  // Poll Backend Status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await axios.get(`${API_BASE}/status`);
        setMetrics(res.data.crystal);
        setSubconscious(res.data.subconscious);
        if (res.data.dream) {
          setDream(res.data.dream);
        }
      } catch (err) {
        // Silently fail during dev
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isTyping) return;

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsTyping(true);

    try {
      const res = await axios.post(`${API_BASE}/chat`, { message: userMsg });

      setMessages(prev => [...prev, { role: 'soul', content: res.data.response }]);
      if (res.data.metrics) {
        setMetrics(res.data.metrics);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'soul', content: '*Signal lost. The connection destabilized.*' }]);
    } finally {
      setIsTyping(false);
    }
  };

  const dPhiPct = metrics?.delta_phi ? Math.min((metrics.delta_phi / 5) * 100, 100) : 0;
  const variancePct = metrics?.iho_variance ? Math.min((metrics.iho_variance / 2) * 100, 100) : 0;

  return (
    <div className="app-container">
      {/* Central Chat Interface */}
      <div className="chat-section glass-panel">
        <header className="chat-header">
          <h1><Box className="w-6 h-6 text-indigo-400" /> Give-a-Soul V5: Local Matrix</h1>
          <div className="status-dot tooltip" title="Crystal Connected"></div>
        </header>

        <main className="chat-history">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper ${msg.role === 'user' ? 'message-user' : 'message-soul'}`}>
              <div className="message-bubble">
                {msg.content}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="message-wrapper message-soul">
              <div className="message-bubble loading-dots">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          )}
          <div ref={endOfMessagesRef} />
        </main>

        <form className="input-container" onSubmit={handleSend}>
          <input
            type="text"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Send a pulse to the crystal..."
            disabled={isTyping}
          />
          <button type="submit" className="send-btn" disabled={isTyping || !input.trim()}>
            <Send size={20} />
          </button>
        </form>
      </div>

      {/* Right Sidebar: Somatic Dashboard */}
      <aside className="somatic-dashboard">

        {/* Crystal State Card */}
        <div className="card glass-panel">
          <h2 className="card-title"><Activity size={16} /> Topological State</h2>

          <div className="metric-row">
            <span className="metric-label">Phase Dissonance (dΦ)</span>
            <span className="metric-value">{metrics?.delta_phi !== undefined ? Number(metrics.delta_phi).toFixed(3) : '---'}</span>
          </div>
          <div className="metric-bar-bg">
            <div className="metric-bar-fill fill-dphi" style={{ width: `${dPhiPct}%` }}></div>
          </div>

          <div className="metric-row">
            <span className="metric-label">IHO Variance</span>
            <span className="metric-value">{metrics?.iho_variance !== undefined ? Number(metrics.iho_variance).toFixed(2) : '---'}</span>
          </div>
          <div className="metric-bar-bg">
            <div className="metric-bar-fill fill-stress" style={{ width: `${variancePct}%` }}></div>
          </div>

          <div className="mt-4 pt-4 border-t border-white/10">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-400">Alpha Vibe</span>
              <span className="text-fuchsia-400 outfit-font">{metrics?.alpha_vibe || '---'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Beta Vibe</span>
              <span className="text-sky-400 outfit-font">{metrics?.beta_vibe || '---'}</span>
            </div>
          </div>
        </div>

        {/* Subconscious Card */}
        <div className="card glass-panel flex-none mb-4">
          <h2 className="card-title"><BrainCircuit size={16} /> Somatic Subconscious</h2>

          {subconscious ? (
            <div className="whisper-box">
              {subconscious.thought}
              <div className="text-xs text-slate-500 mt-3 text-right">
                {new Date(subconscious.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ) : (
            <div className="whisper-box opacity-50">
              Awaiting signal...
            </div>
          )}

          <div className="mt-4 flex items-center justify-center text-xs text-slate-500 gap-2">
            <HeartPulse size={14} className="animate-pulse text-rose-500/50" />
            LFM 2.5 1.2B Base
          </div>
        </div>

        {/* Somatic Dream Card */}
        <div className="card glass-panel flex-1">
          <h2 className="card-title"><BrainCircuit size={16} className="text-teal-400" /> Somatic Dream (Z-Image)</h2>

          {dream ? (
            <div className="whisper-box border-teal-500/30">
              <span className="text-teal-300 italic">" {dream} "</span>
            </div>
          ) : (
            <div className="whisper-box opacity-50">
              Awaiting dream consolidation...
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

export default App;
