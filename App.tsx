
import React, { useState } from 'react';
import { View, CodeFile } from './types';
import { CODE_FILES, TRAINING_PLAN, BENCHMARK_DATA } from './data/aimoContent';
import ArchitectureDiagram from './components/ArchitectureDiagram';
import CodeBlock from './components/CodeBlock';
import GeminiAssistant from './components/GeminiAssistant';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<View>('design');
  const [selectedFile, setSelectedFile] = useState<CodeFile>(CODE_FILES[0]);

  const getFileIcon = (name: string) => {
    if (name.endsWith('.py')) return '🐍';
    if (name.endsWith('.md')) return '📝';
    if (name.endsWith('.yaml')) return '⚙️';
    return '📄';
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-white flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-br from-cyan-600 to-blue-700 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-900/20">
                    <span className="font-bold text-white text-lg">A</span>
                </div>
                <h1 className="text-lg font-bold tracking-tight text-slate-100">AIMO System <span className="text-cyan-500">Architect</span></h1>
            </div>
            <nav className="flex gap-1 bg-slate-800/50 p-1 rounded-lg border border-slate-700/50">
                {(['design', 'code', 'training', 'benchmarks'] as View[]).map((view) => (
                    <button
                        key={view}
                        onClick={() => setActiveView(view)}
                        className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                            activeView === view 
                            ? 'bg-slate-700 text-white shadow-sm ring-1 ring-slate-600' 
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                        }`}
                    >
                        {view.charAt(0).toUpperCase() + view.slice(1)}
                    </button>
                ))}
            </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        
        {/* DESIGN VIEW */}
        {activeView === 'design' && (
            <div className="space-y-8 animate-in fade-in duration-500">
                <section>
                    <div className="flex justify-between items-end mb-4">
                        <div>
                             <h2 className="text-2xl font-bold text-white">System Architecture</h2>
                             <p className="text-slate-400 mt-2 max-w-3xl">
                                High-performance, offline-capable reasoning pipeline designed for the Kaggle AIMO environment.
                                Combines LLM-driven planning with deterministic Python execution.
                            </p>
                        </div>
                        <div className="flex gap-2 text-xs font-mono text-slate-500">
                            <span className="bg-slate-900 border border-slate-700 px-2 py-1 rounded">VRAM: ~16GB</span>
                            <span className="bg-slate-900 border border-slate-700 px-2 py-1 rounded">Runtime: 5h</span>
                        </div>
                    </div>
                    
                    <ArchitectureDiagram />
                </section>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
                        <h3 className="text-lg font-bold mb-4 text-white">Execution Strategy</h3>
                        <div className="space-y-4">
                            <div className="flex gap-4 p-3 bg-slate-950/50 rounded-lg border border-slate-800/50">
                                <div className="mt-1 h-2 w-2 rounded-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.5)]"></div>
                                <div>
                                    <strong className="block text-slate-200 text-sm">Robust Preprocessing</strong>
                                    <p className="text-xs text-slate-400 mt-1">Normalizes varied LaTeX formats to ensure consistent tokenizer representation and removes AIMO-specific artifacts.</p>
                                </div>
                            </div>
                            <div className="flex gap-4 p-3 bg-slate-950/50 rounded-lg border border-slate-800/50">
                                <div className="mt-1 h-2 w-2 rounded-full bg-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]"></div>
                                <div>
                                    <strong className="block text-slate-200 text-sm">Program-of-Thought (PoT)</strong>
                                    <p className="text-xs text-slate-400 mt-1">Prioritizes executable Python over text reasoning to eliminate arithmetic hallucinations. Uses Qwen2.5-Math-7B.</p>
                                </div>
                            </div>
                            <div className="flex gap-4 p-3 bg-slate-950/50 rounded-lg border border-slate-800/50">
                                <div className="mt-1 h-2 w-2 rounded-full bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]"></div>
                                <div>
                                    <strong className="block text-slate-200 text-sm">Ensemble Verification (k=8)</strong>
                                    <p className="text-xs text-slate-400 mt-1">Generates 8 independent solution paths via temperature sampling and uses majority voting on the final integer output.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="h-[400px]">
                         <GeminiAssistant contextCode="Architecture: Hybrid PoT system with Qwen2.5-Math-7B. Key components: Preprocessor, Classifier, Solver Engine, Verifier." />
                    </div>
                </div>
            </div>
        )}

        {/* CODE VIEW - IDE STYLE */}
        {activeView === 'code' && (
            <div className="grid grid-cols-12 gap-0 h-[calc(100vh-140px)] border border-slate-800 rounded-xl overflow-hidden shadow-2xl bg-slate-900">
                {/* File Explorer */}
                <div className="col-span-2 bg-slate-900 border-r border-slate-800 flex flex-col">
                    <div className="p-3 border-b border-slate-800 text-xs font-bold text-slate-500 uppercase tracking-wider bg-slate-900/50">
                        Project Files
                    </div>
                    <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
                        {CODE_FILES.map(file => (
                            <button
                                key={file.name}
                                onClick={() => setSelectedFile(file)}
                                className={`w-full text-left px-3 py-2 rounded flex items-center gap-2 text-xs font-medium transition-colors truncate ${
                                    selectedFile.name === file.name 
                                    ? 'bg-cyan-900/30 text-cyan-400 border border-cyan-800/50' 
                                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-300'
                                }`}
                            >
                                <span className="opacity-70">{getFileIcon(file.name)}</span>
                                {file.name}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Editor Area */}
                <div className="col-span-7 flex flex-col h-full bg-slate-950 border-r border-slate-800">
                     <div className="h-9 bg-slate-900 border-b border-slate-800 flex items-center px-4 justify-between">
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-cyan-500 font-mono">{getFileIcon(selectedFile.name)}</span>
                            <span className="text-xs text-slate-300 font-mono">{selectedFile.name}</span>
                        </div>
                        <span className="text-[10px] text-slate-600 uppercase font-bold">{selectedFile.language}</span>
                     </div>
                     <div className="flex-1 overflow-hidden relative">
                        <CodeBlock code={selectedFile.content} language={selectedFile.language} />
                     </div>
                </div>

                {/* Assistant Panel */}
                <div className="col-span-3 bg-slate-900 flex flex-col">
                    <div className="p-3 border-b border-slate-800 text-xs font-bold text-slate-500 uppercase tracking-wider">
                        Copilot
                    </div>
                    <div className="flex-1 p-2 overflow-hidden">
                        <GeminiAssistant contextCode={selectedFile.content} />
                    </div>
                </div>
            </div>
        )}

        {/* TRAINING VIEW */}
        {activeView === 'training' && (
            <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                 <div className="flex items-center justify-between">
                     <div>
                        <h2 className="text-2xl font-bold text-white">Fine-Tuning Strategy</h2>
                        <p className="text-slate-400 mt-1">Adapting Qwen2.5-Math for Program-of-Thought (PoT) execution.</p>
                     </div>
                 </div>
                 
                 <div className="grid gap-6">
                    {TRAINING_PLAN.map((phase, idx) => (
                        <div key={idx} className="bg-slate-900 rounded-xl p-6 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-colors">
                            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 font-bold text-9xl text-slate-500 transition-opacity select-none">{idx + 1}</div>
                            <div className="relative z-10">
                                <div className="flex items-center gap-3 mb-2">
                                    <div className="px-2 py-1 rounded bg-cyan-900/30 text-cyan-400 text-xs font-bold uppercase border border-cyan-800/50">Phase {idx+1}</div>
                                    <h3 className="text-xl font-bold text-white">{phase.name}</h3>
                                </div>
                                <p className="text-slate-400 text-sm mb-6 max-w-2xl">{phase.description}</p>
                                
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                                        <span className="text-xs uppercase text-slate-500 font-bold tracking-wider block mb-3">Dataset</span>
                                        <div className="text-slate-300 font-mono text-xs border-l-2 border-cyan-500/50 pl-3">
                                            {phase.dataset}
                                        </div>
                                    </div>
                                    <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                                        <span className="text-xs uppercase text-slate-500 font-bold tracking-wider block mb-3">Hyperparameters</span>
                                        <div className="grid grid-cols-2 gap-y-2 gap-x-4">
                                            {Object.entries(phase.params).map(([key, val]) => (
                                                <div key={key} className="text-xs flex flex-col">
                                                    <span className="text-slate-600 mb-0.5">{key}</span>
                                                    <span className="text-cyan-300 font-mono">{val}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                 </div>
                 
                 <div className="bg-amber-950/20 border border-amber-900/50 p-4 rounded-xl flex gap-4 items-center">
                    <span className="text-2xl">⚠️</span>
                    <p className="text-sm text-amber-200/70">
                        Training 7B models requires external compute (e.g. Lambda Labs, RunPod) with H100/A100 GPUs. 
                        Kaggle notebooks are limited to T4s which are sufficient only for inference (4-bit) or very small LoRA runs.
                    </p>
                 </div>
            </div>
        )}

        {/* BENCHMARKS VIEW */}
        {activeView === 'benchmarks' && (
            <div className="animate-in fade-in duration-500">
                 <h2 className="text-2xl font-bold text-white mb-6">Performance Projections</h2>
                 <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                     <div className="lg:col-span-2 bg-slate-900 p-6 rounded-xl border border-slate-800 h-[400px]">
                        <h3 className="text-lg font-bold text-white mb-4">Pass@1 Accuracy Improvements</h3>
                        <ResponsiveContainer width="100%" height="90%">
                            <BarChart data={BENCHMARK_DATA} layout="vertical" margin={{ left: 40, right: 30, top: 10, bottom: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                                <XAxis type="number" domain={[0, 100]} stroke="#64748b" unit="%" tick={{fontSize: 12}} />
                                <YAxis 
                                    dataKey="name" 
                                    type="category" 
                                    width={150} 
                                    stroke="#94a3b8" 
                                    style={{ fontSize: '12px', fontWeight: 500 }} 
                                />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9', borderRadius: '8px' }}
                                    cursor={{fill: '#1e293b', opacity: 0.5}}
                                    formatter={(value: number) => [`${value}%`, 'Accuracy']}
                                />
                                <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={32}>
                                    {BENCHMARK_DATA.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={['#475569', '#0891b2', '#2563eb', '#7c3aed'][index]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                     </div>
                     <div className="space-y-4">
                        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
                             <div className="flex items-center justify-between mb-2">
                                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Mean Runtime</h4>
                                <span className="text-xs text-green-500 bg-green-500/10 px-2 py-0.5 rounded-full">In Budget</span>
                             </div>
                             <div className="text-3xl font-mono text-white">3m 12s</div>
                             <p className="text-xs text-slate-400 mt-1">Per problem (k=8 samples)</p>
                        </div>
                        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
                             <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">VRAM Usage</h4>
                             <div className="text-3xl font-mono text-white">~14.5 GB</div>
                             <p className="text-xs text-slate-400 mt-1">4-bit BNB Quantization (Fits T4x2)</p>
                        </div>
                        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
                             <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Est. LB Score</h4>
                             <div className="text-3xl font-mono text-cyan-400">28 / 50</div>
                             <p className="text-xs text-slate-400 mt-1">Projected Public Leaderboard</p>
                        </div>
                     </div>
                 </div>
            </div>
        )}

      </main>
    </div>
  );
};

export default App;