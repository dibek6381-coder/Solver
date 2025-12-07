import React from 'react';
import { SYSTEM_MODULES } from '../data/aimoContent';

const ArchitectureDiagram: React.FC = () => {
    // A simplified visual representation using SVG
    return (
        <div className="w-full overflow-x-auto p-8 bg-slate-900 rounded-xl shadow-2xl border border-slate-700">
            <h3 className="text-xl font-bold text-white mb-6">Pipeline Architecture</h3>
            <div className="relative min-w-[800px] h-[400px]">
                <svg className="w-full h-full">
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
                        </marker>
                    </defs>
                    
                    {/* Connections */}
                    <path d="M150 180 L250 180" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <path d="M350 180 L450 180" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <path d="M500 210 L500 300" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <path d="M550 330 L650 330" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <path d="M700 300 L700 210 M700 210 L850 180" stroke="#475569" strokeWidth="2" fill="none" markerEnd="url(#arrowhead)" />
                    <path d="M950 180 L1050 180" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)" />

                    {SYSTEM_MODULES.map((module) => (
                        <g key={module.id} transform={`translate(${module.x}, ${module.y})`}>
                            <rect 
                                x="0" y="0" 
                                width="140" height="60" 
                                rx="8" 
                                className="fill-slate-800 stroke-cyan-500 stroke-2"
                            />
                            <text x="70" y="25" textAnchor="middle" className="fill-white text-xs font-bold pointer-events-none">
                                {module.title}
                            </text>
                            <text x="70" y="45" textAnchor="middle" className="fill-slate-400 text-[10px] pointer-events-none">
                                {module.description}
                            </text>
                        </g>
                    ))}
                </svg>
            </div>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {SYSTEM_MODULES.map(m => (
                    <div key={m.id} className="bg-slate-800 p-3 rounded border border-slate-700">
                        <div className="font-semibold text-cyan-400">{m.title}</div>
                        <div className="text-xs text-slate-300">{m.description}</div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ArchitectureDiagram;