import React, { useState } from 'react';
import { GoogleGenAI } from '@google/genai';

interface GeminiAssistantProps {
    contextCode?: string;
}

const GeminiAssistant: React.FC<GeminiAssistantProps> = ({ contextCode }) => {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [apiKey, setApiKey] = useState('');
    const [showKeyInput, setShowKeyInput] = useState(false);

    const handleAsk = async () => {
        if (!query) return;
        
        const key = process.env.API_KEY || apiKey;
        
        if (!key) {
            setShowKeyInput(true);
            return;
        }

        setLoading(true);
        try {
            const ai = new GoogleGenAI({ apiKey: key });
            
            const prompt = `
                You are an expert AI engineer for Kaggle math competitions.
                Analyze the following code context and answer the user's question.
                
                Code Context: 
                ${contextCode ? contextCode.substring(0, 3000) : 'No code selected.'}
                
                Question: ${query}
                
                Provide a concise, technical answer explaining the logic or suggesting improvements.
            `;

            const result = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt
            });
            
            setResponse(result.text || 'No response generated.');
        } catch (error) {
            setResponse('Error: Could not connect to Gemini. Please check your API Key.');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 h-full flex flex-col">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <span className="text-2xl">✨</span> AI Copilot
            </h3>
            
            <div className="flex-1 overflow-y-auto mb-4 bg-slate-900 rounded p-3 min-h-[200px] text-sm text-slate-300 whitespace-pre-wrap font-mono">
                {response || "Select a file and ask questions about the implementation strategies."}
            </div>

            {showKeyInput && !process.env.API_KEY && (
                <div className="mb-2">
                    <input 
                        type="password" 
                        placeholder="Enter Gemini API Key"
                        className="w-full bg-slate-700 text-white px-3 py-2 rounded text-xs border border-slate-600 focus:border-cyan-500 outline-none"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                    />
                </div>
            )}

            <div className="flex gap-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. Explain the sandbox timeout logic"
                    className="flex-1 bg-slate-700 text-white px-3 py-2 rounded text-sm border border-slate-600 focus:border-cyan-500 outline-none"
                    onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
                />
                <button
                    onClick={handleAsk}
                    disabled={loading}
                    className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded text-sm font-semibold disabled:opacity-50 transition-colors"
                >
                    {loading ? '...' : 'Ask'}
                </button>
            </div>
        </div>
    );
};

export default GeminiAssistant;