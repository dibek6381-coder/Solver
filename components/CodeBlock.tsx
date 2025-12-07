import React from 'react';

interface CodeBlockProps {
    code: string;
    language: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, language }) => {
    const copyToClipboard = () => {
        navigator.clipboard.writeText(code);
    };

    return (
        <div className="relative group h-full">
            <div className="absolute right-4 top-4 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                <button 
                    onClick={copyToClipboard}
                    className="bg-slate-700/80 hover:bg-slate-600 backdrop-blur text-white text-xs px-3 py-1.5 rounded shadow-lg border border-slate-600"
                >
                    Copy
                </button>
            </div>
            <pre className="bg-slate-950 text-slate-200 p-6 h-full overflow-auto text-sm font-mono leading-relaxed border-none outline-none">
                <code className={`language-${language}`}>{code}</code>
            </pre>
        </div>
    );
};

export default CodeBlock;