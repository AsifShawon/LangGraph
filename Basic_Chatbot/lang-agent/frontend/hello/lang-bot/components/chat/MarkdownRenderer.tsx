// components/chat/MarkdownRenderer.tsx

import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import { Copy, Check } from 'lucide-react';
import { useState } from 'react';
import Image from 'next/image';
import 'katex/dist/katex.min.css'; // Ensure KaTeX CSS is imported

interface MarkdownRendererProps {
  content: string;
  isDarkMode?: boolean;
  className?: string;
}

interface CodeBlockProps {
  children?: React.ReactNode;
  className?: string;
  isDarkMode: boolean;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ children, className, isDarkMode }) => {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';
  const codeString = String(children).replace(/\n$/, ''); // Clean up trailing newline

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(codeString);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  // Render multi-line code blocks with syntax highlighting
  if (language) {
    return (
      <div className="relative group code-block-wrapper">
        <div className="absolute right-2 top-2 z-10">
          <button
            onClick={copyToClipboard}
            className={`p-2 rounded-md transition-all duration-200 ${
              isDarkMode
                ? 'bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-600 hover:text-gray-800'
            } opacity-0 group-hover:opacity-100`}
            title="Copy code"
          >
            {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
          </button>
        </div>
        <SyntaxHighlighter
          style={isDarkMode ? oneDark : oneLight}
          language={language}
          PreTag="div"
          className="!p-4 rounded-lg"
          wrapLines={true}
          wrapLongLines={true}
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    );
  }

  // Render inline code snippets
  return (
    <code
      className={`px-1.5 py-1 rounded-md text-sm font-mono ${
        isDarkMode
          ? 'bg-gray-800 text-cyan-300'
          : 'bg-gray-200 text-cyan-700'
      }`}
    >
      {children}
    </code>
  );
};

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ 
  content, 
  isDarkMode = true, 
  className = '' 
}) => {
  return (
    <div className={`markdown-content ${isDarkMode ? 'dark' : 'light'} ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          // Use our custom CodeBlock component for all `code` elements
          code: (props) => <CodeBlock {...props} isDarkMode={isDarkMode} />,

          // Headings
          h1: ({ node, ...props }) => <h1 className="text-2xl font-bold mt-6 mb-3 pb-2 border-b" {...props} />,
          h2: ({ node, ...props }) => <h2 className="text-xl font-bold mt-5 mb-2 pb-1 border-b" {...props} />,
          h3: ({ node, ...props }) => <h3 className="text-lg font-semibold mt-4 mb-2" {...props} />,

          // Paragraphs
          p: ({ node, ...props }) => (
            <p 
              className="mb-4 leading-relaxed" 
              style={{ whiteSpace: 'pre-wrap' }} // <-- ADD THIS LINE
              {...props} 
            />
          ),

          // Lists
          ul: ({ node, ...props }) => <ul className="mb-4 ml-5 space-y-2 list-disc" {...props} />,
          ol: ({ node, ...props }) => <ol className="mb-4 ml-5 space-y-2 list-decimal" {...props} />,
          
          // Links
          a: ({ node, ...props }) => <a className="font-medium underline decoration-2 underline-offset-2 transition-colors text-cyan-400 hover:text-cyan-300 decoration-cyan-400/50" {...props} />,

          // Blockquotes
          blockquote: ({ node, ...props }) => <blockquote className="border-l-4 pl-4 py-2 my-4 italic" {...props} />,

          // Tables
          table: ({ node, ...props }) => <div className="overflow-x-auto my-4"><table className="min-w-full border-collapse" {...props} /></div>,
          th: ({ node, ...props }) => <th className="px-4 py-2 text-left font-semibold border" {...props} />,
          td: ({ node, ...props }) => <td className="px-4 py-2 border" {...props} />,

          // Horizontal Rule
          hr: ({ node, ...props }) => <hr className="my-6 border-t-2" {...props} />,

          // Images
          img: ({ node, src, alt, ...props }) => (
            <Image
              src={src || ''}
              alt={alt || ''}
              width={800}
              height={400}
              className="max-w-full h-auto rounded-lg shadow-lg my-4"
            />
          ),

          // --- KEY CHANGES FOR MATH RENDERING ---

          // FIX 1: Remove special styling for inline math (`math-inline`).
          // KaTeX will handle this correctly by default, letting it blend with the text.
          span: ({ node, className, ...props }) => {
            if (className === 'math math-inline') {
              return <span className="math-inline" {...props} />;
            }
            return <span className={className} {...props} />;
          },

          // FIX 2: Add a styled wrapper for block math (`math-display`).
          // This creates a dedicated, clean-looking block for equations.
          div: ({ node, className, ...props }) => {
            if (className === 'math math-display') {
              return <div className="math-display-wrapper"><div className={className} {...props} /></div>;
            }
            return <div className={className} {...props} />;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;