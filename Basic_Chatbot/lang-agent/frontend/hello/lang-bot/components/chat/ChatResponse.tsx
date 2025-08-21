import React from 'react';
import MarkdownRenderer from '../chat/MarkdownRenderer';
import { useAgentStream } from '../../hooks/useAgentStream';

interface ChatResponseProps {
  message: string;
  threadId?: string;
  isDarkMode?: boolean;
}

const ChatResponse: React.FC<ChatResponseProps> = ({ message, threadId, isDarkMode = true }) => {
  const { thinking, answer, error } = useAgentStream({ message, threadId });

  return (
    <div>
      {thinking && (
        <div className="mb-4 p-4 bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 rounded animate-pulse">
          <h4 className="font-semibold mb-2">Agent is thinking...</h4>
          <MarkdownRenderer content={thinking} isDarkMode={isDarkMode} />
        </div>
      )}
      {answer && (
        <div className="mb-4 p-4 bg-cyan-50 border-l-4 border-cyan-400 text-cyan-800 rounded">
          <h4 className="font-semibold mb-2">Final Answer</h4>
          <MarkdownRenderer content={answer} isDarkMode={isDarkMode} />
        </div>
      )}
      {error && (
        <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-400 text-red-800 rounded">
          <h4 className="font-semibold mb-2">Error</h4>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default ChatResponse;
