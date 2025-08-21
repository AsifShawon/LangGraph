'use client';

import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Brain, Trash2 } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface ChatHeaderProps {
  onClearChat: () => void;
  messageCount: number;
}

export function ChatHeader({ onClearChat, messageCount }: ChatHeaderProps) {
  const router = useRouter();

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="sticky top-0 z-10 bg-gradient-to-b from-slate-900 via-slate-900/95 to-transparent pb-6 mb-6"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button
            onClick={() => router.push('/')}
            variant="ghost"
            size="sm"
            className="text-gray-400 hover:text-white hover:bg-white/10"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
          
          <div className="flex items-center gap-2">
            <Brain className="w-6 h-6 text-cyan-400" />
            <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
              Physics Chatbot
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-400">
            {messageCount} messages
          </span>
          
          {messageCount > 0 && (
            <Button
              onClick={onClearChat}
              variant="ghost"
              size="sm"
              className="text-gray-400 hover:text-red-400 hover:bg-red-500/10"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear Chat
            </Button>
          )}
        </div>
      </div>
    </motion.header>
  );
}