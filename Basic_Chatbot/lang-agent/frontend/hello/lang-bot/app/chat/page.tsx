// app/chat/page.tsx

"use client";

import React from 'react';
import { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Sparkles, 
  Menu, 
  X, 
  Settings, 
  History, 
  BookOpen, 
  User, 
  Bot, 
  Send, 
  Mic,
  MicOff,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  MessageCircle,
  Trash2,
  Edit3,
  Plus,
  Sun,
  Moon,
  ChevronDown,
  Clock
} from 'lucide-react';
import MarkdownRenderer from '../../components/chat/MarkdownRenderer';

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  isTyping?: boolean;
  thinkingContent?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  timestamp: Date;
}

interface BackendMessage {
  type: string;
  content: string;
}

interface ConversationHistory {
  thread_id: string;
  messages: BackendMessage[];
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [openAccordionId, setOpenAccordionId] = useState<string | null>(null);
  const [availableChats, setAvailableChats] = useState<string[]>([]);
  const [isLoadingChats, setIsLoadingChats] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);


  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Fetch available chat threads
  const fetchAvailableChats = useCallback(async () => {
    setIsLoadingChats(true);
    try {
      const response = await fetch('/api/chats');
      if (response.ok) {
        const chats = await response.json();
        setAvailableChats(chats);
      } else {
        console.error('Failed to fetch chats');
      }
    } catch (error) {
      console.error('Error fetching chats:', error);
    } finally {
      setIsLoadingChats(false);
    }
  }, []);

  // Load available chats when sidebar opens
  useEffect(() => {
    if (sidebarOpen && availableChats.length === 0) {
      fetchAvailableChats();
    }
  }, [sidebarOpen, availableChats.length, fetchAvailableChats]);

  // Load conversation history for a specific thread
  const loadConversationHistory = async (threadId: string) => {
    setIsLoadingHistory(true);
    try {
      const response = await fetch(`/api/chats/${threadId}`);
      if (response.ok) {
        const data: ConversationHistory = await response.json();
        const convertedMessages: Message[] = data.messages.map((msg, index) => ({
          id: `${threadId}-${index}`,
          content: msg.content,
          isUser: msg.type === 'HumanMessage',
          timestamp: new Date(),
          isTyping: false,
        }));
        setMessages(convertedMessages);
        setActiveThreadId(threadId);
        setSidebarOpen(false); // Close sidebar on mobile after selecting
      } else {
        console.error('Failed to load conversation history');
      }
    } catch (error) {
      console.error('Error loading conversation:', error);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  // Switch to a different chat thread
  const switchToChat = (threadId: string) => {
    if (threadId === activeThreadId) return;
    loadConversationHistory(threadId);
  };

  // Generate a title for a chat based on first message
  const generateChatTitle = (threadId: string): string => {
    const chatData = availableChats.find(id => id === threadId);
    if (!chatData) return `Chat ${threadId.slice(-8)}`;
    return `Chat ${threadId.slice(-8)}`;
  };

  const handleSendMessage = async (content?: string) => {
    const messageContent = content || inputValue.trim();
    if (!messageContent || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: messageContent,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    const aiMessageId = (Date.now() + 1).toString();
    const aiMessage: Message = {
      id: aiMessageId,
      content: '',
      isUser: false,
      timestamp: new Date(),
      isTyping: true,
      thinkingContent: '',
    };
    setMessages(prev => [...prev, aiMessage]);
    
    try {
      const requestBody = {
        message: messageContent,
        thread_id: activeThreadId,
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      // console.log('Response:', response);

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; 

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.substring(6);
            const data = JSON.parse(jsonStr);

            if (data.type === 'thread_id') {
              setActiveThreadId(data.thread_id);
              // If this is a new thread, refresh the chat list
              if (!activeThreadId) {
                fetchAvailableChats();
              }
            } else if (data.type === 'thinking') {
              setMessages(prev => prev.map(msg => 
                msg.id === aiMessageId 
                  ? { ...msg, thinkingContent: data.content } 
                  : msg
              ));
            } else if (data.type === 'answer') {
               setMessages(prev => prev.map(msg => 
                msg.id === aiMessageId 
                  ? { ...msg, content: msg.content + data.content, isTyping: true } 
                  : msg
              ));
            }
          }
        }
      }

    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = 'Sorry, there was an error. Please ensure the backend server is running and accessible.';
      setMessages(prev => prev.map(msg => 
        msg.id === aiMessageId 
          ? { ...msg, content: errorMessage, isTyping: false }
          : msg
      ));
    } finally {
      setIsLoading(false);
      setMessages(prev => prev.map(msg => 
        msg.id === aiMessageId 
          ? { ...msg, isTyping: false } 
          : msg
      ));
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleListening = () => setIsListening(!isListening);
  const copyMessage = (content: string) => navigator.clipboard.writeText(content).catch(err => console.error('Failed to copy:', err));
  
  const handleClearChat = () => {
    setMessages([]);
    setActiveThreadId(null); 
    // Refresh available chats list
    fetchAvailableChats();
  };
  
  const createNewSession = () => {
    setMessages([]);
    setActiveThreadId(null); 
    setSidebarOpen(false); // Close sidebar on mobile
  };
  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  const themeClasses = isDarkMode ? 'bg-gray-950 text-white' : 'bg-gray-50 text-gray-900';
  const cardClasses = isDarkMode ? 'bg-gray-900/50 border-gray-800' : 'bg-white/50 border-gray-200';
  const inputClasses = isDarkMode ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400' : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500';

  return (
    <div className={`h-screen flex overflow-hidden transition-colors duration-300 ${themeClasses}`}>
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-gradient-to-r from-purple-500/15 to-pink-500/15 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }} />
      </div>

      {sidebarOpen && (
        <>
          <div 
            className="fixed inset-0 bg-black/50 z-40 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="fixed top-0 left-0 bottom-0 w-80 z-50 bg-gray-900/95 backdrop-blur-xl border-r border-gray-800 flex flex-col">
            <div className="p-6 border-b border-gray-800">
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <h2 className="text-xl font-bold text-white">Physics Chatbot</h2>
                </div>
                <button 
                  onClick={() => setSidebarOpen(false)}
                  className="text-gray-400 hover:text-white p-2 rounded-lg hover:bg-gray-800 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <button
                onClick={createNewSession}
                className="w-full flex items-center justify-center gap-2 p-3 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-colors"
              >
                <Plus className="w-4 h-4" />
                New Chat
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Chat History</h3>
                <button
                  onClick={fetchAvailableChats}
                  className="p-1 text-gray-400 hover:text-white transition-colors"
                  disabled={isLoadingChats}
                >
                  <RefreshCw className={`w-4 h-4 ${isLoadingChats ? 'animate-spin' : ''}`} />
                </button>
              </div>
              
              <div className="space-y-2">
                {isLoadingChats ? (
                  <div className="space-y-2">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="p-3 bg-gray-800/50 rounded-lg animate-pulse">
                        <div className="h-4 bg-gray-700 rounded w-3/4 mb-2"></div>
                        <div className="h-3 bg-gray-700 rounded w-1/2"></div>
                      </div>
                    ))}
                  </div>
                ) : availableChats.length === 0 ? (
                  <div className="text-center text-gray-500 py-8">
                    <MessageCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No chat history yet</p>
                    <p className="text-xs">Start a conversation to see your chats here</p>
                  </div>
                ) : (
                  availableChats.map((threadId) => (
                    <button
                      key={threadId}
                      onClick={() => switchToChat(threadId)}
                      disabled={isLoadingHistory}
                      className={`w-full text-left p-3 rounded-lg transition-colors group relative ${
                        activeThreadId === threadId
                          ? 'bg-cyan-600/20 border border-cyan-600/30 text-cyan-300'
                          : 'bg-gray-800/30 hover:bg-gray-800/50 text-gray-300 hover:text-white'
                      } ${isLoadingHistory ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className="flex items-start gap-3">
                        <MessageCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-sm truncate">
                            {generateChatTitle(threadId)}
                          </div>
                          <div className="flex items-center gap-1 mt-1">
                            <Clock className="w-3 h-3 opacity-50" />
                            <span className="text-xs opacity-70">
                              Thread: {threadId.slice(-8)}...
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      {activeThreadId === threadId && (
                        <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
                          <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                        </div>
                      )}
                    </button>
                  ))
                )}
              </div>
            </div>
            
            <div className="p-4 border-t border-gray-800 space-y-3">
              <button
                onClick={toggleTheme}
                className="w-full flex items-center gap-3 p-3 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
              >
                {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                {isDarkMode ? 'Light Mode' : 'Dark Mode'}
              </button>
              
              <div className="flex items-center gap-3 p-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-white">Guest User</p>
                  <p className="text-xs text-gray-500">Free Plan</p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <div className={`flex-1 flex flex-col transition-all duration-300 ${sidebarOpen ? 'md:ml-80' : ''}`}>
        <header className={`flex-shrink-0 flex items-center justify-between p-4 border-b backdrop-blur-sm ${
          isDarkMode ? 'border-gray-800 bg-gray-900/50' : 'border-gray-200 bg-white/50'
        }`}>
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-gray-800/50 transition-colors"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold">Physics Chatbot</h1>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  {activeThreadId ? (
                    <span>
                      Thread: {activeThreadId.slice(-8)}...
                      {isLoadingHistory && (
                        <RefreshCw className="w-3 h-3 animate-spin inline ml-1" />
                      )}
                    </span>
                  ) : (
                    'Online'
                  )}
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-500 hidden sm:block">
              {messages.length} messages
            </span>
            <button 
              onClick={handleClearChat}
              className="px-3 py-2 text-sm rounded-lg transition-colors hover:bg-red-500/20 text-red-400"
            >
              Clear
            </button>
          </div>
        </header>
        
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full p-8">
              <div className={`text-center max-w-md p-8 rounded-2xl backdrop-blur-sm ${cardClasses}`}>
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                  <Sparkles className="w-8 h-8 text-white animate-pulse" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Welcome to Physics Chatbot</h3>
                <p className="text-gray-500 mb-6">
                  Start a conversation with AI. Ask anything, get instant responses with markdown support!
                </p>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <button 
                    onClick={() => setInputValue("What is physics?")}
                    className="p-3 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 transition-colors text-left"
                  >
                    💡 Get started
                  </button>
                  <button 
                    onClick={() => setInputValue("Give me a example of quantum physics")}
                    className="p-3 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 transition-colors text-left"
                  >
                     Quantum Physics
                  </button>
                  <button 
                    onClick={() => setInputValue("How does the sun give us light?")}
                    className="p-3 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 transition-colors text-left"
                  >
                    🌌 How does the sun give us light?
                  </button>
                  <button 
                    onClick={() => setInputValue("Explain speed with math examples")}
                    className="p-3 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 transition-colors text-left"
                  >
                    ✍️ A math demo on speed
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto p-6 space-y-6">
              {messages.map((message, index) => (
                <div
                  key={message.id}
                  className={`flex gap-4 ${message.isUser ? 'justify-end' : 'justify-start'}`}
                  style={{
                    animation: `fadeInUp 0.3s ease-out ${index * 0.1}s both`
                  } as React.CSSProperties}
                >
                  {!message.isUser && (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center flex-shrink-0">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                  )}
                  
                  <div className={`max-w-[80%] ${message.isUser ? 'order-first' : ''}`}>
                    <div
                      className={`p-4 rounded-2xl backdrop-blur-sm relative group ${
                        message.isUser
                          ? 'bg-gradient-to-br from-cyan-500 to-blue-600 text-white ml-auto'
                          : `${cardClasses}`
                      }`}
                    >
                       {!message.isUser && message.thinkingContent && (
                        <div className="mb-4">
                          <button 
                            onClick={() => setOpenAccordionId(openAccordionId === message.id ? null : message.id)}
                            className="flex items-center justify-between w-full text-left text-sm font-medium text-gray-400 hover:text-white"
                          >
                            <span>Thinking Process</span>
                            <ChevronDown className={`w-4 h-4 transition-transform ${openAccordionId === message.id ? 'rotate-180' : ''}`} />
                          </button>
                          {openAccordionId === message.id && (
                            <div className="mt-2 p-3 bg-gray-800/50 rounded-lg">
                               <MarkdownRenderer 
                                content={message.thinkingContent} 
                                isDarkMode={isDarkMode}
                              />
                            </div>
                          )}
                        </div>
                      )}

                      {message.isUser ? (
                        <div className="whitespace-pre-wrap break-words">
                          {message.content}
                        </div>
                      ) : (
                        <div className={`prose prose-sm max-w-none ${isDarkMode ? 'dark' : 'light'}`}>
                          <MarkdownRenderer 
                            content={message.content + (message.isTyping ? '...' : '')} 
                            isDarkMode={isDarkMode}
                          />
                        </div>
                      )}
                      
                      {!message.isUser && !message.isTyping && (
                        <div className="flex items-center gap-2 mt-3 pt-3 border-t border-gray-700/50 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={() => copyMessage(message.content)}
                            className="p-1 rounded hover:bg-gray-700/50 transition-colors"
                          >
                            <Copy className="w-3 h-3" />
                          </button>
                          <button className="p-1 rounded hover:bg-gray-700/50 transition-colors">
                            <ThumbsUp className="w-3 h-3" />
                          </button>
                          <button className="p-1 rounded hover:bg-gray-700/50 transition-colors">
                            <ThumbsDown className="w-3 h-3" />
                          </button>
                          <button className="p-1 rounded hover:bg-gray-700/50 transition-colors">
                            <RefreshCw className="w-3 h-3" />
                          </button>
                        </div>
                      )}
                    </div>
                    
                    <div className={`text-xs text-gray-500 mt-2 ${message.isUser ? 'text-right' : 'text-left'}`}>
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                  
                  {message.isUser && (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center flex-shrink-0">
                      <User className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
              ))}
              
              {isLoading && messages[messages.length -1]?.isTyping && (
                <div className="flex gap-4 justify-start">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <div className={`p-4 rounded-2xl ${cardClasses}`}>
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
        
        <div className={`flex-shrink-0 p-4 border-t backdrop-blur-sm ${
          isDarkMode ? 'border-gray-800 bg-gray-900/50' : 'border-gray-200 bg-white/50'
        }`}>
          <div className="max-w-4xl mx-auto">
            <div className={`flex items-end gap-3 p-3 rounded-2xl border transition-all ${inputClasses}`}>
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="flex-1 bg-transparent border-none outline-none resize-none min-h-[20px] max-h-[120px]"
                rows={1}
                disabled={isLoading}
              />
              
              <div className="flex items-center gap-2">
                <button
                  onClick={toggleListening}
                  className={`p-2 rounded-lg transition-colors ${
                    isListening 
                      ? 'bg-red-500 text-white animate-pulse' 
                      : 'hover:bg-gray-700 text-gray-400'
                  }`}
                >
                </button>
                
                <button
                  onClick={() => handleSendMessage()}
                  disabled={!inputValue.trim() || isLoading}
                  className={`p-2 rounded-lg transition-all ${
                    inputValue.trim() && !isLoading
                      ? 'bg-cyan-600 hover:bg-cyan-700 text-white transform hover:scale-105'
                      : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  {isLoading ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
            
            <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
              <span>Press Enter to send, Shift + Enter for new line</span>
              <span>{inputValue.length}/2000</span>
            </div>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
