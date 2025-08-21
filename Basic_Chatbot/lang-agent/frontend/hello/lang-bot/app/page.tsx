'use client';

import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ArrowRight, MessageCircle, Sparkles, Zap, Brain } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 24,
      },
    },
  };

  const floatingVariants = {
    animate: {
      y: [-10, 10, -10],
      rotate: [0, 5, 0, -5, 0],
      transition: {
        duration: 6,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  };

  return (
    <div className="min-h-screen overflow-hidden relative">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl animate-pulse delay-2000" />
      </div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="relative z-10 container mx-auto px-4 py-20 flex flex-col items-center justify-center min-h-screen"
      >
        {/* Header */}
        <motion.div variants={itemVariants} className="text-center mb-16">
          <motion.div
            variants={floatingVariants}
            animate="animate"
            className="inline-flex items-center mb-6"
          >
            <div className="relative">
              <Brain className="w-16 h-16 text-blue-400 mr-4" />
              <motion.div
                className="absolute -top-2 -right-2 w-6 h-6 bg-cyan-400 rounded-full"
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.7, 1, 0.7],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
              />
            </div>
            <h1 className="text-6xl md:text-8xl font-bold font-space-grotesk bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Physics Chatbot
            </h1>
          </motion.div>
          
          <motion.p
            variants={itemVariants}
            className="text-xl md:text-2xl text-gray-300 max-w-2xl mx-auto leading-relaxed"
          >
            Experience the future of AI conversations. Powered by advanced language models
            for intelligent, contextual, and creative discussions.
          </motion.p>
        </motion.div>

        {/* Features Grid */}
        <motion.div variants={itemVariants} className="grid md:grid-cols-3 gap-6 mb-12 w-full max-w-4xl">
          {[
            {
              icon: <Sparkles className="w-8 h-8 text-yellow-400" />,
              title: 'Advanced AI',
              description: 'Powered by cutting-edge language models'
            },
            {
              icon: <Zap className="w-8 h-8 text-blue-400" />,
              title: 'Lightning Fast',
              description: 'Real-time responses with minimal latency'
            },
            {
              icon: <MessageCircle className="w-8 h-8 text-purple-400" />,
              title: 'Rich Conversations',
              description: 'Full markdown support with syntax highlighting'
            }
          ].map((feature, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              whileHover={{ scale: 1.05, y: -5 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            >
              <Card className="glass p-6 h-full hover:glow transition-all duration-300">
                <div className="flex items-center mb-4">
                  {feature.icon}
                  <h3 className="ml-3 text-lg font-semibold text-white">
                    {feature.title}
                  </h3>
                </div>
                <p className="text-gray-400">{feature.description}</p>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* CTA Button */}
        <motion.div variants={itemVariants} className="text-center">
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            <Button
              onClick={() => router.push('/chat')}
              size="lg"
              className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-6 px-12 text-xl rounded-2xl shadow-2xl hover:shadow-blue-500/25 transition-all duration-300 group"
            >
              Start Chatting
              <motion.div
                className="ml-3"
                animate={{ x: [0, 5, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
              </motion.div>
            </Button>
          </motion.div>
          
          <motion.p
            variants={itemVariants}
            className="text-gray-400 mt-4 text-sm"
          >
            No registration required • Free to use • Powered by AI
          </motion.p>
        </motion.div>
      </motion.div>
    </div>
  );
}