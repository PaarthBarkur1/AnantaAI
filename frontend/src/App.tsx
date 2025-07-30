import { useState, useEffect } from 'react'
import { ChatInput } from './components/ChatInput'
import { Answer } from './components/Answer'
import { Suggestions } from './components/Suggestions'
import { Header } from './components/Header'
import { TypingIndicator } from './components/LoadingSkeleton'
import { ErrorMessage } from './components/ErrorMessage'
import { useQA } from './hooks/useQA'

function App() {
  const [query, setQuery] = useState('')
  const [conversationHistory, setConversationHistory] = useState<Array<{query: string, answer: any}>>([])
  const { answer, loading, error, sendQuery } = useQA()

  const handleSubmit = async (text: string, prewrittenAnswer?: string) => {
    setQuery(text)

    if (prewrittenAnswer) {
      // Use prewritten answer for popular topics
      const mockAnswer = {
        answer: prewrittenAnswer,
        confidence: 1.0,
        sources: [],
        processing_time: 0.1
      }
      setConversationHistory(prev => [...prev, { query: text, answer: mockAnswer }])
    } else {
      // Use backend for custom queries
      await sendQuery(text)
    }
  }

  useEffect(() => {
    if (answer && query && !conversationHistory.some(item => item.query === query && item.answer === answer)) {
      setConversationHistory(prev => [...prev, { query, answer }])
    }
  }, [answer, query, conversationHistory])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-800 via-slate-700 to-blue-900">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-500/10 to-indigo-600/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-blue-400/10 to-cyan-600/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      <Header />

      <main className="relative z-10 container mx-auto px-4 py-8 max-w-4xl">
        {/* Welcome section for first load */}
        {conversationHistory.length === 0 && !loading && (
          <div className="text-center mb-12 animate-fade-in">
            <div className="mb-8">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl mb-6 shadow-lg">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a9 9 0 117.072 0l-.548.547A3.374 3.374 0 0014.846 21H9.154a3.374 3.374 0 00-2.548-1.146l-.548-.547z" />
                </svg>
              </div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-200 via-cyan-200 to-blue-300 bg-clip-text text-transparent mb-4">
                Welcome to AnantaAI
              </h1>
              <p className="text-xl text-slate-300 max-w-2xl mx-auto">
                Your intelligent assistant for IISc M.Mgt queries. Ask anything about the program, admissions, curriculum, or campus life.
              </p>
            </div>
          </div>
        )}

        {/* Conversation history */}
        {conversationHistory.length > 0 && (
          <div className="space-y-8 mb-8">
            {conversationHistory.map((item, index) => (
              <div key={index} className="space-y-4">
                {/* User query */}
                <div className="flex justify-end">
                  <div className="max-w-3xl bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl rounded-tr-md px-6 py-4 shadow-lg">
                    <p className="text-sm font-medium mb-1">You</p>
                    <p>{item.query}</p>
                  </div>
                </div>
                {/* AI response */}
                <Answer answer={item.answer} query={item.query} />
              </div>
            ))}
          </div>
        )}

        {/* Current loading state */}
        {loading && (
          <div className="flex justify-end mb-4">
            <div className="max-w-3xl bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl rounded-tr-md px-6 py-4 shadow-lg">
              <p className="text-sm font-medium mb-1">You</p>
              <p>{query}</p>
            </div>
          </div>
        )}

        {loading && <TypingIndicator />}

        {/* Error state */}
        {error && (
          <ErrorMessage
            error={error}
            onRetry={() => query && sendQuery(query)}
          />
        )}

        {/* Current answer (if not in history yet) */}
        {answer && !conversationHistory.some(item => item.query === query && item.answer === answer) && (
          <Answer answer={answer} query={query} />
        )}

        {/* Chat input - always at bottom */}
        <div className="sticky bottom-4">
          <ChatInput
            onSubmit={handleSubmit}
            disabled={loading}
          />
        </div>

        {/* Suggestions - always show */}
        <Suggestions onSelect={handleSubmit} />
      </main>
    </div>
  )
}

export default App
