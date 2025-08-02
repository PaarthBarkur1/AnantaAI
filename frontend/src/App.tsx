import { useState, useEffect, useRef } from 'react'
import { ChatInput } from './components/ChatInput'
import { Answer } from './components/Answer'
import { Suggestions } from './components/Suggestions'
import { Header } from './components/Header'
import { TypingIndicator } from './components/LoadingSkeleton'
import { ErrorMessage } from './components/ErrorMessage'
import { useQA } from './hooks/useQA'

function App() {
  const [conversationHistory, setConversationHistory] = useState<Array<{query: string, answer: any}>>([])
  const { answer, loading, error, sendQuery, setAnswer } = useQA()
  const currentQueryRef = useRef<string>('')

  const handleSubmit = async (text: string, prewrittenAnswer?: string) => {
    if (prewrittenAnswer) {
      // Use prewritten answer - add directly to history
      const mockAnswer = {
        answer: prewrittenAnswer,
        confidence: 1.0,
        sources: [],
        processing_time: 0.1
      }
      setConversationHistory(prev => [...prev, { query: text, answer: mockAnswer }])
    } else {
      // Use backend for custom queries
      console.log('Setting currentQueryRef to:', text)
      currentQueryRef.current = text
      // Add user question to history immediately with a placeholder answer
      setConversationHistory(prev => [...prev, { 
        query: text, 
        answer: { 
          answer: '', 
          confidence: 0, 
          sources: [], 
          processing_time: 0 
        } 
      }])
      await sendQuery(text)
    }
  }

  // Update the last conversation item with the backend response when it arrives
  useEffect(() => {
    console.log('Answer effect triggered:', { answer, currentQueryRef: currentQueryRef.current })
    if (answer && currentQueryRef.current) {
      console.log('Updating conversation with answer:', answer)
      setConversationHistory(prev => {
        const newHistory = [...prev]
        // Find and update the last item that has an empty answer (our placeholder)
        const lastIndex = newHistory.length - 1
        console.log('Checking last item:', { 
          lastIndex, 
          lastItem: newHistory[lastIndex], 
          currentQuery: currentQueryRef.current,
          lastItemQuery: newHistory[lastIndex]?.query,
          lastItemAnswer: newHistory[lastIndex]?.answer?.answer 
        })
        
        // More flexible matching - just find the last item with empty answer
        if (lastIndex >= 0 && newHistory[lastIndex].answer.answer === '') {
          console.log('Updating item at index:', lastIndex, 'with answer:', answer)
          newHistory[lastIndex] = { 
            query: newHistory[lastIndex].query, // Keep the original query
            answer: {
              answer: answer.answer,
              confidence: answer.confidence,
              sources: answer.sources || [],
              processing_time: answer.processing_time
            }
          }
          console.log('Updated item:', newHistory[lastIndex])
        } else {
          console.log('No matching item found to update')
        }
        return newHistory
      })
      // Don't clear currentQueryRef until after the state update
      setTimeout(() => {
        console.log('Clearing currentQueryRef')
        currentQueryRef.current = ''
      }, 100)
    }
  }, [answer])

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
            {conversationHistory.map((item, index) => {
              return (
                <div key={`${index}-${item.query}-${item.answer.answer?.substring(0, 10)}`} className="space-y-4">
                  {/* User query */}
                  <div className="flex justify-end">
                    <div className="max-w-3xl bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl rounded-tr-md px-6 py-4 shadow-lg">
                      <p className="text-sm font-medium mb-1">You</p>
                      <p>{item.query}</p>
                    </div>
                  </div>
                  {/* AI response - only show if there's an actual answer */}
                  {item.answer.answer && <Answer answer={item.answer} query={item.query} />}
                  {/* Show typing indicator if this is the last item and it has an empty answer */}
                  {index === conversationHistory.length - 1 && item.answer.answer === '' && loading && (
                    <TypingIndicator />
                  )}
                </div>
              )
            })}
          </div>
        )}

        {/* Error state */}
        {error && (
          <ErrorMessage
            error={error}
            onRetry={() => currentQueryRef.current && sendQuery(currentQueryRef.current)}
          />
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
