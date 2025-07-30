import { useState } from 'react'
import { ChatInput } from './components/ChatInput'
import { Answer } from './components/Answer'
import { Suggestions } from './components/Suggestions'
import { Header } from './components/Header'
import { useQA } from './hooks/useQA'

function App() {
  const [query, setQuery] = useState('')
  const { answer, loading, error, sendQuery } = useQA()

  const handleSubmit = async (text: string) => {
    setQuery(text)
    await sendQuery(text)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <ChatInput 
          onSubmit={handleSubmit}
          disabled={loading}
        />

        {loading && (
          <div className="mt-8 flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        )}

        {error && (
          <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {answer && <Answer answer={answer} query={query} />}

        <Suggestions onSelect={handleSubmit} />
      </main>
    </div>
  )
}

export default App
