import { useState } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface Answer {
  answer: string
  confidence: number
  sources: Array<{
    source: string
    url: string
    confidence: number
  }>
  processing_time: number
}

export function useQA() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [answer, setAnswer] = useState<Answer | null>(null)

  const sendQuery = async (text: string) => {
    console.log('Sending query:', text)
    setLoading(true)
    setError(null)
    setAnswer(null) // Clear previous answer when starting new query
    
    try {
      const response = await axios.post(`${API_URL}/api/query`, {
        text,
        max_results: 3
      })
      console.log('API response received:', response.data)
      setAnswer(response.data)
    } catch (err) {
      console.error('API error:', err)
      setError(err instanceof Error ? err.message : 'An error occurred')
      setAnswer(null)
    } finally {
      setLoading(false)
    }
  }

  return {
    loading,
    error,
    answer,
    sendQuery,
    setAnswer
  }
}
