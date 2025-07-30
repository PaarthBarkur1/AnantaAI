import { useState } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

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
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post(`${API_URL}/api/query`, {
        text,
        max_results: 3
      })
      setAnswer(response.data)
    } catch (err) {
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
    sendQuery
  }
}
