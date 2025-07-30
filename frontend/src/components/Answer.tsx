import { useState } from 'react'
import { ClipboardDocumentIcon, HandThumbUpIcon, HandThumbDownIcon, ShareIcon } from '@heroicons/react/24/outline'
import { CheckIcon } from '@heroicons/react/24/solid'
import ReactMarkdown from 'react-markdown'

interface AnswerProps {
  answer: {
    answer: string
    confidence: number
    sources: Array<{
      source: string
      url: string
      confidence: number
    }>
    processing_time: number
  }
  query: string
}

export function Answer({ answer, query }: AnswerProps) {
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(answer.answer)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleFeedback = (type: 'up' | 'down') => {
    setFeedback(type)
    // Here you could send feedback to your backend
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-50'
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
  }

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    return 'Low'
  }

  return (
    <div className="flex justify-start mb-8">
      <div className="max-w-4xl w-full">
        {/* AI Avatar and Header */}
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a9 9 0 117.072 0l-.548.547A3.374 3.374 0 0014.846 21H9.154a3.374 3.374 0 00-2.548-1.146l-.548-.547z" />
              </svg>
            </div>
          </div>

          <div className="flex-1">
            {/* Answer Card */}
            <div className="bg-slate-800/90 backdrop-blur-sm rounded-2xl rounded-tl-md shadow-lg border border-slate-600/30 overflow-hidden">
              {/* Header */}
              <div className="px-6 py-4 border-b border-slate-700/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-sm font-medium text-slate-100">AnantaAI</span>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(answer.confidence)}`}>
                      {getConfidenceLabel(answer.confidence)} confidence
                    </div>
                  </div>
                  <div className="text-xs text-slate-400">
                    {answer.processing_time.toFixed(2)}s
                  </div>
                </div>
              </div>

              {/* Answer Content */}
              <div className="px-6 py-6">
                <div className="prose prose-gray max-w-none">
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => <p className="mb-4 text-slate-200 leading-relaxed">{children}</p>,
                      ul: ({ children }) => <ul className="mb-4 space-y-2 text-slate-200">{children}</ul>,
                      ol: ({ children }) => <ol className="mb-4 space-y-2 text-slate-200">{children}</ol>,
                      li: ({ children }) => <li className="flex items-start"><span className="mr-2">•</span><span>{children}</span></li>,
                      strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
                      em: ({ children }) => <em className="italic text-slate-300">{children}</em>,
                      code: ({ children }) => <code className="px-2 py-1 bg-slate-700 rounded text-sm font-mono text-slate-200">{children}</code>,
                    }}
                  >
                    {answer.answer}
                  </ReactMarkdown>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="px-6 py-4 border-t border-slate-700/50 bg-slate-900/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={handleCopy}
                      className="flex items-center space-x-2 px-3 py-2 text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-700/80 rounded-lg transition-colors duration-200"
                    >
                      {copied ? (
                        <CheckIcon className="h-4 w-4 text-green-500" />
                      ) : (
                        <ClipboardDocumentIcon className="h-4 w-4" />
                      )}
                      <span>{copied ? 'Copied!' : 'Copy'}</span>
                    </button>

                    <button
                      onClick={() => handleFeedback('up')}
                      className={`p-2 rounded-lg transition-colors duration-200 ${
                        feedback === 'up'
                          ? 'text-green-400 bg-green-900/50'
                          : 'text-slate-400 hover:text-green-400 hover:bg-green-900/30'
                      }`}
                    >
                      <HandThumbUpIcon className="h-4 w-4" />
                    </button>

                    <button
                      onClick={() => handleFeedback('down')}
                      className={`p-2 rounded-lg transition-colors duration-200 ${
                        feedback === 'down'
                          ? 'text-red-400 bg-red-900/50'
                          : 'text-slate-400 hover:text-red-400 hover:bg-red-900/30'
                      }`}
                    >
                      <HandThumbDownIcon className="h-4 w-4" />
                    </button>

                    <button className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700/80 rounded-lg transition-colors duration-200">
                      <ShareIcon className="h-4 w-4" />
                    </button>
                  </div>

                  <div className="text-xs text-slate-400">
                    {(answer.confidence * 100).toFixed(1)}% confidence
                  </div>
                </div>
              </div>
            </div>

            {/* Sources */}
            {answer.sources.length > 0 && (
              <div className="mt-4 bg-slate-800/60 backdrop-blur-sm rounded-xl border border-slate-600/30 p-4">
                <h4 className="text-sm font-medium text-slate-300 mb-3 flex items-center">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.102m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                  </svg>
                  Sources
                </h4>
                <div className="grid gap-3 sm:grid-cols-2">
                  {answer.sources.map((source, i) => (
                    <a
                      key={i}
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block p-3 bg-slate-700/80 rounded-lg border border-slate-600/30 hover:border-blue-400/50 hover:shadow-md transition-all duration-200 group"
                    >
                      <div className="text-sm font-medium text-slate-200 group-hover:text-blue-300 transition-colors">
                        {source.source}
                      </div>
                      <div className="text-xs text-slate-400 mt-1">
                        {(source.confidence * 100).toFixed(1)}% relevance
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
