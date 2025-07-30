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
  return (
    <div className="mt-8 space-y-4">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <h3 className="text-sm font-medium text-gray-500">Your Question</h3>
          <p className="mt-1 text-gray-900">{query}</p>
        </div>
        
        <div className="p-4">
          <h3 className="text-sm font-medium text-gray-500">Answer</h3>
          <p className="mt-2 text-gray-900 whitespace-pre-wrap">{answer.answer}</p>
          
          <div className="mt-4 flex items-center gap-4 text-sm text-gray-500">
            <span>
              Confidence: {(answer.confidence * 100).toFixed(1)}%
            </span>
            <span>
              Time: {answer.processing_time.toFixed(2)}s
            </span>
          </div>
        </div>
      </div>

      {answer.sources.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h3 className="text-sm font-medium text-gray-500 mb-3">Sources</h3>
          <div className="space-y-3">
            {answer.sources.map((source, i) => (
              <div key={i} className="text-sm">
                <a 
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  {source.source}
                </a>
                <div className="text-gray-500 text-xs mt-1">
                  Confidence: {(source.confidence * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
