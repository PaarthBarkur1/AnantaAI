const SUGGESTIONS = [
  "What are the eligibility criteria?",
  "Tell me about placement statistics",
  "How is the curriculum structured?",
  "What is the admission process?",
  "What is the average package?",
]

interface SuggestionsProps {
  onSelect: (text: string) => void
}

export function Suggestions({ onSelect }: SuggestionsProps) {
  return (
    <div className="mt-8">
      <h3 className="text-sm font-medium text-gray-500 mb-3">Popular Questions</h3>
      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map((suggestion, i) => (
          <button
            key={i}
            onClick={() => onSelect(suggestion)}
            className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  )
}
