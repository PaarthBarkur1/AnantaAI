import { useState, useRef, useEffect } from 'react'
import { PaperAirplaneIcon } from '@heroicons/react/24/solid'

interface ChatInputProps {
  onSubmit: (text: string, prewrittenAnswer?: string) => void
  disabled?: boolean
}

export function ChatInput({ onSubmit, disabled }: ChatInputProps) {
  const [text, setText] = useState('')
  const [isFocused, setIsFocused] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (text.trim()) {
      onSubmit(text, undefined)  // Pass undefined for prewritten answer
      setText('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [text])

  return (
    <div className="relative">
      {/* Glassmorphism container */}
      <div className={`relative bg-slate-800/90 backdrop-blur-xl rounded-3xl border transition-all duration-300 shadow-lg ${
        isFocused
          ? 'border-blue-400/50 shadow-xl shadow-blue-500/20 ring-1 ring-blue-500/30'
          : 'border-slate-600/50 hover:border-blue-400/30'
      }`}>
        <form onSubmit={handleSubmit} className="flex items-end p-4 gap-3">
          {/* Text input */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              disabled={disabled}
              placeholder="Ask me anything about IISc M.Mgt program..."
              className="w-full resize-none bg-transparent border-none outline-none placeholder-slate-400 text-slate-100 text-base leading-6 max-h-32 min-h-[24px] py-2"
              rows={1}
            />
          </div>

          {/* Send button */}
          <div className="pb-2">
            <button
              type="submit"
              disabled={disabled || !text.trim()}
              className={`p-3 rounded-xl transition-all duration-200 ${
                text.trim() && !disabled
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              }`}
            >
              <PaperAirplaneIcon className="h-5 w-5" />
            </button>
          </div>
        </form>

        {/* Suggestions bar */}
        {isFocused && !text && (
          <div className="px-4 pb-4">
            <div className="flex gap-2 flex-wrap">
              {[
                "What are the admission requirements?",
                "Tell me about placements",
                "What courses are offered?",
                "Campus facilities"
              ].map((suggestion, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => {
                    onSubmit(suggestion, undefined)
                    setText('')
                  }}
                  className="px-3 py-1.5 text-sm bg-slate-700/80 hover:bg-slate-600/80 rounded-full text-slate-300 transition-colors duration-200"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Character count and tips */}
      {text && (
        <div className="flex justify-between items-center mt-2 px-4 text-xs text-slate-400">
          <span>Press Enter to send, Shift+Enter for new line</span>
          <span>{text.length}/2000</span>
        </div>
      )}
    </div>
  )
}
