export function LoadingSkeleton() {
  return (
    <div className="flex justify-start mb-8">
      <div className="max-w-4xl w-full">
        <div className="flex items-start space-x-4">
          {/* Avatar skeleton */}
          <div className="flex-shrink-0">
            <div className="w-10 h-10 bg-gradient-to-br from-gray-200 to-gray-300 rounded-full animate-pulse"></div>
          </div>

          <div className="flex-1">
            {/* Answer card skeleton */}
            <div className="bg-slate-800/90 backdrop-blur-sm rounded-2xl rounded-tl-md shadow-lg border border-slate-600/30 overflow-hidden">
              {/* Header skeleton */}
              <div className="px-6 py-4 border-b border-slate-700/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="h-4 bg-slate-600 rounded w-20 animate-pulse"></div>
                    <div className="h-6 bg-slate-600 rounded-full w-24 animate-pulse"></div>
                  </div>
                  <div className="h-3 bg-slate-600 rounded w-12 animate-pulse"></div>
                </div>
              </div>

              {/* Content skeleton */}
              <div className="px-6 py-6 space-y-3">
                <div className="h-4 bg-slate-600 rounded w-full animate-pulse"></div>
                <div className="h-4 bg-slate-600 rounded w-5/6 animate-pulse"></div>
                <div className="h-4 bg-slate-600 rounded w-4/5 animate-pulse"></div>
                <div className="h-4 bg-slate-600 rounded w-3/4 animate-pulse"></div>
                <div className="h-4 bg-slate-600 rounded w-5/6 animate-pulse"></div>
                <div className="h-4 bg-slate-600 rounded w-2/3 animate-pulse"></div>
              </div>

              {/* Action buttons skeleton */}
              <div className="px-6 py-4 border-t border-slate-700/50 bg-slate-900/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {[1, 2, 3, 4].map((i) => (
                      <div key={i} className="h-8 w-8 bg-slate-600 rounded-lg animate-pulse"></div>
                    ))}
                  </div>
                  <div className="h-3 bg-slate-600 rounded w-20 animate-pulse"></div>
                </div>
              </div>
            </div>

            {/* Sources skeleton */}
            <div className="mt-4 bg-slate-800/60 backdrop-blur-sm rounded-xl border border-slate-600/30 p-4">
              <div className="h-4 bg-slate-600 rounded w-16 mb-3 animate-pulse"></div>
              <div className="grid gap-3 sm:grid-cols-2">
                {[1, 2].map((i) => (
                  <div key={i} className="p-3 bg-slate-700/80 rounded-lg border border-slate-600/30">
                    <div className="h-4 bg-slate-600 rounded w-3/4 mb-2 animate-pulse"></div>
                    <div className="h-3 bg-slate-600 rounded w-1/2 animate-pulse"></div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export function TypingIndicator() {
  return (
    <div className="flex justify-start mb-8">
      <div className="bg-slate-800/90 backdrop-blur-sm rounded-2xl rounded-tl-md px-6 py-4 shadow-lg border border-slate-600/30">
        <div className="flex items-center space-x-3">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce delay-100"></div>
            <div className="w-2 h-2 bg-blue-300 rounded-full animate-bounce delay-200"></div>
          </div>
          <span className="text-slate-300 text-sm">AnantaAI is thinking...</span>
        </div>
      </div>
    </div>
  )
}
