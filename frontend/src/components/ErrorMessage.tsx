import { ExclamationTriangleIcon, ArrowPathIcon } from '@heroicons/react/24/outline'

interface ErrorMessageProps {
  error: string
  onRetry?: () => void
}

export function ErrorMessage({ error, onRetry }: ErrorMessageProps) {
  return (
    <div className="mb-8 p-6 bg-red-900/20 backdrop-blur-sm border border-red-500/30 rounded-2xl shadow-lg animate-fade-in">
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0">
          <div className="w-10 h-10 bg-red-900/50 rounded-full flex items-center justify-center">
            <ExclamationTriangleIcon className="w-5 h-5 text-red-400" />
          </div>
        </div>

        <div className="flex-1">
          <h3 className="text-sm font-medium text-red-300 mb-1">
            Something went wrong
          </h3>
          <p className="text-sm text-red-200 mb-4">
            {error}
          </p>

          {onRetry && (
            <button
              onClick={onRetry}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-red-800/50 hover:bg-red-700/50 text-red-200 rounded-lg transition-colors duration-200 text-sm font-medium"
            >
              <ArrowPathIcon className="w-4 h-4" />
              <span>Try again</span>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
