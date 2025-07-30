import { UserCircleIcon, Cog6ToothIcon } from '@heroicons/react/24/outline'

export function Header() {

  return (
    <header className="relative z-20 bg-slate-800/90 backdrop-blur-xl border-b border-slate-700/50 shadow-sm">
      <div className="container mx-auto px-4 py-4 max-w-6xl">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a9 9 0 117.072 0l-.548.547A3.374 3.374 0 0014.846 21H9.154a3.374 3.374 0 00-2.548-1.146l-.548-.547z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-200 to-cyan-300 bg-clip-text text-transparent">
                  AnantaAI
                </h1>
                <p className="text-sm text-slate-400 hidden sm:block">
                  IISc M.Mgt Assistant
                </p>
              </div>
            </div>
          </div>



          {/* User Actions */}
          <div className="flex items-center space-x-3">
            {/* Settings */}
            <button className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700/80 rounded-xl transition-colors duration-200">
              <Cog6ToothIcon className="h-5 w-5" />
            </button>

            {/* User Profile */}
            <button className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700/80 rounded-xl transition-colors duration-200">
              <UserCircleIcon className="h-5 w-5" />
            </button>


          </div>
        </div>


      </div>
    </header>
  )
}
