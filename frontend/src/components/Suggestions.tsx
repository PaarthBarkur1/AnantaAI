import { useState } from 'react'
import { AcademicCapIcon, BriefcaseIcon, DocumentTextIcon, BuildingOfficeIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline'

const SUGGESTION_CATEGORIES = [
  {
    title: "Admissions",
    icon: AcademicCapIcon,
    color: "from-blue-500 to-cyan-500",
    suggestions: [
      {
        question: "What are the eligibility criteria for M.Mgt?",
        answer: "To be eligible for the M.Mgt program at IISc, candidates must have:\n\n• A Bachelor's degree in any discipline with at least 50% marks\n• Valid CAT/XAT/GMAT score\n• Work experience is preferred but not mandatory\n• Strong academic background and leadership potential\n\nThe program welcomes students from diverse academic backgrounds including engineering, science, commerce, and humanities."
      },
      {
        question: "How do I apply for the program?",
        answer: "The application process for IISc M.Mgt involves:\n\n1. **Online Application**: Submit application through the official IISc website\n2. **Documents Required**: Academic transcripts, test scores (CAT/XAT/GMAT), work experience certificates, SOP\n3. **Application Fee**: Pay the required application fee online\n4. **Shortlisting**: Based on academic performance and test scores\n5. **Interview**: Shortlisted candidates are called for personal interview\n6. **Final Selection**: Based on overall profile evaluation\n\nApplications typically open in November-December each year."
      },
      {
        question: "What is the selection process?",
        answer: "The selection process is comprehensive and includes:\n\n**Stage 1: Application Screening**\n• Academic performance evaluation\n• Entrance test scores (CAT/XAT/GMAT)\n• Work experience assessment\n\n**Stage 2: Personal Interview**\n• Technical and managerial aptitude\n• Communication skills\n• Leadership potential\n• Fit with program objectives\n\n**Final Selection**\n• Holistic evaluation of all parameters\n• Diversity in batch composition\n• Academic merit and potential"
      },
      {
        question: "When are the application deadlines?",
        answer: "Key dates for M.Mgt admissions:\n\n• **Application Opens**: November\n• **Application Deadline**: January 31st\n• **Shortlist Announcement**: February\n• **Interview Dates**: March\n• **Final Results**: April\n• **Program Starts**: July\n\n*Note: Dates may vary slightly each year. Please check the official IISc website for exact dates.*"
      }
    ]
  },
  {
    title: "Placements",
    icon: BriefcaseIcon,
    color: "from-green-500 to-emerald-500",
    suggestions: [
      {
        question: "What are the placement statistics?",
        answer: "IISc M.Mgt has excellent placement records:\n\n**Recent Statistics:**\n• **Placement Rate**: 100% for willing students\n• **Average Package**: ₹18-22 LPA\n• **Highest Package**: ₹45+ LPA\n• **Median Package**: ₹16-20 LPA\n\n**Sector Distribution:**\n• Consulting: 35%\n• Technology: 30%\n• Finance: 20%\n• Manufacturing: 15%\n\nThe placement cell provides comprehensive support throughout the process."
      },
      {
        question: "Which companies visit for recruitment?",
        answer: "Top recruiters include:\n\n**Consulting Firms:**\n• McKinsey & Company\n• Boston Consulting Group\n• Bain & Company\n• Deloitte\n• PwC\n\n**Technology Companies:**\n• Microsoft\n• Google\n• Amazon\n• Flipkart\n• Uber\n\n**Financial Services:**\n• Goldman Sachs\n• JP Morgan\n• ICICI Bank\n• HDFC Bank\n\n**Manufacturing:**\n• Tata Group\n• Mahindra\n• L&T\n• Godrej\n\nOver 100+ companies participate in the placement process annually."
      },
      {
        question: "What is the average salary package?",
        answer: "Salary packages at IISc M.Mgt are competitive:\n\n**Package Distribution:**\n• **Average CTC**: ₹18-22 LPA\n• **Median CTC**: ₹16-20 LPA\n• **Top 10% Average**: ₹35+ LPA\n• **Highest Package**: ₹45+ LPA\n\n**Sector-wise Averages:**\n• Consulting: ₹25-30 LPA\n• Technology: ₹20-25 LPA\n• Finance: ₹18-24 LPA\n• Manufacturing: ₹15-20 LPA\n\n*Packages include base salary, performance bonus, and other benefits.*"
      },
      {
        question: "What roles do graduates get?",
        answer: "M.Mgt graduates are placed in diverse roles:\n\n**Management Consulting:**\n• Business Analyst\n• Associate Consultant\n• Strategy Consultant\n\n**Technology:**\n• Product Manager\n• Business Development\n• Program Manager\n\n**Finance:**\n• Investment Banking Analyst\n• Corporate Finance\n• Risk Management\n\n**General Management:**\n• Management Trainee\n• Assistant Manager\n• Business Development Manager\n\nGraduates often progress to senior leadership roles within 5-7 years."
      }
    ]
  },
  {
    title: "Curriculum",
    icon: DocumentTextIcon,
    color: "from-purple-500 to-violet-500",
    suggestions: [
      {
        question: "How is the curriculum structured?",
        answer: "The M.Mgt curriculum is designed as a 2-year full-time program:\n\n**Year 1 (Foundation):**\n• Core management subjects\n• Quantitative methods\n• Economics and finance\n• Marketing and operations\n• Organizational behavior\n\n**Year 2 (Specialization):**\n• Elective courses\n• Industry projects\n• Internship (summer)\n• Capstone project\n\n**Key Features:**\n• Case-based learning\n• Industry interaction\n• Research projects\n• International exposure opportunities"
      },
      {
        question: "What subjects are covered?",
        answer: "Comprehensive curriculum covering:\n\n**Core Subjects:**\n• Financial Management\n• Marketing Management\n• Operations Management\n• Human Resource Management\n• Strategic Management\n• Business Analytics\n• Economics for Managers\n• Organizational Behavior\n\n**Electives:**\n• Digital Marketing\n• Supply Chain Management\n• Investment Banking\n• Entrepreneurship\n• Technology Management\n• Sustainability\n• International Business\n\n**Special Features:**\n• Industry projects\n• Case competitions\n• Research methodology"
      },
      {
        question: "Are there any specializations?",
        answer: "The program offers flexible specialization options:\n\n**Major Specializations:**\n• **Finance**: Investment banking, corporate finance, risk management\n• **Marketing**: Digital marketing, brand management, consumer insights\n• **Operations**: Supply chain, analytics, technology management\n• **Strategy**: Business strategy, consulting, entrepreneurship\n\n**Emerging Areas:**\n• Data Analytics\n• Digital Transformation\n• Sustainability\n• Innovation Management\n\n**Flexibility:**\n• Students can choose multiple electives\n• Cross-functional projects encouraged\n• Industry-specific tracks available"
      },
      {
        question: "What is the duration of the program?",
        answer: "The M.Mgt program duration details:\n\n**Total Duration**: 2 Years (4 Semesters)\n\n**Academic Calendar:**\n• **Semester 1**: July - November\n• **Semester 2**: December - April\n• **Summer Internship**: May - June\n• **Semester 3**: July - November\n• **Semester 4**: December - April\n\n**Key Milestones:**\n• Foundation courses: First year\n• Summer internship: Between years\n• Specialization: Second year\n• Capstone project: Final semester\n\n**Total Credits**: 120+ credits\n**Class Hours**: 25-30 hours per week"
      }
    ]
  },
  {
    title: "Campus Life",
    icon: BuildingOfficeIcon,
    color: "from-orange-500 to-red-500",
    suggestions: [
      {
        question: "What facilities are available on campus?",
        answer: "IISc offers world-class facilities:\n\n**Academic Facilities:**\n• Modern classrooms with AV equipment\n• Computer labs with latest software\n• Library with extensive business collection\n• Case study rooms\n• Seminar halls\n\n**Recreation:**\n• Sports complex with gym\n• Swimming pool\n• Tennis and badminton courts\n• Cricket and football grounds\n• Yoga and meditation center\n\n**Other Amenities:**\n• Medical center\n• Banking and ATM\n• Cafeterias and food courts\n• Guest house\n• Transportation services"
      },
      {
        question: "How is the hostel accommodation?",
        answer: "Comfortable hostel facilities for students:\n\n**Accommodation Options:**\n• Single occupancy rooms\n• Shared rooms (for first year)\n• Separate hostels for men and women\n• Married student quarters\n\n**Room Amenities:**\n• Furnished rooms with study table\n• High-speed internet\n• 24/7 electricity and water\n• Laundry facilities\n• Common areas and recreation rooms\n\n**Hostel Services:**\n• Mess facilities with varied menu\n• Security and maintenance\n• Housekeeping services\n• Wi-Fi connectivity\n\n**Cost**: Approximately ₹40,000-60,000 per year"
      },
      {
        question: "What extracurricular activities are there?",
        answer: "Rich extracurricular ecosystem:\n\n**Student Clubs:**\n• Consulting Club\n• Finance Club\n• Marketing Club\n• Entrepreneurship Cell\n• Cultural Committee\n• Sports Committee\n\n**Events & Competitions:**\n• Annual business fest\n• Case competitions\n• Industry conclaves\n• Cultural festivals\n• Sports tournaments\n• Alumni meets\n\n**Leadership Opportunities:**\n• Student council positions\n• Club leadership roles\n• Event organization\n• Peer mentoring\n• Community service projects"
      },
      {
        question: "How is the campus environment?",
        answer: "IISc provides an inspiring academic environment:\n\n**Campus Highlights:**\n• 400+ acre green campus\n• Heritage buildings with modern facilities\n• Research-oriented atmosphere\n• Diverse student community\n• Faculty-student interaction\n\n**Academic Culture:**\n• Collaborative learning\n• Innovation and research focus\n• Industry-academia interface\n• Global perspective\n• Ethical leadership emphasis\n\n**Location Benefits:**\n• Located in Bangalore (Silicon Valley of India)\n• Easy access to industry\n• Vibrant startup ecosystem\n• Pleasant weather year-round\n• Rich cultural heritage"
      }
    ]
  }
]

interface SuggestionsProps {
  onSelect: (question: string, answer?: string) => void
}

export function Suggestions({ onSelect }: SuggestionsProps) {
  const [expandedItems, setExpandedItems] = useState<{[key: string]: boolean}>({})

  const toggleExpanded = (categoryIndex: number, suggestionIndex: number) => {
    const key = `${categoryIndex}-${suggestionIndex}`
    setExpandedItems(prev => ({
      ...prev,
      [key]: !prev[key]
    }))
  }
  return (
    <div className="mt-12 space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-slate-100 mb-2">
          Explore Popular Topics
        </h2>
        <p className="text-slate-300">
          Get started with these commonly asked questions
        </p>
      </div>

      <div className="suggestions-grid">
        {SUGGESTION_CATEGORIES.map((category, categoryIndex) => (
          <div
            key={categoryIndex}
            className="bg-slate-800/80 backdrop-blur-sm rounded-2xl border border-slate-600/30 shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden group"
          >
            {/* Category Header */}
            <div className={`p-6 bg-gradient-to-r ${category.color} text-white`}>
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-white/20 rounded-lg">
                  <category.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold">{category.title}</h3>
              </div>
            </div>

            {/* Suggestions */}
            <div className="p-6 space-y-3">
              {category.suggestions.map((suggestion, suggestionIndex) => {
                const key = `${categoryIndex}-${suggestionIndex}`
                const isExpanded = expandedItems[key]

                return (
                  <div key={suggestionIndex} className="border border-transparent rounded-xl overflow-hidden relative">
                    <button
                      onClick={() => toggleExpanded(categoryIndex, suggestionIndex)}
                      className="w-full text-left p-3 rounded-xl bg-slate-700/80 hover:bg-blue-900/30 hover:border-blue-400/50 border border-transparent transition-all duration-200 group-hover:shadow-sm"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-slate-300 hover:text-blue-300 transition-colors">
                          {suggestion.question}
                        </span>
                        {isExpanded ? (
                          <ChevronUpIcon className="w-4 h-4 text-slate-500 group-hover:text-blue-400 transition-colors" />
                        ) : (
                          <ChevronDownIcon className="w-4 h-4 text-slate-500 group-hover:text-blue-400 transition-colors" />
                        )}
                      </div>
                    </button>

                    {isExpanded && (
                      <div className="mt-2 p-4 bg-slate-800/60 rounded-xl border border-slate-600/30 animate-fade-in">
                        <div className="prose prose-sm max-w-none">
                          <div
                            className="text-slate-200 leading-relaxed whitespace-pre-line text-sm"
                            style={{
                              lineHeight: '1.6',
                              fontFamily: 'inherit'
                            }}
                          >
                            {suggestion.answer.split('\n').map((line, index) => {
                              if (line.startsWith('**') && line.endsWith('**')) {
                                return (
                                  <div key={index} className="font-semibold text-blue-300 mt-3 mb-2">
                                    {line.replace(/\*\*/g, '')}
                                  </div>
                                )
                              } else if (line.startsWith('• ')) {
                                return (
                                  <div key={index} className="ml-4 mb-1 flex items-start">
                                    <span className="text-blue-400 mr-2 mt-1">•</span>
                                    <span>{line.substring(2)}</span>
                                  </div>
                                )
                              } else if (line.trim() === '') {
                                return <div key={index} className="h-2"></div>
                              } else {
                                return (
                                  <div key={index} className="mb-1">
                                    {line}
                                  </div>
                                )
                              }
                            })}
                          </div>
                        </div>

                        {/* Optional: Add to chat button */}
                        <div className="mt-4 pt-3 border-t border-slate-600/30">
                          <button
                            onClick={() => onSelect(suggestion.question, suggestion.answer)}
                            className="text-xs text-blue-400 hover:text-blue-300 transition-colors flex items-center space-x-1"
                          >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.959 8.959 0 01-4.906-1.456L3 21l2.544-5.094A8.959 8.959 0 013 12c0-4.418 3.582-8 8-8s8 3.582 8 8z" />
                            </svg>
                            <span>Add to chat</span>
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="bg-slate-800/60 backdrop-blur-sm rounded-2xl border border-slate-600/30 p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4 text-center">
          Quick Actions
        </h3>
        <div className="flex flex-wrap justify-center gap-3">
          {[
            { text: "Compare with other programs", icon: "📊" },
            { text: "Download brochure", icon: "📄" },
            { text: "Contact admissions", icon: "📞" },
            { text: "Virtual campus tour", icon: "🏛️" }
          ].map((action, index) => (
            <button
              key={index}
              onClick={() => onSelect(action.text, undefined)}
              className="flex items-center space-x-2 px-4 py-2 bg-slate-700/80 hover:bg-blue-900/30 rounded-full border border-slate-600/30 hover:border-blue-400/50 transition-all duration-200 text-sm text-slate-300 hover:text-blue-300"
            >
              <span>{action.icon}</span>
              <span>{action.text}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
