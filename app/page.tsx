"use client"

import { useEffect, useState } from "react"
import { TrendingUp, Activity, Zap, Sparkles, Rocket, Brain, Target, Flame, Star } from "lucide-react"

interface Signal {
  symbol: string
  strike: number
  option_type: string
  entry_price: number
  confidence: number
  reason: string
  timestamp: string
}

export default function TradingDashboard() {
  const [signals, setSignals] = useState<Signal[]>([])
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState({
    totalSignals: 0,
    avgConfidence: 0,
    activeSignals: 0,
  })

  const fetchSignals = async () => {
    try {
      const response = await fetch(
        "https://raw.githubusercontent.com/athreya9/my-trading-platform/main/data/signals.json",
      )
      const data = await response.json()
      setSignals(data)

      // Calculate stats
      const total = data.length
      const avg = data.reduce((acc: number, s: Signal) => acc + s.confidence, 0) / total
      const active = data.filter((s: Signal) => {
        const signalTime = new Date(s.timestamp).getTime()
        const now = Date.now()
        return now - signalTime < 3600000 // Active if less than 1 hour old
      }).length

      setStats({
        totalSignals: total,
        avgConfidence: Math.round(avg * 100),
        activeSignals: active,
      })
      setLoading(false)
    } catch (error) {
      console.error("[v0] Error fetching signals:", error)
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSignals()
    const interval = setInterval(fetchSignals, 15000) // Auto-refresh every 15s
    return () => clearInterval(interval)
  }, [])

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "from-green-400 to-emerald-500"
    if (confidence >= 0.6) return "from-yellow-400 to-orange-500"
    return "from-red-400 to-pink-500"
  }

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      {/* Enhanced animated gradient background */}
      <div className="fixed inset-0 bg-gradient-to-br from-purple-900/30 via-black to-cyan-900/30 animate-pulse" />
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-pink-500/20 via-purple-500/10 to-transparent animate-pulse" />
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-cyan-500/15 via-transparent to-transparent" />

      {/* Floating particles */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full animate-bounce opacity-30"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 3}s`
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8 max-w-7xl">
        {/* Enhanced Header */}
        <header className="mb-16 text-center relative">
          <div className="absolute inset-0 flex items-center justify-center opacity-10">
            <Sparkles className="w-96 h-96 text-purple-500 animate-spin" style={{ animationDuration: '20s' }} />
          </div>
          <div className="relative">
            <div className="flex items-center justify-center gap-4 mb-6">
              <Rocket className="w-12 h-12 text-purple-400 animate-bounce" />
              <Brain className="w-10 h-10 text-pink-400 animate-pulse" />
              <Target className="w-8 h-8 text-cyan-400 animate-spin" />
            </div>
            <h1 className="text-7xl md:text-8xl font-black mb-6 bg-gradient-to-r from-purple-400 via-pink-500 to-cyan-400 bg-clip-text text-transparent animate-pulse">
              AI TRADING
            </h1>
            <div className="flex items-center justify-center gap-2 mb-4">
              <Flame className="w-6 h-6 text-orange-400 animate-bounce" />
              <span className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                PLATFORM
              </span>
              <Flame className="w-6 h-6 text-orange-400 animate-bounce" />
            </div>
            <p className="text-gray-300 text-xl font-medium">
              Real-time signals powered by <span className="text-purple-400 font-bold">artificial intelligence</span> ✨
            </p>
            <div className="flex items-center justify-center gap-1 mt-4">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="w-5 h-5 text-yellow-400 fill-current animate-pulse" style={{ animationDelay: `${i * 0.2}s` }} />
              ))}
            </div>
          </div>
        </header>

        {/* Enhanced Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          <div className="group relative overflow-hidden rounded-3xl bg-gradient-to-br from-purple-500/20 to-purple-900/20 backdrop-blur-2xl border-2 border-purple-500/30 p-8 hover:border-purple-400/70 transition-all duration-500 hover:scale-110 hover:rotate-1 hover:shadow-[0_0_50px_rgba(168,85,247,0.6)] cursor-pointer">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="absolute top-2 right-2">
              <div className="w-3 h-3 rounded-full bg-purple-400 animate-ping" />
            </div>
            <div className="relative">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 rounded-2xl bg-purple-500/20 border border-purple-400/50">
                  <TrendingUp className="w-8 h-8 text-purple-400" />
                </div>
                <div className="text-xs font-bold text-purple-300 bg-purple-500/20 px-3 py-1 rounded-full border border-purple-400/50">
                  TOTAL
                </div>
              </div>
              <p className="text-gray-300 text-sm mb-3 font-medium">Total Signals Generated</p>
              <p className="text-5xl font-black text-purple-400 mb-2">{stats.totalSignals}</p>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-xs text-green-400 font-bold">ALL TIME</span>
              </div>
            </div>
          </div>

          <div className="group relative overflow-hidden rounded-3xl bg-gradient-to-br from-pink-500/20 to-pink-900/20 backdrop-blur-2xl border-2 border-pink-500/30 p-8 hover:border-pink-400/70 transition-all duration-500 hover:scale-110 hover:rotate-1 hover:shadow-[0_0_50px_rgba(236,72,153,0.6)] cursor-pointer">
            <div className="absolute inset-0 bg-gradient-to-br from-pink-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="absolute top-2 right-2">
              <div className="w-3 h-3 rounded-full bg-pink-400 animate-ping" />
            </div>
            <div className="relative">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 rounded-2xl bg-pink-500/20 border border-pink-400/50">
                  <Activity className="w-8 h-8 text-pink-400" />
                </div>
                <div className="text-xs font-bold text-pink-300 bg-pink-500/20 px-3 py-1 rounded-full border border-pink-400/50">
                  ACCURACY
                </div>
              </div>
              <p className="text-gray-300 text-sm mb-3 font-medium">Average Confidence Score</p>
              <p className="text-5xl font-black text-pink-400 mb-2">{stats.avgConfidence}%</p>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                <span className="text-xs text-yellow-400 font-bold">HIGH PRECISION</span>
              </div>
            </div>
          </div>

          <div className="group relative overflow-hidden rounded-3xl bg-gradient-to-br from-cyan-500/20 to-cyan-900/20 backdrop-blur-2xl border-2 border-cyan-500/30 p-8 hover:border-cyan-400/70 transition-all duration-500 hover:scale-110 hover:rotate-1 hover:shadow-[0_0_50px_rgba(34,211,238,0.6)] cursor-pointer">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="absolute top-2 right-2">
              <div className="w-3 h-3 rounded-full bg-cyan-400 animate-ping" />
            </div>
            <div className="relative">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 rounded-2xl bg-cyan-500/20 border border-cyan-400/50">
                  <Zap className="w-8 h-8 text-cyan-400" />
                </div>
                <div className="text-xs font-bold text-cyan-300 bg-cyan-500/20 px-3 py-1 rounded-full border border-cyan-400/50">
                  LIVE
                </div>
              </div>
              <p className="text-gray-300 text-sm mb-3 font-medium">Active Trading Signals</p>
              <p className="text-5xl font-black text-cyan-400 mb-2">{stats.activeSignals}</p>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-xs text-green-400 font-bold">REAL-TIME</span>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Live Signals Section */}
        <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-gray-900/60 to-gray-800/60 backdrop-blur-2xl border-2 border-gray-600/50 p-10">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 via-pink-500/5 to-cyan-500/5 animate-pulse" />

          <div className="relative flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <div className="p-4 rounded-2xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-400/50">
                <Brain className="w-10 h-10 text-purple-400" />
              </div>
              <div>
                <h2 className="text-4xl font-black bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
                  LIVE SIGNALS
                </h2>
                <p className="text-gray-400 font-medium">AI-powered trading recommendations</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 bg-green-500/20 px-4 py-2 rounded-full border border-green-400/50">
                <div className="w-3 h-3 rounded-full bg-green-400 animate-ping" />
                <span className="text-sm font-bold text-green-400">LIVE</span>
              </div>
              <div className="text-xs text-gray-500 bg-gray-800/50 px-3 py-2 rounded-full border border-gray-600/50">
                Auto-refresh: 15s
              </div>
            </div>
          </div>

          {loading ? (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="relative mb-8">
                <div className="w-20 h-20 rounded-full border-4 border-purple-500/20 border-t-purple-400 animate-spin" />
                <div className="absolute inset-2 w-16 h-16 rounded-full border-4 border-pink-500/20 border-t-pink-400 animate-spin" style={{ animationDirection: 'reverse' }} />
                <div className="absolute inset-4 w-12 h-12 rounded-full border-4 border-cyan-500/20 border-t-cyan-400 animate-spin" />
              </div>
              <div className="flex gap-3 mb-6">
                <div className="w-4 h-4 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-4 h-4 rounded-full bg-pink-500 animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-4 h-4 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
              <p className="text-gray-300 text-xl font-bold mb-2">Loading AI Signals...</p>
              <p className="text-gray-500">Analyzing market data in real-time</p>
            </div>
          ) : signals.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="relative mb-8">
                <Brain className="w-24 h-24 text-purple-400 animate-pulse" />
                <div className="absolute -top-2 -right-2">
                  <Sparkles className="w-8 h-8 text-yellow-400 animate-bounce" />
                </div>
              </div>
              <div className="flex gap-3 mb-6">
                <div className="w-4 h-4 rounded-full bg-purple-500 animate-pulse" style={{ animationDelay: "0ms" }} />
                <div className="w-4 h-4 rounded-full bg-pink-500 animate-pulse" style={{ animationDelay: "300ms" }} />
                <div className="w-4 h-4 rounded-full bg-cyan-500 animate-pulse" style={{ animationDelay: "600ms" }} />
              </div>
              <p className="text-gray-300 text-2xl font-bold mb-3">No Signals Available</p>
              <p className="text-gray-500 text-lg mb-4">AI is analyzing market patterns...</p>
              <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 px-6 py-3 rounded-full border border-purple-400/50">
                <span className="text-purple-300 font-bold">🤖 AI Working...</span>
              </div>
            </div>
          ) : (
            <div className="grid gap-6">
              {signals.map((signal, index) => (
                <div
                  key={index}
                  className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-xl border-2 border-gray-600/50 p-8 hover:border-purple-400/70 transition-all duration-500 hover:scale-[1.03] hover:rotate-1 hover:shadow-[0_0_40px_rgba(168,85,247,0.4)] cursor-pointer"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-pink-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                  {/* Floating elements */}
                  <div className="absolute top-4 right-4 opacity-20 group-hover:opacity-40 transition-opacity">
                    <Sparkles className="w-6 h-6 text-purple-400 animate-pulse" />
                  </div>

                  <div className="relative grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
                    {/* Enhanced Symbol & Strike */}
                    <div className="lg:col-span-3">
                      <div className="flex items-center gap-4">
                        <div className="relative">
                          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center font-black text-xl shadow-lg">
                            {signal.symbol.substring(0, 2)}
                          </div>
                          <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-green-400 animate-ping" />
                        </div>
                        <div>
                          <p className="font-black text-2xl text-white mb-1">{signal.symbol}</p>
                          <div className="flex items-center gap-2">
                            <Target className="w-4 h-4 text-gray-400" />
                            <p className="text-sm text-gray-400 font-medium">Strike: {signal.strike}</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Option Type */}
                    <div className="lg:col-span-2">
                      <div className="flex flex-col gap-2">
                        <span className="text-xs text-gray-400 font-bold uppercase tracking-wider">Option Type</span>
                        <span
                          className={`inline-block px-4 py-2 rounded-xl text-sm font-black border-2 ${signal.option_type === "CE"
                              ? "bg-green-500/30 text-green-300 border-green-400/70 shadow-[0_0_20px_rgba(34,197,94,0.3)]"
                              : "bg-red-500/30 text-red-300 border-red-400/70 shadow-[0_0_20px_rgba(239,68,68,0.3)]"
                            }`}
                        >
                          {signal.option_type}
                        </span>
                      </div>
                    </div>

                    {/* Enhanced Entry Price */}
                    <div className="lg:col-span-2">
                      <div className="flex flex-col gap-2">
                        <span className="text-xs text-gray-400 font-bold uppercase tracking-wider">Entry Price</span>
                        <div className="flex items-center gap-2">
                          <span className="text-2xl font-black text-cyan-400">₹{signal.entry_price}</span>
                          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Confidence */}
                    <div className="lg:col-span-2">
                      <div className="flex flex-col gap-3">
                        <span className="text-xs text-gray-400 font-bold uppercase tracking-wider">AI Confidence</span>
                        <div className="relative">
                          <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden border border-gray-600">
                            <div
                              className={`h-full bg-gradient-to-r ${getConfidenceColor(signal.confidence)} rounded-full transition-all duration-1000 shadow-lg`}
                              style={{ width: `${signal.confidence * 100}%` }}
                            />
                          </div>
                          <div className="flex items-center justify-between mt-2">
                            <span className="text-sm font-black text-white">{Math.round(signal.confidence * 100)}%</span>
                            <div className="flex gap-1">
                              {[...Array(5)].map((_, i) => (
                                <div
                                  key={i}
                                  className={`w-2 h-2 rounded-full ${i < Math.floor(signal.confidence * 5) ? 'bg-yellow-400' : 'bg-gray-600'
                                    }`}
                                />
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Enhanced AI Reasoning */}
                    <div className="lg:col-span-2">
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center gap-2">
                          <Brain className="w-4 h-4 text-purple-400" />
                          <span className="text-xs text-gray-400 font-bold uppercase tracking-wider">AI Analysis</span>
                        </div>
                        <p className="text-sm text-gray-300 font-medium leading-relaxed">{signal.reason}</p>
                      </div>
                    </div>

                    {/* Enhanced Timestamp */}
                    <div className="lg:col-span-1">
                      <div className="flex flex-col items-end gap-3">
                        <div className="text-right">
                          <p className="text-xs text-gray-500 mb-1">Generated at</p>
                          <p className="text-sm font-bold text-white">{formatTime(signal.timestamp)}</p>
                        </div>
                        <div className="flex flex-col gap-2">
                          <span className="inline-block px-3 py-1 rounded-full text-xs font-black bg-green-500/30 text-green-300 border border-green-400/70 shadow-[0_0_15px_rgba(34,197,94,0.3)]">
                            🔥 LIVE
                          </span>
                          <span className="inline-block px-3 py-1 rounded-full text-xs font-black bg-purple-500/30 text-purple-300 border border-purple-400/70">
                            🤖 AI
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
