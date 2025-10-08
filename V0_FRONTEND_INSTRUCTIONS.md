# 🎨 V0 FRONTEND INSTRUCTIONS

## 📋 PROMPT FOR V0.DEV

Copy this exact prompt to V0:

```
Create a modern Gen-Z trading dashboard with these exact requirements:

DESIGN STYLE:
- Dark theme with neon accents (purple, pink, cyan)
- Glassmorphism cards with blur effects
- Animated gradients and floating elements
- Mobile-first responsive design
- Modern typography (Inter font)

LAYOUT:
1. Header: "🚀 AI TRADING PLATFORM" with live status indicator
2. Stats cards (3 columns): Active Signals, High Confidence, Market Status
3. Live signals section with cards showing:
   - Symbol (NIFTY/SENSEX/BANKNIFTY) + Strike + Option Type
   - Entry price in ₹
   - Confidence percentage with color coding
   - AI reasoning text
   - Timestamp
   - Live/Demo badges

FEATURES:
- Auto-refresh every 15 seconds
- Empty state: "🔍 Scanning Markets..." with animated dots
- Confidence color coding: >80% green, >70% yellow, <70% red
- Hover animations on cards
- Loading states with skeleton screens

DATA SOURCE:
Fetch from: https://raw.githubusercontent.com/athreya9/my-trading-platform/main/data/signals.json

SAMPLE DATA STRUCTURE:
{
  "symbol": "SENSEX",
  "strike": 82100,
  "option_type": "CE", 
  "entry_price": 250,
  "confidence": 0.85,
  "reason": "Strong bullish momentum with volume surge",
  "timestamp": "2025-01-08T10:30:00Z"
}

Make it look like a premium trading app that Gen-Z would love to use.
```

## 🔗 INTEGRATION STEPS

1. **Go to v0.dev**
2. **Paste the prompt above**
3. **Generate the design**
4. **Download the code**
5. **Deploy to new Vercel project**

## 📊 DATA CONNECTION

The V0 frontend will automatically connect to your live data at:
`https://raw.githubusercontent.com/athreya9/my-trading-platform/main/data/signals.json`

## 🚀 DEPLOYMENT

1. Create new GitHub repo: `ai-trading-v0-frontend`
2. Upload V0 generated code
3. Connect to Vercel
4. Deploy

This keeps your main codebase untouched while giving you a premium UI.

## 🎯 EXPECTED RESULT

- Professional trading dashboard
- Real-time signal updates
- Modern Gen-Z aesthetics
- Mobile responsive
- No demo data - only live signals