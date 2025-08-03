# 🚀 AnantaAI Frontend Deployment Guide

## Quick Setup for Sharing with Friends

### Option 1: Netlify (Recommended - Free & Easy)

#### Step 1: Get Your Local IP Address
First, find your computer's local IP address:

**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" (usually something like 192.168.1.xxx)

**Mac/Linux:**
```bash
ifconfig | grep inet
```

#### Step 2: Update Environment Configuration
1. Open `frontend/.env.production`
2. Replace `YOUR_IP_ADDRESS` with your actual IP address
   ```
   VITE_API_URL=http://192.168.1.100:8000
   ```

#### Step 3: Deploy to Netlify
1. **Create account**: Go to [netlify.com](https://netlify.com) and sign up (free)
2. **Deploy method**: Choose one:

   **Method A: Drag & Drop (Easiest)**
   ```bash
   cd frontend
   npm run build
   ```
   Then drag the `dist` folder to Netlify's deploy area

   **Method B: Git Integration (Best for updates)**
   - Push your code to GitHub
   - Connect Netlify to your GitHub repo
   - Auto-deploys on every push!

#### Step 4: Configure Environment Variables (if using Git method)
In Netlify dashboard:
1. Go to Site Settings → Environment Variables
2. Add: `VITE_API_URL` = `http://YOUR_IP_ADDRESS:8000`

### Option 2: Vercel (Alternative)

1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repo
3. Add environment variable: `VITE_API_URL` = `http://YOUR_IP_ADDRESS:8000`
4. Deploy!

### Option 3: GitHub Pages (Free but requires public repo)

1. Build the project:
   ```bash
   cd frontend
   npm run build
   ```
2. Push the `dist` folder to a `gh-pages` branch
3. Enable GitHub Pages in repo settings

## 🔧 Backend Setup for External Access

### Make Your Backend Accessible
Your backend needs to accept connections from external IPs:

1. **Update CORS settings** (if needed)
2. **Start backend with external access**:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

### Firewall Configuration
Make sure port 8000 is open:

**Windows:**
- Windows Defender Firewall → Allow an app → Add port 8000

**Mac:**
- System Preferences → Security & Privacy → Firewall → Options → Add port 8000

## 📱 Sharing with Friends

Once deployed, you'll get a URL like:
- Netlify: `https://your-app-name.netlify.app`
- Vercel: `https://your-app-name.vercel.app`

**Important Notes:**
1. **Keep your backend running** on your local machine
2. **Stay on the same network** (WiFi) as your friends for best performance
3. **Your computer must be on** for the backend to work

## 🔄 Quick Updates

To update your deployed frontend:
1. Make changes to your code
2. **Netlify**: Re-run `npm run build` and drag new `dist` folder
3. **Vercel/GitHub**: Just push to your repo (auto-deploys)

## 🌐 Alternative: Temporary Sharing

For quick demos without deployment:

### Option A: ngrok (Tunneling)
```bash
# Install ngrok
npm install -g ngrok

# Expose your backend
ngrok http 8000

# Use the ngrok URL in your frontend
```

### Option B: Local Network Sharing
1. Build frontend: `npm run build`
2. Serve locally: `npx serve dist -p 3000`
3. Share: `http://YOUR_IP_ADDRESS:3000`

## 🎯 Recommended Flow

1. **For quick demos**: Use ngrok or local network sharing
2. **For ongoing sharing**: Deploy to Netlify (easiest)
3. **For professional use**: Deploy to Vercel with custom domain

Your friends will be able to access the beautiful AnantaAI interface while the backend runs on your machine! 🚀
