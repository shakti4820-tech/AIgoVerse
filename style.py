import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Core Typography & Variables */
:root {
    --bg-primary: #030712;
    --bg-secondary: #0b0f19;
    --border-color: rgba(31, 41, 55, 0.4);
    --glow-cyan: rgba(0, 212, 255, 0.15);
    --cyan: #00d4ff;
    --purple: #a855f7;
    --emerald: #06d6a0;
    --gold: #ffd166;
    --rose: #ff6b6b;
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
}

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-primary);
}

.main {
    background: var(--bg-primary);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}

/* Stunning Glassmorphic App Background */
.stApp {
    background: radial-gradient(circle at 80% 20%, rgba(168, 85, 247, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 10% 80%, rgba(0, 212, 255, 0.08) 0%, transparent 40%),
                var(--bg-primary);
    background-attachment: fixed;
}

/* Premium Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: #1f2937;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--cyan);
}

/* Futuristic Hero Banner with Glowing Border */
.hero-banner {
    background: linear-gradient(135deg, rgba(13, 27, 42, 0.8), rgba(26, 45, 74, 0.8));
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 20px;
    padding: 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px 0 rgba(0, 212, 255, 0.05),
                inset 0 0 20px rgba(0, 212, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    animation: fadeIn 0.8s ease-out;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(0, 212, 255, 0.15) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--cyan);
    margin: 0 0 0.6rem 0;
    text-shadow: 0 0 25px rgba(0, 212, 255, 0.35);
    letter-spacing: -0.02em;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 300;
    line-height: 1.6;
    margin: 0;
}

/* Glassmorphic Metric Cards */
div[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-top: 3px solid var(--cyan) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(0, 212, 255, 0.4) !important;
    box-shadow: 0 10px 30px rgba(0, 212, 255, 0.08) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-family: 'Space Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.05) !important;
}

/* Custom Interactive Learning Cards */
.feature-card {
    background: linear-gradient(135deg, rgba(15, 28, 46, 0.4), rgba(22, 32, 48, 0.4));
    border: 1px solid rgba(30, 58, 95, 0.4);
    border-left: 4px solid var(--cyan);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.feature-card:hover {
    transform: translateX(4px);
    border-color: rgba(0, 212, 255, 0.3);
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.05);
}
.feature-title {
    font-family: 'Space Mono', monospace;
    color: var(--cyan);
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.feature-desc {
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.7;
}

/* Beautiful Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #090d1f 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.3) !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label, 
section[data-testid="stSidebar"] .stSlider label, 
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #94a3b8 !important;
    margin-bottom: 0.3rem !important;
}

/* Elegant Dynamic Glass Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13, 21, 34, 0.6) !important;
    border: 1px solid rgba(30, 58, 95, 0.4) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 0.3rem 0.5rem 0 0.5rem !important;
    gap: 0.4rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--cyan) !important;
    background: rgba(255, 255, 255, 0.02) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 212, 255, 0.08) !important;
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
    font-weight: 700 !important;
}

/* Custom Buttons & Inputs */
.stButton button {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(0, 144, 204, 0.15)) !important;
    border: 1px solid rgba(0, 212, 255, 0.4) !important;
    color: var(--cyan) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.05) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100%;
}
.stButton button:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.3), rgba(0, 144, 204, 0.3)) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 25px rgba(0, 212, 255, 0.2) !important;
    transform: translateY(-1px) !important;
}
.stButton button:active {
    transform: translateY(1px) !important;
}

/* Premium Tags */
.algo-tag {
    display: inline-block;
    background: rgba(0, 212, 255, 0.08);
    color: var(--cyan);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin: 0.25rem;
    box-shadow: 0 2px 8px rgba(0, 212, 255, 0.02);
    transition: all 0.2s ease;
}
.algo-tag:hover {
    background: rgba(0, 212, 255, 0.15);
    border-color: var(--cyan);
}

/* Interactive Info/Warning Blocks */
.stInfo, .stSuccess, .stWarning, .stError {
    background: rgba(11, 15, 25, 0.8) !important;
    border: 1px solid var(--border-color) !important;
    border-left: 4px solid var(--cyan) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
    backdrop-filter: blur(8px) !important;
    padding: 1rem !important;
}
.stInfo { border-left-color: var(--cyan) !important; }
.stSuccess { border-left-color: var(--emerald) !important; }
.stWarning { border-left-color: var(--gold) !important; }
.stError { border-left-color: var(--rose) !important; }

/* Custom Home Section Cards */
.home-card {
    background: linear-gradient(135deg, rgba(15, 28, 46, 0.4), rgba(22, 32, 48, 0.4));
    border: 1px solid rgba(30, 58, 95, 0.4);
    border-top: 4px solid var(--cyan);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}
.home-card:hover {
    transform: translateY(-6px);
    border-color: rgba(0, 212, 255, 0.3);
    box-shadow: 0 15px 35px rgba(0, 212, 255, 0.08);
}
.home-card-icon {
    font-size: 2.8rem;
    margin-bottom: 0.8rem;
    filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.1));
}
.home-card-title {
    font-family: 'Space Mono', monospace;
    color: var(--cyan);
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    letter-spacing: -0.01em;
}
.home-card-desc {
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.7;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-header {
    font-family: 'Space Mono', monospace;
    color: var(--cyan);
    font-size: 1.05rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid rgba(30, 58, 95, 0.4);
    padding-bottom: 0.6rem;
    margin: 2rem 0 1.2rem 0;
    text-shadow: 0 0 15px rgba(0, 212, 255, 0.15);
}

/* Video Wrapper */
.video-container {
    position: relative;
    width: 100%;
    padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
    height: 0;
    overflow: hidden;
    border-radius: 12px;
    border: 1px solid rgba(0, 212, 255, 0.2);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    margin: 1rem 0;
}
.video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 0;
}

/* Custom GFG & Doc Cards */
.link-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}
.gfg-link-card {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: var(--cyan) !important;
    text-decoration: none !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    transition: all 0.2s ease;
}
.gfg-link-card:hover {
    background: rgba(0, 212, 255, 0.12);
    border-color: var(--cyan);
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.1);
}
</style>
"""

def inject_style():
    st.markdown(CSS, unsafe_allow_html=True)
