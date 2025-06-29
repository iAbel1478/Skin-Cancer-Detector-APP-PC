const sharp = require('sharp');

// Create a beautiful gradient shield favicon with enhanced purple gradients
const svg = `
<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="shieldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4F46E5;stop-opacity:1" />
      <stop offset="25%" style="stop-color:#6366F1;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#8B5CF6;stop-opacity:1" />
      <stop offset="75%" style="stop-color:#A855F7;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#7C3AED;stop-opacity:1" />
    </linearGradient>
    
    <radialGradient id="highlightGradient" cx="30%" cy="30%" r="70%">
      <stop offset="0%" style="stop-color:#C084FC;stop-opacity:0.9" />
      <stop offset="50%" style="stop-color:#A78BFA;stop-opacity:0.5" />
      <stop offset="100%" style="stop-color:#7C3AED;stop-opacity:0" />
    </radialGradient>
    
    <linearGradient id="crossGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#FFFFFF;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#F8FAFC;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#F3F4F6;stop-opacity:1" />
    </linearGradient>
    
    <radialGradient id="purpleGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#E879F9;stop-opacity:0.4" />
      <stop offset="100%" style="stop-color:#A855F7;stop-opacity:0" />
    </radialGradient>
  </defs>
  
  <!-- Shield shape with enhanced gradient -->
  <path d="M16 2L28 8V16C28 22.627 22.627 28 16 28C9.373 28 4 22.627 4 16V8L16 2Z" 
        fill="url(#shieldGradient)" stroke="#1E1B4B" stroke-width="1"/>
  
  <!-- Purple glow overlay -->
  <path d="M16 2L28 8V16C28 22.627 22.627 28 16 28C9.373 28 4 22.627 4 16V8L16 2Z" 
        fill="url(#purpleGlow)" opacity="0.6"/>
  
  <!-- Highlight overlay -->
  <path d="M16 2L28 8V16C28 22.627 22.627 28 16 28C9.373 28 4 22.627 4 16V8L16 2Z" 
        fill="url(#highlightGradient)" opacity="0.4"/>
  
  <!-- Medical cross with gradient -->
  <rect x="14" y="10" width="4" height="12" fill="url(#crossGradient)" rx="1"/>
  <rect x="10" y="14" width="12" height="4" fill="url(#crossGradient)" rx="1"/>
  
  <!-- Enhanced skin texture dots with gradients -->
  <circle cx="8" cy="8" r="1.5" fill="url(#crossGradient)" opacity="0.8"/>
  <circle cx="24" cy="8" r="1.5" fill="url(#crossGradient)" opacity="0.8"/>
  <circle cx="8" cy="24" r="1.5" fill="url(#crossGradient)" opacity="0.8"/>
  <circle cx="24" cy="24" r="1.5" fill="url(#crossGradient)" opacity="0.8"/>
  
  <!-- Additional highlight dots -->
  <circle cx="12" cy="12" r="0.8" fill="white" opacity="0.6"/>
  <circle cx="20" cy="20" r="0.8" fill="white" opacity="0.6"/>
</svg>
`;

sharp(Buffer.from(svg))
  .png()
  .toFile('assets/images/favicon.png')
  .then(() => console.log('✅ Beautiful enhanced purple gradient shield favicon created!'))
  .catch(err => console.error('Error:', err)); 