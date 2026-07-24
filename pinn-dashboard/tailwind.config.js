/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        serif: ['"Instrument Serif"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        bg: '#050403',
        surface: '#0a0605',
        border: 'rgba(255,143,107,0.12)',
        accent: '#ff8f6b',
        'accent-dim': '#cc6b49',
        glass: 'rgba(255,255,255,0.02)',
      },
      backgroundImage: {
        'shiny-text': 'linear-gradient(90deg, #ff9c85 0%, #fff 45%, #fff 55%, #ff9c85 100%)',
      },
    },
  },
  plugins: [],
}
