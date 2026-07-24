import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X } from 'lucide-react'
import { useHashRoute } from '../hooks/useHashRoute'

const NAV_LINKS = [
  { label: 'Home',           href: '#home' },
  { label: 'Burgers\'',      href: '#burgers' },
  { label: 'UQ Comparison',  href: '#uq' },
  { label: 'Ocean',          href: '#ocean' },
  { label: 'Inverse',        href: '#inverse' },
  { label: 'Neural Operator',href: '#neural-operator' },
  { label: '2D Darcy',       href: '#darcy' },
  { label: 'Ablations',      href: '#ablations' },
]

export default function Nav() {
  const [scrolled, setScrolled] = useState(false)
  const [hidden, setHidden] = useState(false)
  const [open, setOpen] = useState(false)
  const { section, navigate } = useHashRoute()

  useEffect(() => {
    let lastY = window.scrollY
    const handler = () => {
      const y = window.scrollY
      setScrolled(y > 40)
      // Hide while actively scrolling down past the top area; reveal again
      // as soon as the user scrolls up, so the nav never just sits fixed
      // over content you're trying to read — it gets out of the way.
      if (y > lastY && y > 120) {
        setHidden(true)
      } else {
        setHidden(false)
      }
      lastY = y
    }
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [])

  function goTo(href: string) {
    navigate(href)
    setOpen(false)
  }

  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex justify-center px-4 pt-4">
      <motion.nav
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: hidden && !open ? -100 : 0 }}
        transition={{ duration: 0.35, ease: 'easeInOut' }}
        className={`nav-glass w-full max-w-5xl px-5 py-2.5 flex items-center justify-between transition-all duration-300 rounded-full ${
          scrolled ? 'shadow-lg' : ''
        }`}
      >
        {/* Logo */}
        <button onClick={() => goTo('#home')} className="flex items-center gap-2.5 shrink-0">
          <span
            className="font-serif text-lg font-semibold tracking-wide"
            style={{ color: '#e8e0da', fontFamily: '"Instrument Serif", serif' }}
          >
            Physics-informed neural networks
          </span>
        </button>

        {/* Desktop links — with a sliding highlight pill behind the active link */}
        <div className="hidden lg:flex items-center gap-1 relative">
          {NAV_LINKS.map((l) => {
            const key = l.href.replace('#', '')
            const isActive = section === key
            return (
              <button
                key={l.href}
                onClick={() => goTo(l.href)}
                className={`relative px-3 py-1.5 rounded-full text-xs font-medium tracking-wide
                           transition-colors duration-200
                           ${isActive ? 'text-white' : 'text-white/60 hover:text-white hover:bg-white/5'}`}
              >
                {isActive && (
                  <motion.span
                    layoutId="nav-active-pill"
                    className="absolute inset-0 rounded-full"
                    style={{ background: 'rgba(102,199,255,0.16)', border: '1px solid rgba(102,199,255,0.35)' }}
                    transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                  />
                )}
                <span className="relative z-10">{l.label}</span>
              </button>
            )
          })}
        </div>

        {/* Mobile hamburger */}
        <button
          className="lg:hidden text-white/70 hover:text-white transition-colors"
          onClick={() => setOpen(!open)}
          aria-label="Toggle menu"
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </motion.nav>

      {/* Mobile drawer — slides down, links slide/fade in with a stagger */}
      <AnimatePresence>
        {open && (
          <motion.div
            key="drawer"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.22 }}
            className="nav-glass absolute top-full left-4 right-4 mt-2 p-4 flex flex-col gap-1 rounded-2xl"
            style={{ top: '72px' }}
          >
            {NAV_LINKS.map((l, i) => {
              const isActive = section === l.href.replace('#', '')
              return (
                <motion.button
                  key={l.href}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03, duration: 0.25 }}
                  onClick={() => goTo(l.href)}
                  className={`text-left px-4 py-2.5 rounded-lg text-sm transition-colors
                             ${isActive ? 'text-white bg-white/10' : 'text-white/70 hover:text-white hover:bg-white/5'}`}
                >
                  {l.label}
                </motion.button>
              )
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
