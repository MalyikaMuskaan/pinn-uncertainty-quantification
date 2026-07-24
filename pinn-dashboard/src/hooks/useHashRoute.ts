import { useState, useEffect, useCallback } from 'react'

const VALID_SECTIONS = [
  'home',
  'burgers',
  'uq',
  'ocean',
  'inverse',
  'neural-operator',
  'darcy',
  'ablations',
]

function getHashSection() {
  const h = window.location.hash.replace('#', '')
  return VALID_SECTIONS.includes(h) ? h : 'home'
}

/**
 * Drop into: pinn-dashboard/src/hooks/useHashRoute.ts
 *
 * Turns the dashboard into page-switching navigation instead of one long
 * scrolling page: clicking a nav link shows ONLY that section, updates the
 * URL hash (so refresh/direct links work), and scrolls to the top.
 */
export function useHashRoute() {
  const [section, setSection] = useState(getHashSection)

  useEffect(() => {
    const onHashChange = () => {
      setSection(getHashSection())
      window.scrollTo({ top: 0, behavior: 'auto' })
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const navigate = useCallback((href: string) => {
    const target = href.replace('#', '')
    if (window.location.hash === href) {
      // Same link clicked again — still scroll to top for consistency.
      window.scrollTo({ top: 0, behavior: 'auto' })
      setSection(target)
      return
    }
    window.location.hash = href
  }, [])

  return { section, navigate }
}
