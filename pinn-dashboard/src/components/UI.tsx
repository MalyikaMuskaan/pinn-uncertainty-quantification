import { useRef } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

/* ------------------------------------------------------------------ */
/*  Shared floating-card entrance + idle-drift + hover behaviour.       */
/*  Cards enter staggered, tilted in 3D (as if drifting up out of      */
/*  the water toward the viewer), settle into a small resting tilt,    */
/*  then idle-bob gently. On hover they lift and straighten.           */
/* ------------------------------------------------------------------ */

// A small, deterministic per-card resting tilt so the cluster reads as
// "scattered floating" rather than a perfectly flat grid.
function restTilt(index: number) {
  const pattern = [-3, 2, -1.5, 2.5, -2, 1.5, -2.5, 3]
  return pattern[index % pattern.length]
}

interface FloatingProps {
  index?: number
  className?: string
  flat?: boolean
  tilt3d?: boolean
  children: React.ReactNode
}

function Floating({ index = 0, className = '', flat = false, tilt3d = false, children }: FloatingProps) {
  const tilt = flat ? 0 : restTilt(index)
  const idleDuration = 5 + (index % 4) * 0.7 // 5–7.1s, varies per card so they don't sync

  // Cursor-driven 3D tilt: card sits perfectly flat/straight at rest, and
  // smoothly rotates to follow the pointer while hovered — springs back to
  // flat the instant the cursor leaves.
  const ref = useRef<HTMLDivElement>(null)
  const rawRotateX = useMotionValue(0)
  const rawRotateY = useMotionValue(0)
  const rotateX = useSpring(rawRotateX, { stiffness: 220, damping: 22 })
  const rotateY = useSpring(rawRotateY, { stiffness: 220, damping: 22 })

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!tilt3d || !ref.current) return
    const rect = ref.current.getBoundingClientRect()
    const px = (e.clientX - rect.left) / rect.width - 0.5
    const py = (e.clientY - rect.top) / rect.height - 0.5
    rawRotateY.set(px * 10)
    rawRotateX.set(-py * 10)
  }
  function handleMouseLeave() {
    rawRotateX.set(0)
    rawRotateY.set(0)
  }

  return (
    <motion.div
      className={`self-start ${className}`}
      style={{ transformStyle: 'preserve-3d', perspective: tilt3d ? 1000 : undefined }}
      initial={{ opacity: 0, y: 36, scale: 0.86, rotateX: flat ? 0 : -14, rotateZ: tilt }}
      whileInView={{ opacity: 1, y: 0, scale: 1, rotateX: 0, rotateZ: tilt }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.7, delay: index * 0.09, ease: [0.16, 1, 0.3, 1] }}
      whileHover={!tilt3d ? { y: -5, rotateZ: 0, rotateX: 0, scale: 1.015, transition: { duration: 0.25 } } : undefined}
    >
      <motion.div
        ref={ref}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={tilt3d ? { rotateX, rotateY, transformStyle: 'preserve-3d' } : undefined}
        whileHover={tilt3d ? { scale: 1.02, transition: { duration: 0.2 } } : undefined}
        animate={flat ? { y: [0, -4, 0] } : { y: [0, -4, 0], rotateZ: [tilt, tilt + 1, tilt] }}
        transition={{ duration: idleDuration, repeat: Infinity, ease: 'easeInOut', delay: index * 0.15 }}
      >
        {children}
      </motion.div>
    </motion.div>
  )
}

interface MetricCardProps {
  value: string
  label: string
  sub?: string
  highlight?: boolean
  index?: number
}

export function MetricCard({ value, label, sub, highlight, index = 0 }: MetricCardProps) {
  return (
    <Floating index={index} flat>
      <div
        className={`liquid-glass px-5 py-4 flex flex-col gap-1 rounded-2xl ${highlight ? 'ring-1 ring-accent/30' : ''}`}
        style={{ '--tw-ring-color': 'rgba(102,199,255,0.3)' } as React.CSSProperties}
      >
        <span className="metric-value">{value}</span>
        <span className="text-white/55 text-xs font-medium tracking-wide uppercase">{label}</span>
        {sub && <span className="text-white/30 text-xs">{sub}</span>}
      </div>
    </Floating>
  )
}

interface PlotImageProps {
  src: string
  alt: string
  className?: string
  index?: number
}

export function PlotImage({ src, alt, className = '', index = 0 }: PlotImageProps) {
  return (
    <Floating index={index} flat tilt3d>
      <div className={`liquid-glass overflow-hidden rounded-2xl flex justify-center items-start ${className}`}>
        <img
          src={src}
          alt={alt}
          className="plot-img"
          onError={(e) => {
            const parent = (e.target as HTMLImageElement).parentElement
            if (parent) {
              parent.innerHTML = `<div class="missing-state">⚠ ${alt} — not yet generated</div>`
            }
          }}
        />
      </div>
    </Floating>
  )
}

export function MissingState({ label }: { label: string }) {
  return <div className="missing-state">⚠ {label} — not yet available</div>
}

export function SectionDivider() {
  return (
    <div
      className="my-0 h-px max-w-5xl mx-auto"
      style={{ background: 'linear-gradient(90deg, transparent, rgba(102,199,255,0.15), transparent)' }}
    />
  )
}
