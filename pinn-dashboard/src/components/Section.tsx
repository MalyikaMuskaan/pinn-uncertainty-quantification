import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { ReactNode } from 'react'

interface SectionProps {
  id: string
  phase: string
  title: string
  subtitle?: string
  children: ReactNode
}

const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  show: { opacity: 1, y: 0, transition: { duration: 0.75, ease: [0.22, 1, 0.36, 1] } },
}

export default function Section({ id, phase, title, subtitle, children }: SectionProps) {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref, { once: true, margin: '-80px 0px' })

  return (
    <motion.section
      ref={ref}
      id={id}
      initial="hidden"
      animate={inView ? 'show' : 'hidden'}
      variants={fadeUp}
      className="relative z-10 py-20 px-6"
    >
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <span className="phase-pill mb-3 inline-block">{phase}</span>
          <h2
            className="text-white mb-2"
            style={{
              fontFamily: '"Instrument Serif", serif',
              fontSize: 'clamp(1.9rem, 4vw, 3rem)',
              fontWeight: 400,
              lineHeight: 1.1,
            }}
          >
            {title}
          </h2>
          {subtitle && (
            <p className="text-white/45 text-sm max-w-2xl leading-relaxed">{subtitle}</p>
          )}
        </div>
        {children}
      </div>
    </motion.section>
  )
}
