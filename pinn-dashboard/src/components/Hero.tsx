import { motion } from 'framer-motion'
import { ArrowDown, Github } from 'lucide-react'
import { STATS } from '../data'

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12 } },
}
const item = {
  hidden: { opacity: 0, y: 32 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] } },
}

export default function Hero() {
  return (
    <section
      id="home"
      className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-24 pb-16"
    >
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="flex flex-col items-center text-center max-w-4xl mx-auto"
      >
        {/* Eyebrow */}
        <motion.p variants={item} className="section-label mb-6">
          Research 
        </motion.p>

        {/* Main heading */}
        <motion.h1
          variants={item}
          className="mb-3"
          style={{
            fontFamily: '"Instrument Serif", serif',
            fontSize: 'clamp(2.6rem, 7vw, 5.5rem)',
            lineHeight: 1.08,
            color: '#e8e0da',
            fontWeight: 400,
          }}
        >
          Seven dives into
        </motion.h1>
        <motion.h1
          variants={item}
          className="shiny-text mb-8"
          style={{
            fontFamily: '"Instrument Serif", serif',
            fontSize: 'clamp(2.6rem, 7vw, 5.5rem)',
            lineHeight: 1.08,
            fontWeight: 400,
          }}
        >
          uncertainty-aware physics
        </motion.h1>

        {/* Subtext */}
        <motion.p
          variants={item}
          className="text-white/55 max-w-2xl text-base leading-relaxed mb-10"
        >
          Physics-informed neural networks combined with three uncertainty quantification
          strategies — deep ensembles, Bayesian VI, and MC Dropout — applied to PDEs
          spanning 1-D Burgers' shocks, advection-diffusion transport, operator learning,
          2-D Darcy flow, and inverse parameter recovery.
        </motion.p>

        {/* CTA buttons */}
        <motion.div variants={item} className="flex flex-wrap justify-center gap-3 mb-14">
          <a
            href="#burgers"
            className="liquid-glass flex items-center gap-2 px-6 py-2.5 text-sm font-medium
                       text-white/80 hover:text-white transition-colors rounded-full"
          >
            Explore results
            <ArrowDown size={14} />
          </a>
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 px-6 py-2.5 text-sm font-medium rounded-full
                       text-white/50 hover:text-white/80 transition-colors border
                       border-white/10 hover:border-white/20"
          >
            <Github size={14} />
            View on GitHub
          </a>
        </motion.div>

        {/* Stat strip */}
        <motion.div
          variants={container}
          className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 w-full max-w-3xl"
        >
          {STATS.map((s) => (
            <motion.div
              key={s.label}
              variants={item}
              className="liquid-glass px-4 py-3 flex flex-col items-center gap-1 rounded-2xl"
            >
              <span
                style={{
                  fontFamily: '"Instrument Serif", serif',
                  fontSize: '1.55rem',
                  color: '#66c7ff',
                  lineHeight: 1,
                }}
              >
                {s.value}
              </span>
              <span className="text-white/45 text-center" style={{ fontSize: '0.65rem', letterSpacing: '0.06em' }}>
                {s.label}
              </span>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>
    </section>
  )
}
