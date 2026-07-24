import { AnimatePresence, motion } from 'framer-motion'
import FluidBackground from './components/FluidBackground'
import Nav from './components/Nav'
import Hero from './components/Hero'
import BurgersSection from './components/SectionBurgers'
import UQSection from './components/SectionUQ'
import OceanSection from './components/SectionOcean'
import InverseSection from './components/SectionInverse'
import NeuralOperatorSection from './components/SectionNeuralOperator'
import DarcySection from './components/SectionDarcy'
import AblationsSection from './components/SectionAblations'
import Footer from './components/Footer'
import { SectionDivider } from './components/UI'
import { useHashRoute } from './hooks/useHashRoute'

const SECTION_MAP: Record<string, JSX.Element> = {
  home: <Hero />,
  burgers: <BurgersSection />,
  uq: <UQSection />,
  ocean: <OceanSection />,
  inverse: <InverseSection />,
  'neural-operator': <NeuralOperatorSection />,
  darcy: <DarcySection />,
  ablations: <AblationsSection />,
}

export default function App() {
  const { section } = useHashRoute()
  const activeElement = SECTION_MAP[section] ?? <Hero />

  return (
    <div className="relative min-h-screen" style={{ background: '#050403' }}>
      <FluidBackground />

      <div
        className="fixed inset-0 z-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse 80% 60% at 50% 80%, rgba(5,4,3,0.72) 0%, transparent 70%),
            radial-gradient(ellipse 100% 40% at 50% 0%, rgba(5,4,3,0.8) 0%, transparent 60%)
          `,
        }}
      />

      <div className="relative z-10">
        <Nav />

        <AnimatePresence mode="wait">
          <motion.div
            key={section}
            initial={{ opacity: 0, y: 24, filter: 'blur(6px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            exit={{ opacity: 0, y: -16, filter: 'blur(6px)' }}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
          >
            {activeElement}
          </motion.div>
        </AnimatePresence>

        <SectionDivider />
        <Footer />
      </div>
    </div>
  )
}
