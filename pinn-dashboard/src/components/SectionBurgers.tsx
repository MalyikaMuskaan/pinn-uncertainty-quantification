import Section from './Section'
import { MetricCard, PlotImage } from './UI'
import Line3D from './Line3D'
import { BURGERS, OCEAN, DARCY, img } from '../data'

export default function BurgersSection() {
  return (
    <Section
      id="burgers"
      phase="Phase 1"
      title="Burgers' equation baseline"
      subtitle="1-D viscous Burgers' equation solved with a physics-informed neural network. The sharp shock at x ≈ 0 is the key challenge — standard neural networks smear it, but the PDE residual loss keeps it crisp."
    >
      {/* Metric row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
        <MetricCard index={0} value={BURGERS.mse.toExponential(2)} label="MSE" highlight />
        <MetricCard index={1} value="4-layer" label="Architecture" sub="50 neurons / layer, tanh" />
        <MetricCard index={2} value="20 k" label="Collocation pts" />
        <MetricCard index={3} value="10 k" label="Adam epochs" />
      </div>

      {/* Images */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PlotImage index={4}
          src={img('burgers_pinn/outputs/heatmap.png')}
          alt="Burgers solution heatmap"
        />
        <PlotImage index={5}
          src={img('burgers_pinn/outputs/time_slices.png')}
          alt="Burgers time slices"
        />
      </div>

      {/* Real, auto-rotating 3D — this baseline's MSE next to the later
          phases that build on it, nothing fabricated. */}
      <div className="mt-4 mb-4">
        <Line3D
          points={[
            { x: 0, y: BURGERS.mse, label: 'Burgers\'' },
            { x: 1, y: OCEAN.mse, label: 'Ocean' },
            { x: 2, y: DARCY.mse, label: 'Darcy' },
          ]}
          logY
          title="MSE across PDE phases — 3D"
          xAxisLabel="phase"
          yAxisLabel="MSE (log)"
          color="#66c7ff"
          height={280}
        />
      </div>

      <div className="mt-4">
        <PlotImage index={6}
          src={img('burgers_pinn/outputs/loss_history.png')}
          alt="Burgers loss history"
        />
      </div>

      {/* Context note */}
      <div className="liquid-glass mt-6 px-5 py-4 rounded-2xl">
        <p className="text-white/50 text-sm leading-relaxed">
          The Burgers' PINN achieves MSE{' '}
          <span style={{ color: '#66c7ff' }}>{BURGERS.mse.toExponential(2)}</span> against
          the Crank-Nicolson finite-difference reference. Shock sharpness is preserved via
          the PDE residual penalty — no explicit shock-capturing scheme is needed.
          This baseline is the foundation for all UQ phases that follow.
        </p>
      </div>
    </Section>
  )
}
