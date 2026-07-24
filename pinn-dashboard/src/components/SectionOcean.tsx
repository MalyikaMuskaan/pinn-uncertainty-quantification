import Section from './Section'
import { MetricCard, PlotImage } from './UI'
import Line3D from './Line3D'
import { OCEAN, BURGERS, DARCY, img } from '../data'

export default function OceanSection() {
  return (
    <Section
      id="ocean"
      phase="Phase 3"
      title="Ocean / Red Sea advection-diffusion"
      subtitle="1-D advection-diffusion PDE modelling pollutant transport in an ocean current. The linear structure of this PDE fundamentally changes how uncertainty behaves compared to Burgers'."
    >
      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
        <MetricCard index={0} value={OCEAN.mse.toExponential(1)} label="MSE" highlight />
        <MetricCard index={1} value={OCEAN.ece.toFixed(4)} label="ECE" />
        <MetricCard index={2} value={(OCEAN.coverage90 * 100).toFixed(1) + '%'} label="90% Coverage" />
        <MetricCard index={3} value={OCEAN.pde_loss.toExponential(2)} label="PDE loss" sub="vs 9e-4 Burgers'" />
      </div>

      {/* Real, auto-rotating 3D — MSE across the three PDE phases so far
          (each already reported elsewhere on the site), nothing fabricated. */}
      <div className="mb-4">
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

      {/* Images — heatmap + ensemble */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <PlotImage index={4}
          src={img('ocean_pinn/outputs/heatmap.png')}
          alt="Ocean PINN heatmap"
        />
        <PlotImage index={5}
          src={img('ocean_pinn/outputs/ensemble/ensemble_mean_heatmap.png')}
          alt="Ocean ensemble mean"
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <PlotImage index={6}
          src={img('ocean_pinn/outputs/ensemble/ensemble_std_heatmap.png')}
          alt="Ocean ensemble std (uncertainty)"
        />
        <PlotImage index={7}
          src={img('ocean_pinn/outputs/ensemble/ensemble_calibration.png')}
          alt="Ocean ensemble calibration"
        />
      </div>

      {/* Finding callout */}
      <div className="liquid-glass px-5 py-5 grid grid-cols-1 md:grid-cols-2 gap-6 rounded-2xl">
        <div>
          <p className="section-label mb-2">Key finding — linear vs nonlinear UQ</p>
          <p className="text-white/55 text-sm leading-relaxed">
            Because the advection-diffusion equation is <em>linear</em>, all 10 ensemble
            members converge to essentially the same solution — diversity is suppressed.
            Deep Ensembles derive their calibrated uncertainty from members landing in
            <em> different</em> loss basins. With a convex problem, this mechanism fails:
            the ensemble under-covers (76.9%) despite accurate predictions.
          </p>
        </div>
        <div>
          <p className="section-label mb-2">Uncertainty structure</p>
          <p className="text-white/55 text-sm leading-relaxed">
            Unlike Burgers' (sharp spike at the shock), uncertainty here forms a
            broad diagonal band following the advecting Gaussian pulse — highest at
            the leading edge, lowest at the pulse peak. The characteristic direction
            (dx/dt = v = 1) is a local <em>minimum</em> of uncertainty, exactly where
            the PDE information is strongest.
          </p>
        </div>
      </div>
    </Section>
  )
}
