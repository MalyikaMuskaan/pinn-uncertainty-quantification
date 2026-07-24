import Section from './Section'
import { PlotImage } from './UI'
import Line3D from './Line3D'
import { FAILURE, ABLATION_ENSEMBLE, ABLATION_WEIGHTING, img } from '../data'

export default function AblationsSection() {
  return (
    <Section
      id="ablations"
      phase="Phase 6 + 7"
      title="Failure analysis & ablations"
      subtitle="What breaks PINNs? Pushing ν into the ultra-low-viscosity regime exposes fundamental limitations. Ablations on ensemble size and loss weighting isolate which design choices matter."
    >
      {/* Failure analysis */}
      <h3
        className="section-label mb-4"
        style={{ fontSize: '0.75rem', color: 'rgba(102, 199, 255,0.95)' }}
      >
        Error vs viscosity ν
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <PlotImage index={0}
          src={img('burgers_pinn/outputs/failure_analysis/failure_error_vs_nu.png')}
          alt="Error vs nu degradation"
        />
        <PlotImage index={1}
          src={img('burgers_pinn/outputs/failure_analysis/failure_heatmap_comparison.png')}
          alt="Heatmap comparison across nu"
        />
      </div>

      {/* Real, auto-rotating 3D version — actual FAILURE data points,
          nothing fabricated, just rendered in 3D instead of flat. */}
      <div className="mb-6">
        <Line3D
          points={FAILURE.map((f) => ({ x: f.nu, y: f.rel_l2, label: `${(f.rel_l2 * 100).toFixed(1)}%` }))}
          logX
          logY
          title="Error vs viscosity ν — 3D"
          xAxisLabel="ν (log)"
          yAxisLabel="Rel-L2 error (log)"
        />
      </div>

      <div className="liquid-glass overflow-x-auto mb-8 rounded-2xl">
        <table className="data-table">
          <thead>
            <tr>
              <th>ν (label)</th>
              <th>ν value</th>
              <th>Rel-L2</th>
              <th>Shock behaviour</th>
            </tr>
          </thead>
          <tbody>
            {FAILURE.map((f, i) => (
              <tr key={i} className={i === 0 ? 'winner' : ''}>
                <td>{f.label}</td>
                <td>{f.nu.toExponential(3)}</td>
                <td
                  style={{
                    color: f.rel_l2 < 0.1 ? '#66c7ff' : 'rgba(232,224,218,0.55)',
                  }}
                >
                  {(f.rel_l2 * 100).toFixed(1)}%
                </td>
                <td className="text-white/45 text-xs">
                  {i === 0 ? 'Baseline — shock captured' :
                   i === 1 ? 'Steeper shock, some smearing' :
                   i === 2 ? 'Near-discontinuous, significant error' :
                             'Discontinuous — PINN fails'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Ensemble size ablation */}
      <h3 className="section-label mb-4" style={{ fontSize: '0.75rem', color: 'rgba(102, 199, 255,0.95)' }}>
        Ablation A — ensemble size M
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <PlotImage index={2}
          src={img('burgers_pinn/outputs/ablation/ensemble_size/ablation_ensemble_size.png')}
          alt="Ensemble size ablation"
        />
        <div className="liquid-glass overflow-auto rounded-2xl">
          <table className="data-table">
            <thead>
              <tr>
                <th>M</th>
                <th>ECE</th>
                <th>90% coverage</th>
                <th>MSE</th>
              </tr>
            </thead>
            <tbody>
              {ABLATION_ENSEMBLE.map((r) => (
                <tr key={r.M} className={r.M === 10 ? 'winner' : ''}>
                  <td>{r.M}</td>
                  <td>{r.ece.toFixed(4)}</td>
                  <td>{(r.coverage90 * 100).toFixed(1)}%</td>
                  <td>{r.mse.toExponential(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mb-6">
        <Line3D
          points={ABLATION_ENSEMBLE.map((r) => ({ x: r.M, y: r.mse, label: `M=${r.M}` }))}
          logY
          title="Ensemble size M vs MSE — 3D"
          xAxisLabel="M"
          yAxisLabel="MSE (log)"
          color="#a78bfa"
        />
      </div>

      {/* Loss weighting ablation */}
      <h3 className="section-label mb-4" style={{ fontSize: '0.75rem', color: 'rgba(102, 199, 255,0.95)' }}>
        Ablation B — loss weighting scheme
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <PlotImage index={3}
          src={img('burgers_pinn/outputs/ablation/loss_weighting/ablation_loss_weighting.png')}
          alt="Loss weighting ablation"
        />
        <div className="liquid-glass overflow-auto rounded-2xl">
          <table className="data-table">
            <thead>
              <tr>
                <th>Scheme</th>
                <th>MSE</th>
                <th>Rel-L2</th>
              </tr>
            </thead>
            <tbody>
              {ABLATION_WEIGHTING.map((r, i) => (
                <tr key={i} className={i === 0 ? 'winner' : ''}>
                  <td className="text-xs leading-tight">{r.scheme}</td>
                  <td>{r.mse.toExponential(2)}</td>
                  <td
                    style={{ color: i === 0 ? '#66c7ff' : i === 2 ? 'rgba(232,224,218,0.4)' : undefined }}
                  >
                    {(r.rel_l2 * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mb-4">
        <Line3D
          points={ABLATION_WEIGHTING.map((r, i) => ({ x: i, y: r.rel_l2, label: `${(r.rel_l2 * 100).toFixed(1)}%` }))}
          logY
          title="Loss weighting scheme vs Rel-L2 — 3D"
          xAxisLabel="scheme (a→c)"
          yAxisLabel="Rel-L2 (log)"
          color="#ffb14f"
        />
      </div>

      <div className="liquid-glass px-5 py-4 rounded-2xl">
        <p className="text-white/50 text-sm leading-relaxed">
          <span style={{ color: '#66c7ff' }}>Key takeaways:</span>{' '}
          (1) Accuracy improves monotonically with M but calibration does not — M=10 gives
          the best ECE (0.071). (2) λ_ic=10, λ_bc=10 (baseline weighting) is the clear
          winner — auto-balanced weighting collapses training with 51% rel-L2 error,
          likely because the auto-scale over-penalises the PDE residual at initialisation.
          (3) PINN error increases sharply below ν ≈ 1.6 × 10⁻³, where the shock
          half-width becomes sub-collocation-point.
        </p>
      </div>
    </Section>
  )
}
