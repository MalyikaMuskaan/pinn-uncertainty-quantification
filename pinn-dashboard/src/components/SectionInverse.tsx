import Section from './Section'
import { MetricCard, PlotImage } from './UI'
import Line3D from './Line3D'
import { INVERSE, img } from '../data'

export default function InverseSection() {
  const nuTrue = INVERSE.nu_true

  return (
    <Section
      id="inverse"
      phase="Phase 4"
      title="Inverse problem — ν recovery"
      subtitle="Recover viscosity ν from sparse, noisy sensor observations using a 10-member ensemble. True ν = 1/π × 10⁻² ≈ 3.183 × 10⁻³."
    >
      {/* Stat row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
        <MetricCard index={0} value={nuTrue.toExponential(3)} label="True ν" />
        <MetricCard index={1} value="17.6%" label="Best error" sub="20% noise, 100 sensors" highlight />
        <MetricCard index={2} value="240.8%" label="Worst error" sub="2% noise, 20 sensors" />
        <MetricCard index={3} value="9" label="Sweep conditions" sub="3 noise × 3 sensor counts" />
      </div>

      {/* ν convergence + uncertainty plots */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <PlotImage index={4}
          src={img('inverse_problem/outputs/nu_convergence.png')}
          alt="ν convergence over training"
        />
        <PlotImage index={5}
          src={img('inverse_problem/outputs/robustness/robustness_error_vs_sensors.png')}
          alt="Error vs sensor count"
        />
      </div>

      {/* Real, auto-rotating 3D version — actual robustness sweep data, all
          three noise levels as separate colored lines in ONE chart (matches
          the static image's single-chart-with-legend layout, and keeps this
          page to a single extra WebGL canvas instead of three). */}
      <div className="mb-8">
        <Line3D
          series={(['0.5%', '1.0%', '2.0%'] as const).map((noiseLevel, i) => ({
            name: `noise ${noiseLevel}`,
            color: ['#66c7ff', '#a78bfa', '#ffb14f'][i],
            points: INVERSE.robustness
              .filter((r) => r.noise === noiseLevel)
              .map((r) => ({ x: r.sensors, y: r.errPct, label: `${r.errPct.toFixed(0)}%` })),
          }))}
          title="Error vs sensor density, by noise level — 3D"
          xAxisLabel="sensors"
          yAxisLabel="error %"
          height={340}
        />
      </div>

      {/* Robustness sweep table */}
      <div className="liquid-glass overflow-x-auto mb-8 rounded-2xl">
        <table className="data-table">
          <thead>
            <tr>
              <th>Noise</th>
              <th>Sensors</th>
              <th>ν mean</th>
              <th>Error %</th>
              <th>True in 90% CI?</th>
            </tr>
          </thead>
          <tbody>
            {INVERSE.robustness.map((r, i) => (
              <tr key={i} className={r.inCI ? 'winner' : ''}>
                <td>{r.noise}</td>
                <td>{r.sensors}</td>
                <td>—</td>
                <td
                  style={{
                    color:
                      r.errPct < 50 ? '#66c7ff' :
                      r.errPct < 120 ? '#e8e0da' :
                      'rgba(232,224,218,0.45)',
                  }}
                >
                  {r.errPct.toFixed(1)}%
                </td>
                <td>{r.inCI ? '✓ yes' : '✗ no'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Honest bug note */}
      <div className="liquid-glass px-5 py-5 grid grid-cols-1 md:grid-cols-2 gap-6 rounded-2xl">
        <div>
          <p className="section-label mb-2">Bug found & fixed</p>
          <p className="text-white/55 text-sm leading-relaxed">
            The initial implementation used a gradient-clipping hyperparameter tuned for the
            forward PINN, which severely stunted ν convergence — errors of 400–700% persisted
            even with clean data. After diagnosing the issue (tracked in
            <code className="text-white/40 text-xs mx-1">outputs/diagnostics/</code>),
            the clipping threshold was adjusted and errors fell to the 17–240% range seen here.
          </p>
        </div>
        <div>
          <p className="section-label mb-2">Why errors remain high</p>
          <p className="text-white/55 text-sm leading-relaxed">
            The true ν (3.18 × 10⁻³) is at the sharp-shock regime boundary. The loss surface
            for ν is highly non-convex near the shock: small ν changes cause large solution
            differences, making gradient-based recovery sensitive to noise. More sensors
            (100) consistently reduce errors by constraining the shock position better.
          </p>
        </div>
      </div>
    </Section>
  )
}
