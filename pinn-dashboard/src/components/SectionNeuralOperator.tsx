import Section from './Section'
import { MetricCard, PlotImage } from './UI'
import Line3D from './Line3D'
import { FNO, img } from '../data'

export default function NeuralOperatorSection() {
  return (
    <Section
      id="neural-operator"
      phase="Phase 5a"
      title="Fourier Neural Operator"
      subtitle="FNO learns the solution operator: one training run maps any initial condition u₀(x) to its full trajectory u(x,t). Compare to a PINN that must retrain from scratch per IC."
    >
      {/* Key metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
        <MetricCard index={0}
          value={(FNO.fno_rel_l2_mean * 100).toFixed(2) + '%'}
          label="FNO rel-L2"
          highlight
        />
        <MetricCard index={1}
          value={(FNO.pinn_rel_l2_mean * 100).toFixed(1) + '%'}
          label="PINN rel-L2"
          sub="4.7× worse"
        />
        <MetricCard index={2}
          value={FNO.fno_infer_ms.toFixed(1) + ' ms'}
          label="FNO infer/IC"
          sub="post-JIT warmup"
        />
        <MetricCard index={3}
          value={FNO.fno_train_s + ' s'}
          label="FNO train"
          sub="one-time fixed cost"
        />
      </div>

      {/* Comparison table */}
      <div className="liquid-glass overflow-x-auto mb-8 rounded-2xl">
        <table className="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>FNO</th>
              <th>PINN (per-IC)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Rel L2 mean</td>
              <td style={{ color: '#66c7ff' }}><strong>{(FNO.fno_rel_l2_mean * 100).toFixed(2)}%</strong></td>
              <td>{(FNO.pinn_rel_l2_mean * 100).toFixed(2)}%</td>
            </tr>
            <tr>
              <td>Rel L2 std</td>
              <td>{(FNO.fno_rel_l2_std * 100).toFixed(2)}%</td>
              <td>{(FNO.pinn_rel_l2_std * 100).toFixed(2)}%</td>
            </tr>
            <tr>
              <td>Training cost</td>
              <td style={{ color: '#66c7ff' }}>{FNO.fno_train_s} s (one-time)</td>
              <td>{FNO.pinn_train_s_per_ic.toFixed(1)} s / IC</td>
            </tr>
            <tr>
              <td>Inference / IC</td>
              <td>~{FNO.fno_infer_ms} ms</td>
              <td>{FNO.pinn_infer_ms} ms</td>
            </tr>
            <tr>
              <td>Generalises to new ICs?</td>
              <td style={{ color: '#66c7ff' }}>Yes</td>
              <td>No — retrain required</td>
            </tr>
            <tr>
              <td>Physics-constrained?</td>
              <td>No</td>
              <td style={{ color: '#66c7ff' }}>Yes</td>
            </tr>
            <tr>
              <td>Parameters</td>
              <td colSpan={2}>{FNO.params.toLocaleString()} (FNO)</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Real, auto-rotating 3D versions — actual FNO vs PINN numbers above,
          nothing fabricated. The cost-vs-instances chart is the same
          break-even arithmetic described in the paragraph below (76s flat
          vs 21.6s × n), just plotted instead of only stated. */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <Line3D
          points={[
            { x: 0, y: FNO.fno_rel_l2_mean * 100, label: 'FNO' },
            { x: 1, y: FNO.pinn_rel_l2_mean * 100, label: 'PINN' },
          ]}
          title="Rel-L2 error — 3D"
          xAxisLabel="method"
          yAxisLabel="rel-L2 %"
          color="#66c7ff"
          height={260}
        />
        <Line3D
          points={[
            { x: 0, y: FNO.fno_infer_ms, label: 'FNO' },
            { x: 1, y: FNO.pinn_infer_ms, label: 'PINN' },
          ]}
          title="Inference time / IC — 3D"
          xAxisLabel="method"
          yAxisLabel="ms"
          color="#a78bfa"
          height={260}
        />
        <Line3D
          series={[
            {
              name: 'FNO',
              color: '#66c7ff',
              points: Array.from({ length: 8 }, (_, i) => ({ x: i + 1, y: FNO.fno_train_s })),
            },
            {
              name: 'PINN',
              color: '#ffb14f',
              points: Array.from({ length: 8 }, (_, i) => ({ x: i + 1, y: FNO.pinn_train_s_per_ic * (i + 1) })),
            },
          ]}
          title="Cost vs # ICs — 3D"
          xAxisLabel="# instances"
          yAxisLabel="total seconds"
          height={260}
        />
      </div>

      {/* Comparison plots */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        {[0, 1, 2].map((i) => (
          <PlotImage index={4}
            key={i}
            src={img(`neural_operator/outputs/plots/comparison_${String(i).padStart(3, '0')}.png`)}
            alt={`FNO vs GT — test IC ${i}`}
          />
        ))}
      </div>

      <PlotImage index={5}
        src={img('neural_operator/outputs/plots/summary_table.png')}
        alt="FNO vs PINN summary table"
      />

      <div className="liquid-glass mt-6 px-5 py-4 rounded-2xl">
        <p className="text-white/50 text-sm leading-relaxed">
          <span style={{ color: '#66c7ff' }}>Break-even at ~4 ICs:</span> FNO's 76 s one-time
          training cost is recovered after just 4 PINN retrains (at ~22 s each). For any workload
          requiring ≥ 5 distinct initial conditions, the FNO is strictly cheaper in wall time and
          ~8 000× faster per inference query once trained. The accuracy gap (6.98% vs 32.75%) is
          partly because the PINN's Adam + L-BFGS schedule was tuned for the single canonical
          -sin(πx) IC, not for arbitrary random sinusoidal ICs.
        </p>
      </div>
    </Section>
  )
}
