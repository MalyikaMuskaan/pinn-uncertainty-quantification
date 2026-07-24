import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

/* ------------------------------------------------------------------ */
/*  Real, interactive rotating 3-D line/scatter chart — for converting  */
/*  flat 2-D line plots (error vs viscosity, ablation curves, etc.)     */
/*  into the floor-grid + connected-points style you referenced.       */
/*  Uses your REAL data points — nothing fabricated — just rendered    */
/*  as a live 3-D scene instead of a flat matplotlib line.              */
/* ------------------------------------------------------------------ */

interface Point3DSpec {
  x: number       // raw x value (e.g. viscosity ν)
  y: number       // raw y value (e.g. rel-L2 error)
  label?: string  // shown next to the point marker
}

interface Series3DSpec {
  points: Point3DSpec[]
  color: string
  name?: string // shown as a small legend sprite
}

interface Line3DProps {
  points?: Point3DSpec[]       // single-series shorthand
  series?: Series3DSpec[]      // multi-series (e.g. FNO vs PINN)
  title?: string
  xAxisLabel?: string
  yAxisLabel?: string
  logX?: boolean
  logY?: boolean
  color?: string
  height?: number
}

function makeLabelSprite(text: string, color = 'rgba(232,224,218,0.9)') {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!
  const fontSize = 42
  ctx.font = `500 ${fontSize}px Inter, sans-serif`
  const w = Math.ceil(ctx.measureText(text).width) + 24
  const h = fontSize + 20
  canvas.width = w
  canvas.height = h
  ctx.font = `500 ${fontSize}px Inter, sans-serif`
  ctx.fillStyle = color
  ctx.textBaseline = 'top'
  ctx.fillText(text, 12, 6)
  const texture = new THREE.CanvasTexture(canvas)
  texture.minFilter = THREE.LinearFilter
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false })
  const sprite = new THREE.Sprite(material)
  const scale = 0.0032
  sprite.scale.set(w * scale, h * scale, 1)
  return sprite
}

export default function Line3D({
  points, series, title, xAxisLabel, yAxisLabel, logX = false, logY = false,
  color = '#66c7ff', height = 340,
}: Line3DProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const [rotating, setRotating] = useState(true)

  const allSeries: Series3DSpec[] = series && series.length > 0
    ? series
    : [{ points: points ?? [], color }]

  useEffect(() => {
    if (controlsRef.current) controlsRef.current.autoRotate = rotating
  }, [rotating])

  useEffect(() => {
    const mount = mountRef.current
    const totalPoints = allSeries.reduce((n, s) => n + s.points.length, 0)
    if (!mount || totalPoints === 0) return

    const allXs = allSeries.flatMap((s) => s.points.map((p) => (logX ? Math.log10(p.x) : p.x)))
    const allYs = allSeries.flatMap((s) => s.points.map((p) => (logY ? Math.log10(p.y) : p.y)))
    const xMin = Math.min(...allXs), xMax = Math.max(...allXs)
    const yMin = Math.min(...allYs), yMax = Math.max(...allYs)
    const xRange = xMax - xMin || 1
    const yRange = yMax - yMin || 1

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, mount.clientWidth / height, 0.1, 100)
    camera.position.set(2.0, 1.4, 2.2)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, height)
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 1
    controls.maxDistance = 6
    controls.target.set(0, 0.3, 0)
    controls.autoRotate = rotating
    controls.autoRotateSpeed = 2
    controlsRef.current = controls

    // Floor grid, like the reference chart
    const grid = new THREE.GridHelper(2.6, 14, 0x66c7ff, 0x2a3540)
    ;(grid.material as THREE.Material).transparent = true
    ;(grid.material as THREE.Material).opacity = 0.25
    grid.position.y = -0.02
    scene.add(grid)

    const sphereGeo = new THREE.SphereGeometry(0.028, 20, 20)

    // Each series gets its own Z-depth band so multiple lines (e.g. FNO vs
    // PINN) sit visibly apart from each other instead of overlapping.
    allSeries.forEach((s, seriesIdx) => {
      const zBase = allSeries.length > 1 ? (seriesIdx / (allSeries.length - 1)) * 0.8 - 0.4 : 0
      const norm = s.points.map((p, i) => {
        const nx = (((logX ? Math.log10(p.x) : p.x) - xMin) / xRange) * 2 - 1
        const ny = (((logY ? Math.log10(p.y) : p.y) - yMin) / yRange) * 1.1
        const nz = zBase + (s.points.length > 1 ? (i / (s.points.length - 1)) * 0.3 - 0.15 : 0)
        return new THREE.Vector3(nx, ny, nz)
      })

      if (norm.length > 1) {
        const curve = new THREE.CatmullRomCurve3(norm, false, 'catmullrom', 0.15)
        const tubeGeo = new THREE.TubeGeometry(curve, 100, 0.012, 8, false)
        const tubeMat = new THREE.MeshStandardMaterial({ color: s.color, emissive: s.color, emissiveIntensity: 0.25, roughness: 0.4 })
        scene.add(new THREE.Mesh(tubeGeo, tubeMat))
      }

      norm.forEach((v, i) => {
        const mat = new THREE.MeshStandardMaterial({ color: s.color, emissive: s.color, emissiveIntensity: 0.4 })
        const sphere = new THREE.Mesh(sphereGeo, mat)
        sphere.position.copy(v)
        scene.add(sphere)

        // Thin vertical drop-line to the floor, like a stem plot.
        const dropGeo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(v.x, v.y, v.z),
          new THREE.Vector3(v.x, -0.02, v.z),
        ])
        const dropMat = new THREE.LineBasicMaterial({ color: s.color, transparent: true, opacity: 0.25 })
        scene.add(new THREE.Line(dropGeo, dropMat))

        if (s.points[i].label) {
          const sprite = makeLabelSprite(s.points[i].label!, s.color)
          sprite.position.set(v.x, v.y + 0.16, v.z)
          scene.add(sprite)
        }
      })

      if (s.name) {
        const nameSprite = makeLabelSprite(s.name, s.color)
        const last = norm[norm.length - 1]
        nameSprite.position.set(last.x + 0.15, last.y + 0.05, last.z)
        scene.add(nameSprite)
      }
    })

    if (xAxisLabel) {
      const s = makeLabelSprite(xAxisLabel, 'rgba(102,199,255,0.85)')
      s.position.set(1.15, -0.05, 0.6)
      scene.add(s)
    }
    if (yAxisLabel) {
      const s = makeLabelSprite(yAxisLabel, 'rgba(102,199,255,0.85)')
      s.position.set(-1.3, yRange ? 0.6 : 0.3, -0.5)
      scene.add(s)
    }

    scene.add(new THREE.AmbientLight(0xffffff, 0.8))
    const dir = new THREE.DirectionalLight(0xffffff, 0.6)
    dir.position.set(2, 3, 2)
    scene.add(dir)

    let raf: number
    const animate = () => {
      raf = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    const resizeObserver = new ResizeObserver(() => {
      if (!mount) return
      camera.aspect = mount.clientWidth / height
      camera.updateProjectionMatrix()
      renderer.setSize(mount.clientWidth, height)
    })
    resizeObserver.observe(mount)

    return () => {
      cancelAnimationFrame(raf)
      resizeObserver.disconnect()
      controls.dispose()
      controlsRef.current = null
      renderer.dispose()
      renderer.forceContextLoss()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, series, logX, logY, color, height])

  return (
    <div className="liquid-glass rounded-2xl overflow-hidden relative self-start" style={{ height: 'fit-content' }}>
      {title && (
        <div className="px-4 pt-3 pb-1">
          <p className="text-white/70 text-xs font-medium tracking-wide">{title}</p>
        </div>
      )}
      <div ref={mountRef} style={{ width: '100%', height, cursor: 'grab' }} />
      <div className="flex items-center justify-between px-4 pb-3 -mt-1">
        <p className="text-white/25 text-[0.65rem]">Drag to rotate · scroll to zoom · real data points</p>
        <button
          onClick={() => setRotating((r) => !r)}
          className="text-[0.65rem] px-2.5 py-1 rounded-full transition-colors"
          style={{
            background: rotating ? 'rgba(102,199,255,0.16)' : 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(102,199,255,0.3)',
            color: rotating ? '#a8e0ff' : 'rgba(232,224,218,0.5)',
          }}
        >
          Rotate {rotating ? 'On' : 'Off'}
        </button>
      </div>
    </div>
  )
}
