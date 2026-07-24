import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'
import { IncomingMessage, ServerResponse } from 'http'

// Project root is one level above pinn-dashboard/
const projectRoot = path.resolve(__dirname, '..')

/**
 * Vite plugin: serve /outputs/<rest> from <projectRoot>/<rest>
 * e.g.  GET /outputs/burgers_pinn/outputs/heatmap.png
 *       → reads  d:/pnn/burgers_pinn/outputs/heatmap.png
 * No symlinks, no copies, zero config for the user.
 */
function serveOutputsPlugin() {
  return {
    name: 'serve-project-outputs',
    configureServer(server: { middlewares: { use: (fn: (req: IncomingMessage, res: ServerResponse, next: () => void) => void) => void } }) {
      server.middlewares.use((req: IncomingMessage, res: ServerResponse, next: () => void) => {
        const url = req.url ?? ''
        if (!url.startsWith('/outputs/')) return next()

        // Strip leading /outputs/ and resolve against project root
        const rel = url.slice('/outputs/'.length).split('?')[0]
        const filePath = path.join(projectRoot, rel)

        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
          return next()
        }

        const ext = path.extname(filePath).toLowerCase()
        const mimeMap: Record<string, string> = {
          '.png': 'image/png',
          '.jpg': 'image/jpeg',
          '.jpeg': 'image/jpeg',
          '.gif': 'image/gif',
          '.svg': 'image/svg+xml',
          '.webp': 'image/webp',
        }
        const mime = mimeMap[ext] ?? 'application/octet-stream'
        res.setHeader('Content-Type', mime)
        res.setHeader('Cache-Control', 'public, max-age=3600')
        fs.createReadStream(filePath).pipe(res)
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), serveOutputsPlugin()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    fs: {
      allow: ['.', projectRoot],
    },
  },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three'],
          vendor: ['react', 'react-dom', 'framer-motion'],
          lucide: ['lucide-react'],
        },
      },
    },
  },
})
