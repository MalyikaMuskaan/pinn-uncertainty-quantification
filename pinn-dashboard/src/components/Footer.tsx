import { Github, Mail, ArrowUp } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="relative z-10 mt-10 border-t" style={{ borderColor: 'rgba(102, 199, 255,0.1)' }}>
      <div className="max-w-5xl mx-auto px-6 py-10 flex flex-col sm:flex-row items-center justify-between gap-6">

        {/* Links */}
        <div className="flex items-center gap-4">
          <a
            href="https://github.com/MalyikaMuskaan"
            target="_blank"
            rel="noreferrer"
            className="liquid-glass flex items-center gap-2 px-4 py-2 text-xs text-white/60 hover:text-white transition-colors rounded-full"
          >
            <Github size={13} />
            GitHub
          </a>
          <a
            href="https://mail.google.com/mail/?view=cm&fs=1&to=malyikamuskaann@gmail.com"
            target="_blank"
            rel="noreferrer"
            className="liquid-glass flex items-center gap-2 px-4 py-2 text-xs text-white/60 hover:text-white transition-colors rounded-full"
          >
            <Mail size={13} />
            Contact
          </a>
          <a
            href="#home"
            className="liquid-glass flex items-center gap-2 px-4 py-2 text-xs text-white/60 hover:text-white transition-colors rounded-full"
          >
            <ArrowUp size={13} />
            Back to top
          </a>
        </div>

        {/* Byline */}
        <p
          className="text-center sm:text-right"
          style={{ fontSize: '0.72rem', color: 'rgba(232,224,218,0.28)', letterSpacing: '0.04em' }}
        >
          <span style={{ color: 'rgba(232,224,218,0.4)' }}>
            Physics-Informed Neural Networks + Uncertainty Quantification
          </span>
          <br />
          <span style={{ fontWeight: 700, color: 'rgba(232,224,218,0.55)' }}>by Malyika Muskaan</span>
        </p>
      </div>
    </footer>
  )
}
