import { motion } from "framer-motion";

/**
 * Animated shiny gradient sweep text, e.g. for a hero heading.
 * Drop into: pinn-dashboard/src/components/ShinyText.tsx
 * Usage:  <ShinyText text="uncertainty-aware physics" />
 */
export default function ShinyText({
  text,
  baseColor = "#a8e0ff",
  shineColor = "#ffffff",
  speed = 3,
  className = "",
}: {
  text: string;
  baseColor?: string;
  shineColor?: string;
  speed?: number;
  className?: string;
}) {
  return (
    <motion.span
      className={className}
      style={{
        display: "inline-block",
        backgroundImage: `linear-gradient(100deg, ${baseColor} 0%, ${baseColor} 40%, ${shineColor} 50%, ${baseColor} 60%, ${baseColor} 100%)`,
        backgroundSize: "200% 100%",
        backgroundClip: "text",
        WebkitBackgroundClip: "text",
        color: "transparent",
        WebkitTextFillColor: "transparent",
      }}
      animate={{ backgroundPositionX: ["0%", "-200%"] }}
      transition={{ duration: speed, repeat: Infinity, ease: "linear" }}
    >
      {text}
    </motion.span>
  );
}
