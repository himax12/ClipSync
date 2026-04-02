"use client";
import { motion } from "framer-motion";

interface BlurFadeProps {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
  yOffset?: number;
}

export function BlurFade({
  children,
  delay = 0,
  duration = 0.4,
  className = "",
  yOffset = 20,
}: BlurFadeProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: yOffset, filter: "blur(10px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{
        delay,
        duration,
        ease: "easeOut",
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
