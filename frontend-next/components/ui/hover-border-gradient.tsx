"use client";
import { useState } from "react";

interface HoverBorderGradientProps {
  children: React.ReactNode;
  className?: string;
  containerClassName?: string;
}

export function HoverBorderGradient({
  children,
  className = "",
  containerClassName = "",
}: HoverBorderGradientProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className={`relative rounded-2xl p-[1px] overflow-hidden transition-all duration-300 ${containerClassName}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        background: isHovered
          ? "linear-gradient(135deg, rgba(167, 139, 250, 0.8), rgba(59, 130, 246, 0.8), rgba(236, 72, 153, 0.8))"
          : "linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))",
      }}
    >
      <div
        className={`rounded-[15px] ${className}`}
        style={{
          background: "rgba(10, 10, 10, 0.9)",
        }}
      >
        {children}
      </div>
    </div>
  );
}
