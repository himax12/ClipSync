"use client";
import { useState, useRef } from "react";

interface SpotlightCardProps {
  children: React.ReactNode;
  className?: string;
}

export function SpotlightCard({ children, className = "" }: SpotlightCardProps) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (cardRef.current) {
      const rect = cardRef.current.getBoundingClientRect();
      setMousePosition({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    }
  };

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`relative overflow-hidden rounded-2xl transition-all duration-300 ${className}`}
      style={{
        background: "rgba(255, 255, 255, 0.03)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(255, 255, 255, 0.08)",
      }}
    >
      {/* Spotlight effect */}
      {isHovered && (
        <div
          className="absolute pointer-events-none transition-opacity duration-300"
          style={{
            left: mousePosition.x,
            top: mousePosition.y,
            width: 300,
            height: 300,
            transform: "translate(-50%, -50%)",
            background: "radial-gradient(circle, rgba(167, 139, 250, 0.15) 0%, transparent 70%)",
            opacity: 1,
          }}
        />
      )}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
