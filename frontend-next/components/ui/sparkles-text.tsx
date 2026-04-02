"use client";
import { useEffect, useState } from "react";

interface SparklesTextProps {
  children: React.ReactNode;
  className?: string;
  sparklesCount?: number;
}

export function SparklesText({ children, className = "", sparklesCount = 12 }: SparklesTextProps) {
  const [sparkles, setSparkles] = useState<
    Array<{ id: number; x: number; y: number; delay: number; duration: number }>
  >([]);

  useEffect(() => {
    const newSparkles = Array.from({ length: sparklesCount }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 3,
      duration: 2 + Math.random() * 2,
    }));
    setSparkles(newSparkles);
  }, [sparklesCount]);

  return (
    <span className={`relative inline-block ${className}`}>
      {sparkles.map((sparkle) => (
        <span
          key={sparkle.id}
          className="absolute inline-block w-1 h-1 bg-white rounded-full animate-sparkle"
          style={{
            left: `${sparkle.x}%`,
            top: `${sparkle.y}%`,
            animationDelay: `${sparkle.delay}s`,
            animationDuration: `${sparkle.duration}s`,
          }}
        />
      ))}
      {children}
    </span>
  );
}
