"use client";
import { useState } from "react";

interface RippleButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  rippleColor?: string;
}

export function RippleButton({
  children,
  className = "",
  rippleColor = "rgba(255, 255, 255, 0.3)",
  ...props
}: RippleButtonProps) {
  const [ripples, setRipples] = useState<Array<{ x: number; y: number; id: number }>>([]);

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    const button = e.currentTarget;
    const rect = button.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const id = Date.now();

    setRipples((prev) => [...prev, { x, y, id }]);
    setTimeout(() => {
      setRipples((prev) => prev.filter((r) => r.id !== id));
    }, 600);

    props.onClick?.(e);
  };

  return (
    <button
      className={`relative overflow-hidden px-6 py-3 bg-white text-black rounded-full font-medium transition-all hover:bg-gray-100 active:scale-95 ${className}`}
      onClick={handleClick}
      {...props}
    >
      <span className="relative z-10 flex items-center justify-center gap-2">
        {children}
      </span>
      {ripples.map((ripple) => (
        <span
          key={ripple.id}
          className="absolute rounded-full animate-ripple"
          style={{
            left: ripple.x,
            top: ripple.y,
            width: 10,
            height: 10,
            background: rippleColor,
            transform: "translate(-50%, -50%)",
          }}
        />
      ))}
    </button>
  );
}
