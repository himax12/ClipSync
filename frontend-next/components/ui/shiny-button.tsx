"use client";
import { forwardRef } from "react";

interface ShinyButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
}

export const ShinyButton = forwardRef<HTMLButtonElement, ShinyButtonProps>(
  ({ children, className = "", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={`relative overflow-hidden px-6 py-3 bg-white text-black rounded-full font-medium transition-all hover:bg-gray-100 active:scale-95 ${className}`}
        {...props}
      >
        <span className="relative z-10 flex items-center justify-center gap-2">
          {children}
        </span>
        <span
          className="absolute inset-0 -translate-x-full skew-x-12 bg-gradient-to-r from-transparent via-white/50 to-transparent animate-shine"
          style={{
            background: "linear-gradient(to right, transparent, rgba(255,255,255,0.4), transparent)",
          }}
        />
      </button>
    );
  }
);

ShinyButton.displayName = "ShinyButton";
