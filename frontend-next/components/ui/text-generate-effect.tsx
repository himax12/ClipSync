"use client";
import { useEffect, useState } from "react";

interface TextGenerateEffectProps {
  words: string;
  className?: string;
  duration?: number;
}

export function TextGenerateEffect({
  words,
  className = "",
  duration = 0.5,
}: TextGenerateEffectProps) {
  const [displayedText, setDisplayedText] = useState("");
  const [isAnimating, setIsAnimating] = useState(true);

  useEffect(() => {
    setDisplayedText("");
    setIsAnimating(true);
    
    let index = 0;
    const interval = setInterval(() => {
      if (index < words.length) {
        setDisplayedText(words.slice(0, index + 1));
        index++;
      } else {
        setIsAnimating(false);
        clearInterval(interval);
      }
    }, duration * 1000 / words.length);

    return () => clearInterval(interval);
  }, [words, duration]);

  return (
    <span className={className}>
      {displayedText}
      {isAnimating && (
        <span className="inline-block w-0.5 h-5 bg-purple-400 ml-1 animate-pulse" />
      )}
    </span>
  );
}
