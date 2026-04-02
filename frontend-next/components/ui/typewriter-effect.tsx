"use client";
import { useEffect, useState } from "react";

interface TypewriterEffectProps {
  words: Array<{ text: string }>;
  className?: string;
  cursorClassName?: string;
}

export function TypewriterEffect({
  words,
  className = "",
  cursorClassName = "",
}: TypewriterEffectProps) {
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [currentText, setCurrentText] = useState("");

  useEffect(() => {
    const word = words[currentWordIndex];
    if (!word) return;

    let charIndex = 0;
    const textInterval = setInterval(() => {
      if (charIndex <= word.text.length) {
        setCurrentText(word.text.slice(0, charIndex));
        charIndex++;
      } else {
        clearInterval(textInterval);
        setTimeout(() => {
          setCurrentWordIndex((prev) => (prev + 1) % words.length);
        }, 2000);
      }
    }, 100);

    return () => clearInterval(textInterval);
  }, [currentWordIndex, words]);

  useEffect(() => {
    setCurrentText("");
    setCurrentWordIndex(0);
  }, [words]);

  return (
    <span className={`inline-flex items-center ${className}`}>
      <span className="text-white">{currentText}</span>
      <span className={`w-0.5 h-6 bg-purple-400 animate-pulse ml-1 ${cursorClassName}`} />
    </span>
  );
}
