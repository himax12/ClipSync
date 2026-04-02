"use client";
import Link from "next/link";
import { useState } from "react";
import { Menu, X, Clapperboard } from "lucide-react";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-black/50 backdrop-blur-md border-b border-white/5">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-2">
            <Clapperboard className="w-5 h-5 text-white" />
            <span className="font-medium text-white tracking-tight">ClipSync</span>
          </Link>
          
          <div className="hidden md:flex items-center gap-8">
            <Link href="/try-now" className="text-sm text-gray-400 hover:text-white transition-colors">
              Try Now
            </Link>
            <Link href="/#features" className="text-sm text-gray-400 hover:text-white transition-colors">
              Features
            </Link>
          </div>
          
          <div className="hidden md:flex items-center gap-4">
            <Link 
              href="/login" 
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Login
            </Link>
            <Link 
              href="/register" 
              className="px-4 py-1.5 bg-white text-black text-sm rounded-full hover:bg-gray-200 transition-colors"
            >
              Get Started
            </Link>
          </div>
          
          <button 
            className="md:hidden p-2 text-white"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>
      
      {isOpen && (
        <div className="md:hidden bg-black/95">
          <div className="px-6 py-4 space-y-3">
            <Link href="/try-now" className="block text-sm text-gray-400">Try Now</Link>
            <Link href="/#features" className="block text-sm text-gray-400">Features</Link>
            <Link href="/login" className="block text-sm text-gray-400">Login</Link>
            <Link href="/register" className="block text-sm font-medium text-white">Get Started</Link>
          </div>
        </div>
      )}
    </nav>
  );
}
