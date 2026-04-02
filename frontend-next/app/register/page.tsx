"use client";
import { useState } from "react";
import Link from "next/link";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { BlurFade } from "@/components/ui/blur-fade";
import { HoverBorderGradient } from "@/components/ui/hover-border-gradient";
import { RippleButton } from "@/components/ui/ripple-button";

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      alert("Register UI demo - connect to backend for actual auth");
    }, 1500);
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-6 pt-16">
      <BlurFade delay={0} duration={0.5}>
        <HoverBorderGradient containerClassName="w-full max-w-sm">
          <div className="p-8">
            <div className="text-center mb-8">
              <h1 className="text-2xl font-bold tracking-tight text-white">Create account</h1>
              <p className="text-gray-400 mt-2">Start creating videos with AI</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-300">Name</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-gray-300">Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-gray-300">Password</label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all pr-10"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              <label className="flex items-start gap-2 cursor-pointer text-sm">
                <input type="checkbox" className="mt-1 rounded border-white/20 bg-white/5" required />
                <span className="text-gray-400">
                  I agree to the{" "}
                  <Link href="/terms" className="text-white hover:underline">Terms of Service</Link>
                  {" "}and{" "}
                  <Link href="/privacy" className="text-white hover:underline">Privacy Policy</Link>
                </span>
              </label>

              <RippleButton
                type="submit"
                disabled={isLoading}
                className="w-full mt-6"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Creating account...
                  </>
                ) : (
                  "Create Account"
                )}
              </RippleButton>
            </form>

            <div className="my-6 flex items-center gap-4">
              <div className="flex-1 h-px bg-white/10" />
              <span className="text-sm text-gray-500">or sign up with</span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

            <button className="w-full py-3 rounded-xl font-medium bg-white/5 border border-white/10 hover:bg-white/10 transition-all flex items-center justify-center gap-2 text-white">
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Google
            </button>

            <p className="text-center text-sm text-gray-400 mt-6">
              Already have an account?{" "}
              <Link href="/login" className="font-medium text-white hover:underline">
                Sign in
              </Link>
            </p>
          </div>
        </HoverBorderGradient>
      </BlurFade>
    </div>
  );
}