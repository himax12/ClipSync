"use client";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="pt-20">
      {/* Hero Section */}
      <section className="min-h-[80vh] flex flex-col items-center justify-center px-6">
        <div className="max-w-3xl mx-auto text-center space-y-10">
          {/* Badge */}
          <div className="animate-fade-in inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-white/10 glass">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-sm text-white/70">AI-Powered Video Editing</span>
          </div>

          {/* Main Heading */}
          <div className="animate-fade-in animate-delay-100 space-y-4">
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-tight">
              <span className="text-white">Sync B-Roll to Your</span>
              <br />
              <span className="gradient-text">A-Roll Content</span>
            </h1>
          </div>

          {/* Subheading */}
          <p className="animate-fade-in animate-delay-200 text-lg text-white/50 max-w-xl mx-auto">
            Upload an A-Roll video and B-Roll clips. Our AI automatically 
            transcribes, analyzes, and assembles the perfect video.
          </p>

          {/* CTA Button */}
          <div className="animate-fade-in animate-delay-300 flex items-center justify-center gap-4">
            <Link 
              href="/try-now"
              className="px-8 py-4 bg-white text-black rounded-full font-semibold hover:bg-gray-100 transition-colors inline-flex items-center gap-2"
            >
              Try Now
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link 
              href="/login"
              className="px-8 py-4 glass rounded-full font-medium hover:bg-white/5 transition-colors"
            >
              Sign In
            </Link>
          </div>

          {/* Quick Stats */}
          <div className="animate-fade-in animate-delay-400 flex items-center justify-center gap-12 pt-8">
            {[
              { value: "10+", label: "B-Roll Clips" },
              { value: "5min", label: "Max Duration" },
              { value: "3-5s", label: "VLM Latency" },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-4xl font-bold gradient-text">{stat.value}</div>
                <div className="text-xs text-white/40 uppercase tracking-widest mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Demo Video Placeholder */}
        <div className="animate-fade-in animate-delay-500 mt-16 w-full max-w-3xl">
          <div className="relative aspect-video rounded-2xl overflow-hidden border border-white/5 glass">
            <div className="absolute inset-0 flex items-center justify-center">
              <button className="w-20 h-20 rounded-full bg-white/10 border border-white/20 flex items-center justify-center hover:bg-white/20 transition-colors backdrop-blur-sm">
                <div className="w-0 h-0 border-t-[10px] border-t-transparent border-l-[16px] border-l-white border-b-[10px] border-b-transparent ml-1" />
              </button>
            </div>
            <div className="absolute bottom-4 left-4 text-sm text-white/40">
              Watch demo
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-32 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              How It Works
            </h2>
            <p className="text-white/50">
              Four simple steps to transform your raw footage
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {[
              { num: "01", title: "Upload A-Roll", desc: "Start with your main talking head video. We support MP4, MOV, and WebM formats up to 5 minutes." },
              { num: "02", title: "Add B-Roll Clips", desc: "Upload your supporting footage — cityscapes, b-roll, stock clips. Up to 10 clips supported." },
              { num: "03", title: "AI Analyzes & Syncs", desc: "Our AI transcribes your audio, analyzes visuals, and intelligently matches B-Roll to your content." },
              { num: "04", title: "Download Result", desc: "Get your polished video with perfectly timed B-Roll cuts. No editing skills required." },
            ].map((step) => (
              <div key={step.num} className="flex items-start gap-6 p-6 rounded-2xl glass hover:border-white/20 transition-all">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-white/5 flex items-center justify-center">
                  <span className="text-sm font-bold text-white/60">{step.num}</span>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">{step.title}</h3>
                  <p className="text-sm text-white/50 leading-relaxed">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6">
        <div className="max-w-2xl mx-auto text-center">
          <div className="glass-strong rounded-3xl p-12">
            <h2 className="text-2xl md:text-3xl font-bold text-white mb-4">
              Ready to get started?
            </h2>
            <p className="text-white/50 mb-8">
              Join thousands of creators using AI to make better videos.
            </p>
            <Link 
              href="/try-now"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white text-black rounded-full font-medium hover:bg-gray-100 transition-colors"
            >
              Start Creating
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}