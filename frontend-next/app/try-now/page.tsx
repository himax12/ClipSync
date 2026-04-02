"use client";
import { useState } from "react";
import { Upload, Link as LinkIcon, Film, ArrowRight, Loader2, FileJson, CheckCircle, XCircle } from "lucide-react";
import { BlurFade } from "@/components/ui/blur-fade";
import { SpotlightCard } from "@/components/ui/spotlight-card";
import { ShinyButton } from "@/components/ui/shiny-button";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type InputMode = "files" | "urls" | "json";

interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  output_path?: string;
  error?: string;
}

interface JsonPayload {
  a_roll: { url: string };
  b_rolls: Array<{ id: string; url: string }>;
}

export default function TryNowPage() {
  const [inputMode, setInputMode] = useState<InputMode>("files");
  const [arollFile] = useState<File | null>(null);
  const [brollFiles] = useState<File[]>([]);
  const [arollUrl, setArollUrl] = useState("");
  const [brollUrls, setBrollUrls] = useState("");
  const [jsonInput, setJsonInput] = useState("");
  const [jsonFile, setJsonFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleJsonUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setJsonFile(file);
      const reader = new FileReader();
      reader.onload = (event) => {
        setJsonInput(event.target?.result as string);
      };
      reader.readAsText(file);
    }
  };

  const handleSubmit = async () => {
    setIsProcessing(true);
    setError(null);
    setJobStatus(null);

    try {
      let payload: JsonPayload;

      if (inputMode === "json") {
        const parsed = JSON.parse(jsonInput);
        payload = {
          a_roll: { url: parsed.aroll || parsed.a_roll },
          b_rolls: (parsed.broll || parsed.b_rolls || []).map((url: string, i: number) => ({
            id: `broll_${i}`,
            url: url
          }))
        };
      } else {
        if (!arollUrl) {
          throw new Error("Please enter an A-Roll URL");
        }
        const brollList = brollUrls.split("\n").filter((url) => url.trim());
        payload = {
          a_roll: { url: arollUrl },
          b_rolls: brollList.map((url, i) => ({
            id: `broll_${i}`,
            url: url.trim()
          }))
        };
      }

      if (!payload.a_roll?.url) {
        throw new Error("A-Roll URL is required");
      }
      if (payload.b_rolls.length === 0) {
        throw new Error("At least one B-Roll URL is required");
      }

      const response = await fetch(`${API_BASE_URL}/api/process/json`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Failed to start processing");
      }

      const data = await response.json();
      setJobId(data.job_id);
      pollJobStatus(data.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setIsProcessing(false);
    }
  };

  const pollJobStatus = async (id: string) => {
    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/status/${id}`);
        if (!response.ok) throw new Error("Failed to get status");
        const status: JobStatus = await response.json();
        setJobStatus(status);

        if (status.status === "complete") {
          setIsProcessing(false);
          return;
        }
        if (status.status === "error") {
          setError(status.error || "Processing failed");
          setIsProcessing(false);
          return;
        }

        // Continue polling
        setTimeout(poll, 2000);
      } catch (err) {
        setTimeout(poll, 5000); // Retry after 5s on error
      }
    };
    poll();
  };

  return (
    <div className="min-h-screen pt-24 pb-16 px-6">
      <div className="max-w-3xl mx-auto">
        <BlurFade delay={0}>
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold tracking-tight mb-4 text-white">
              Create Your Video
            </h1>
            <p className="text-gray-400">
              Upload your A-Roll (main video) and B-Roll clips. Our AI will sync them together.
            </p>
          </div>
        </BlurFade>

        {/* Toggle Input Method */}
        <BlurFade delay={0.1}>
          <div className="flex items-center justify-center gap-2 mb-8">
            {[
              { mode: "files" as InputMode, label: "Upload Files" },
              { mode: "urls" as InputMode, label: "Use URLs" },
              { mode: "json" as InputMode, label: "JSON" },
            ].map(({ mode, label }) => (
              <button
                key={mode}
                onClick={() => setInputMode(mode)}
                className={`px-4 py-2 text-sm rounded-full transition-all ${
                  inputMode === mode
                    ? "bg-white text-black"
                    : "bg-white/10 text-gray-400 hover:text-white"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </BlurFade>

        <div className="space-y-8">
          {inputMode === "json" ? (
            /* JSON Mode */
            <BlurFade delay={0.2}>
              <SpotlightCard className="rounded-2xl p-8">
                <label className="block text-sm font-medium mb-4 text-gray-300">
                  <FileJson className="w-4 h-4 inline mr-2" />
                  JSON Configuration
                </label>

                <textarea
                  placeholder='{"aroll": "url", "broll": ["url1", "url2"]}'
                  value={jsonInput}
                  onChange={(e) => setJsonInput(e.target.value)}
                  rows={5}
                  className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all resize-none font-mono text-sm mb-6"
                />

                <div className="text-center mb-6">
                  <span className="text-sm text-gray-500">or</span>
                </div>

                <div className="border-2 border-dashed border-white/20 rounded-xl p-8 text-center hover:border-purple-500/50 transition-all cursor-pointer bg-white/5">
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleJsonUpload}
                    className="hidden"
                    id="json-upload"
                  />
                  <label htmlFor="json-upload" className="cursor-pointer">
                    {jsonFile ? (
                      <div className="flex flex-col items-center gap-3 text-white">
                        <FileJson className="w-8 h-8 text-purple-400" />
                        <span className="font-medium">{jsonFile.name}</span>
                      </div>
                    ) : (
                      <>
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center mx-auto mb-3">
                          <Upload className="w-5 h-5 text-purple-400" />
                        </div>
                        <p className="text-sm text-gray-400">Upload JSON file</p>
                      </>
                    )}
                  </label>
                </div>

                <div className="mt-6 p-4 bg-white/5 rounded-xl">
                  <p className="text-xs text-gray-400 mb-2">JSON Format:</p>
                  <pre className="text-xs text-gray-500">
{`{
  "aroll": "https://example.com/main.mp4",
  "broll": ["url1", "url2"]
}`}
                  </pre>
                </div>
              </SpotlightCard>
            </BlurFade>
          ) : (
            /* URL Mode */
            <>
              <BlurFade delay={0.2}>
                <SpotlightCard className="rounded-2xl p-8">
                  <label className="block text-sm font-medium mb-4 text-gray-300">
                    <Film className="w-4 h-4 inline mr-2" />
                    A-Roll Video (Main Content)
                  </label>
                  <div className="relative">
                    <LinkIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                    <input
                      type="url"
                      placeholder="https://example.com/video.mp4"
                      value={arollUrl}
                      onChange={(e) => setArollUrl(e.target.value)}
                      className="w-full pl-12 pr-4 py-4 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all"
                    />
                  </div>
                </SpotlightCard>
              </BlurFade>

              <BlurFade delay={0.3}>
                <SpotlightCard className="rounded-2xl p-8">
                  <label className="block text-sm font-medium mb-4 text-gray-300">
                    <Film className="w-4 h-4 inline mr-2" />
                    B-Roll Clips (Supporting Footage)
                  </label>
                  <div className="relative">
                    <LinkIcon className="absolute left-4 top-4 w-5 h-5 text-gray-500" />
                    <textarea
                      placeholder="Enter video URLs, one per line:&#10;https://example.com/clip1.mp4&#10;https://example.com/clip2.mp4"
                      value={brollUrls}
                      onChange={(e) => setBrollUrls(e.target.value)}
                      rows={4}
                      className="w-full pl-12 pr-4 py-4 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all resize-none"
                    />
                  </div>
                </SpotlightCard>
              </BlurFade>
            </>
          )}

          {/* Status Display */}
          {jobStatus && (
            <BlurFade delay={0}>
              <SpotlightCard className="rounded-2xl p-8">
                <div className="flex items-center gap-4 mb-4">
                  {jobStatus.status === "complete" ? (
                    <CheckCircle className="w-6 h-6 text-green-400" />
                  ) : jobStatus.status === "error" ? (
                    <XCircle className="w-6 h-6 text-red-400" />
                  ) : (
                    <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
                  )}
                  <div>
                    <h3 className="font-semibold text-white">{jobStatus.message}</h3>
                    <p className="text-sm text-gray-400">Progress: {jobStatus.progress}%</p>
                  </div>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${jobStatus.progress}%` }}
                  />
                </div>
                {jobStatus.output_path && (
                  <div className="mt-4">
                    <a
                      href={`${API_BASE_URL}/api/download/${jobStatus.job_id}`}
                      className="inline-flex items-center gap-2 px-4 py-2 bg-white text-black rounded-lg text-sm font-medium hover:bg-gray-100"
                    >
                      Download Result
                      <ArrowRight className="w-4 h-4" />
                    </a>
                  </div>
                )}
              </SpotlightCard>
            </BlurFade>
          )}

          {/* Error Display */}
          {error && (
            <BlurFade delay={0}>
              <SpotlightCard className="rounded-2xl p-6 border border-red-500/20">
                <div className="flex items-center gap-3 text-red-400">
                  <XCircle className="w-5 h-5" />
                  <span>{error}</span>
                </div>
              </SpotlightCard>
            </BlurFade>
          )}

          {/* Submit Button */}
          <BlurFade delay={0.4}>
            <div className="flex justify-center">
              <ShinyButton
                onClick={handleSubmit}
                disabled={isProcessing}
                className="px-8 py-4"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    Start Processing
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </ShinyButton>
            </div>
          </BlurFade>
        </div>
      </div>
    </div>
  );
}