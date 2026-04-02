import Link from "next/link";

export default function Footer() {
  return (
    <footer className="py-8">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-sm text-gray-500">
            © 2026 ClipSync
          </div>
          <div className="flex items-center gap-6">
            <Link href="/privacy" className="text-sm text-gray-500 hover:text-white transition-colors">Privacy</Link>
            <Link href="/terms" className="text-sm text-gray-500 hover:text-white transition-colors">Terms</Link>
            <Link href="https://github.com/himax12/ClipSync" className="text-sm text-gray-500 hover:text-white transition-colors">GitHub</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
