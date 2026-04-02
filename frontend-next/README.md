# ClipSync Frontend

AI-Powered B-Roll insertion frontend built with Next.js 15.

## Tech Stack

- **Next.js 15** - App Router
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Lucide React** - Icons

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Project Structure

```
frontend-next/
├── app/
│   ├── globals.css      # Global styles
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Landing page
│   ├── try-now/         # Upload interface
│   ├── login/           # Login page
│   └── register/        # Registration page
├── components/
│   ├── Navbar.tsx       # Navigation
│   └── Footer.tsx       # Footer
├── tailwind.config.ts   # Tailwind configuration
└── package.json
```

## Pages

- `/` - Landing page with hero, features, and CTA
- `/try-now` - Video upload interface
- `/login` - User login
- `/register` - User registration
