'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const path = usePathname();

  const links = [
    { href: '/',          label: 'Accueil' },
    { href: '/predict',   label: 'Estimer un bien' },
    { href: '/analysis',  label: 'Analyse du marché' },
  ];

  return (
    <nav className="bg-mauritania-green text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-2 font-bold text-lg">
            <span className="text-mauritania-gold">🏠</span>
            <span>ImmobilierNK</span>
          </Link>

          <div className="flex gap-6">
            {links.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={`text-sm font-medium transition-colors hover:text-mauritania-gold ${
                  path === href ? 'text-mauritania-gold border-b-2 border-mauritania-gold' : 'text-white/90'
                }`}
              >
                {label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
