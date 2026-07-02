"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  BarChart2,
  Settings,
  ChevronLeft,
  Wind,
  Info,
  X
} from "lucide-react";
import { usePathname } from "next/navigation";

interface NavItem {
  label: string;
  icon: React.ReactNode;
  href: string;
  active?: boolean;
}

const navItems: NavItem[] = [
  { label: "Overview", icon: <LayoutDashboard size={18} />, href: "/", active: true },
  { label: "Forecast", icon: <BarChart2 size={18} />, href: "/statistik" },
];

const bottomItems: NavItem[] = [
  { label: "Pengaturan", icon: <Settings size={18} />, href: "/pengaturan" },
  { label: "Tentang", icon: <Info size={18} />, href: "/tentang" },
];

export default function Sidebar({
  mobileOpen,
  setMobileOpen,
}: {
  mobileOpen: boolean;
  setMobileOpen: (open: boolean) => void;
}) {
  const [open, setOpen] = useState(true);
  const pathname = usePathname(); // Get the current route

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/80 backdrop-blur-sm md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex h-screen flex-col border-r border-slate-800 bg-slate-950 transition-transform duration-300 ease-in-out md:relative md:translate-x-0",
          open ? "w-64 md:w-56" : "w-[60px]",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Logo */}
        <div className="flex h-16 shrink-0 items-center justify-between border-b border-slate-800 px-4">
          <div className="flex items-center gap-2.5 overflow-hidden">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white shadow-lg shadow-blue-900/20">
              <Wind size={16} />
            </div>
            {open && (
              <span className="whitespace-nowrap text-sm font-semibold text-white">
                AQindex
              </span>
            )}
          </div>
          {/* Close button on mobile */}
          <button
            onClick={() => setMobileOpen(false)}
            className="md:hidden text-slate-400 hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        {/* Toggle button (Desktop only) */}
        <button
          onClick={() => setOpen((v) => !v)}
          className="absolute -right-3 top-[70px] z-[40] hidden md:flex h-6 w-6 items-center justify-center rounded-full border border-slate-700 bg-slate-900 text-slate-400 shadow-md transition-all hover:scale-110 hover:bg-slate-800 hover:text-white"
          aria-label={open ? "Tutup sidebar" : "Buka sidebar"}
        >
          <ChevronLeft
            size={13}
            className={cn("transition-transform duration-300", !open && "rotate-180")}
          />
        </button>

        {/* Nav */}
        <nav className="flex flex-1 flex-col gap-1 overflow-hidden px-2 py-4">
          {navItems.map((item) => (
            <NavLink key={item.href} item={{ ...item, active: pathname === item.href }} open={open} />
          ))}
        </nav>

        {/* Bottom */}
        <div className="border-t border-slate-800 px-2 py-3">
          {bottomItems.map((item) => (
            <NavLink key={item.href} item={{ ...item, active: pathname === item.href }} open={open} />
          ))}
        </div>
      </aside>
    </>
  );
}

function NavLink({ item, open }: { item: NavItem; open: boolean }) {
  return (
    <a
      href={item.href}
      title={!open ? item.label : undefined}
      className={cn(
        "group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
        item.active
          ? "bg-white/10 text-white"
          : "text-slate-400 hover:bg-white/5 hover:text-white"
      )}
    >
      <span className="flex items-center justify-center">{item.icon}</span>
      {open && <span className="whitespace-nowrap">{item.label}</span>}
    </a>
  );
}
