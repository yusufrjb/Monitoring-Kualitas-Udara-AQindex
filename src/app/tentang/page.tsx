"use client";

import React from "react";
import { Info, Wind, Database, Code, Zap, Shield, Github, Mail, Globe, Heart } from "lucide-react";
import { cn } from "@/lib/utils";

export default function TentangPage() {
  return (
    <div className="min-h-full bg-slate-50 dark:bg-slate-950">
      <div className="w-full px-6 py-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600 text-white shadow-lg">
              <Info size={20} />
            </div>
            Tentang
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Informasi tentang DashboardAQ
          </p>
        </div>

        <div className="grid gap-6 max-w-4xl">
          {/* Header */}
          <div className="rounded-xl border border-border bg-card p-8 shadow-sm text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-600 text-white shadow-lg mx-auto mb-4">
              <Wind size={32} />
            </div>
            <h2 className="text-2xl font-bold text-foreground mb-2">DashboardAQ v3.0</h2>
            <p className="text-muted-foreground mb-4">
              Sistem Monitoring Kualitas Udara Real-time
            </p>
            <div className="flex items-center justify-center gap-4 text-sm text-muted-foreground">
              <span>© 2026</span>
              <span>•</span>
              <span>Built with Next.js</span>
              <span>•</span>
              <span>Powered by Supabase</span>
            </div>
          </div>

          {/* Fitur */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Zap size={18} className="text-blue-500" />
              Fitur Utama
            </h3>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50 dark:bg-blue-950/30 text-blue-600 shrink-0">
                  <Wind size={20} />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-foreground">Monitoring Real-time</h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    Pantau kualitas udara secara langsung dengan update otomatis
                  </p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-50 dark:bg-emerald-950/30 text-emerald-600 shrink-0">
                  <Database size={20} />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-foreground">Data Agregasi</h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    Analisis pola harian, mingguan, dan bulanan dengan efisien
                  </p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-50 dark:bg-purple-950/30 text-purple-600 shrink-0">
                  <Code size={20} />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-foreground">ML Forecasting</h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    Prediksi kualitas udara dengan model machine learning
                  </p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-orange-50 dark:bg-orange-950/30 text-orange-600 shrink-0">
                  <Shield size={20} />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-foreground">Indeks ISPU</h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    Standar KLHK Indonesia untuk klasifikasi kualitas udara
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Parameter */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Wind size={18} className="text-blue-500" />
              Parameter Polutan
            </h3>
            <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
              {[
                { name: "PM2.5", desc: "Partikulat halus &lt; 2.5µm" },
                { name: "PM10", desc: "Partikulat kasar &lt; 10µm" },
                { name: "NO₂", desc: "Nitrogen Dioksida" },
                { name: "CO", desc: "Karbon Monoksida" },
                { name: "O₃", desc: "Ozon permukaan tanah" },
              ].map((param) => (
                <div key={param.name} className="rounded-lg border border-border bg-muted/40 p-3">
                  <p className="text-sm font-semibold text-foreground">{param.name}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{param.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Standar ISPU */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Shield size={18} className="text-blue-500" />
              Standar ISPU (KLHK)
            </h3>
            <div className="space-y-2">
              {[
                { range: "0 - 50", label: "Baik", color: "bg-emerald-500" },
                { range: "51 - 100", label: "Sedang", color: "bg-yellow-400" },
                { range: "101 - 200", label: "Tidak Sehat", color: "bg-orange-500" },
                { range: "201 - 300", label: "Sangat Tidak Sehat", color: "bg-red-500" },
                { range: "&gt; 300", label: "Berbahaya", color: "bg-rose-800" },
              ].map((item) => (
                <div key={item.range} className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 transition-colors">
                  <div className={cn("h-3 w-3 rounded-full", item.color)} />
                  <span className="text-sm font-medium text-foreground flex-1">{item.label}</span>
                  <span className="text-xs text-muted-foreground">ISPU {item.range}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Teknologi */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Code size={18} className="text-blue-500" />
              Teknologi
            </h3>
            <div className="grid gap-3 text-sm">
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Frontend Framework</span>
                <span className="font-medium text-foreground">Next.js 15</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">UI Components</span>
                <span className="font-medium text-foreground">Radix UI + Tailwind CSS</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Charts</span>
                <span className="font-medium text-foreground">Recharts</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Database</span>
                <span className="font-medium text-foreground">Supabase (PostgreSQL)</span>
              </div>
              <div className="flex items-center justify-between py-2">
                <span className="text-muted-foreground">Real-time</span>
                <span className="font-medium text-foreground">Supabase Realtime</span>
              </div>
            </div>
          </div>

          {/* Kontak */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Mail size={18} className="text-blue-500" />
              Kontak & Support
            </h3>
            <div className="flex flex-wrap gap-3">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-muted/40 hover:bg-muted/60 transition-colors text-sm font-medium text-foreground"
              >
                <Github size={16} />
                GitHub
              </a>
              <a
                href="mailto:support@example.com"
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-muted/40 hover:bg-muted/60 transition-colors text-sm font-medium text-foreground"
              >
                <Mail size={16} />
                Email
              </a>
              <a
                href="https://example.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-muted/40 hover:bg-muted/60 transition-colors text-sm font-medium text-foreground"
              >
                <Globe size={16} />
                Website
              </a>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center py-6">
            <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground mb-2">
              <span>Dibuat dengan</span>
              <Heart size={14} className="text-red-500 fill-red-500" />
              <span>oleh Tim DashboardAQ</span>
            </div>
            <p className="text-xs text-muted-foreground">
              DashboardAQ v3.0 • © 2026 All rights reserved
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
