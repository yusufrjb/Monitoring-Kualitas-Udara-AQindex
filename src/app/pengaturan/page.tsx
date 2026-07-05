"use client";

import React from "react";
import { Settings, Bell, Moon, Sun, RefreshCw, Database, Shield, Globe } from "lucide-react";
import { cn } from "@/lib/utils";

export default function PengaturanPage() {
  const [darkMode, setDarkMode] = React.useState(false);
  const [notifications, setNotifications] = React.useState(true);
  const [autoRefresh, setAutoRefresh] = React.useState(true);

  return (
    <div className="min-h-full bg-slate-50 dark:bg-slate-950">
      <div className="w-full px-6 py-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600 text-white shadow-lg">
              <Settings size={20} />
            </div>
            Pengaturan
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Konfigurasi preferensi dan sistem
          </p>
        </div>

        <div className="grid gap-6 max-w-4xl">
          {/* Tampilan */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Globe size={18} className="text-blue-500" />
              Tampilan
            </h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-foreground">Mode Gelap</p>
                  <p className="text-xs text-muted-foreground">Gunakan tema gelap untuk kenyamanan mata</p>
                </div>
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className={cn(
                    "relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200",
                    darkMode ? "bg-blue-600" : "bg-slate-200"
                  )}
                >
                  <span
                    className={cn(
                      "inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform duration-200",
                      darkMode ? "translate-x-6" : "translate-x-0.5"
                    )}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* Notifikasi */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Bell size={18} className="text-blue-500" />
              Notifikasi
            </h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-foreground">Notifikasi Kualitas Udara</p>
                  <p className="text-xs text-muted-foreground">Dapatkan peringatan saat kualitas udara buruk</p>
                </div>
                <button
                  onClick={() => setNotifications(!notifications)}
                  className={cn(
                    "relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200",
                    notifications ? "bg-blue-600" : "bg-slate-200"
                  )}
                >
                  <span
                    className={cn(
                      "inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform duration-200",
                      notifications ? "translate-x-6" : "translate-x-0.5"
                    )}
                  />
                </button>
              </div>

              {notifications && (
                <div className="ml-4 space-y-2 pt-2 border-t border-border">
                  <label className="flex items-center gap-2 text-sm text-muted-foreground">
                    <input type="checkbox" defaultChecked className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                    ISPU &gt; 100 (Tidak Sehat)
                  </label>
                  <label className="flex items-center gap-2 text-sm text-muted-foreground">
                    <input type="checkbox" defaultChecked className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                    ISPU &gt; 200 (Sangat Tidak Sehat)
                  </label>
                  <label className="flex items-center gap-2 text-sm text-muted-foreground">
                    <input type="checkbox" defaultChecked className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                    ISPU &gt; 300 (Berbahaya)
                  </label>
                </div>
              )}
            </div>
          </div>

          {/* Data */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Database size={18} className="text-blue-500" />
              Data
            </h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-foreground">Auto Refresh</p>
                  <p className="text-xs text-muted-foreground">Perbarui data secara otomatis setiap 1 menit</p>
                </div>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={cn(
                    "relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200",
                    autoRefresh ? "bg-blue-600" : "bg-slate-200"
                  )}
                >
                  <span
                    className={cn(
                      "inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform duration-200",
                      autoRefresh ? "translate-x-6" : "translate-x-0.5"
                    )}
                  />
                </button>
              </div>

              <div className="pt-4 border-t border-border">
                <p className="text-sm font-medium text-foreground mb-2">Sumber Data</p>
                <div className="space-y-2 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span>Realtime: tb_konsentrasi_gas</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500" />
                    <span>Prediksi: tb_prediksi_kualitas_udara</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-purple-500" />
                    <span>Agregasi: air_quality_hourly_agg</span>
                  </div>
                </div>
              </div>

              <button className="mt-4 flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium">
                <RefreshCw size={14} />
                Refresh Data Sekarang
              </button>
            </div>
          </div>

          {/* Keamanan */}
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Shield size={18} className="text-blue-500" />
              Keamanan
            </h2>
            
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between py-2">
                <span className="text-muted-foreground">API Key Status</span>
                <span className="text-emerald-600 font-medium">Aktif</span>
              </div>
              <div className="flex items-center justify-between py-2 border-t border-border">
                <span className="text-muted-foreground">Koneksi Database</span>
                <span className="text-emerald-600 font-medium">Terhubung</span>
              </div>
              <div className="flex items-center justify-between py-2 border-t border-border">
                <span className="text-muted-foreground">Terakhir Diperbarui</span>
                <span className="text-foreground font-medium">Baru saja</span>
              </div>
            </div>
          </div>

          <div className="text-center text-xs text-muted-foreground">
            <p>AQindex v3.0 • © 2026</p>
          </div>
        </div>
      </div>
    </div>
  );
}
