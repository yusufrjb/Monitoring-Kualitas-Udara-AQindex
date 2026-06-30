"use client";

import React, { useState, useEffect } from 'react';
import {
    ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer
} from 'recharts';
import { Zap, TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { supabase } from '@/lib/supabase';
import type { ForecastPoint, ForecastMetadata } from '@/types';

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        const p = payload[0]?.payload;
        if (!p) return null;
        const isHistorical = p.isHistorical;
        const isFitted = p.isFitted;
        return (
            <div className="bg-white/95 border border-slate-200 p-3 rounded-lg shadow-xl max-w-[240px]">
                <p className="text-slate-500 text-xs mb-1">{label}</p>
                {payload.map((entry: any, idx: number) => (
                    entry.value != null && (
                        <p key={idx} className="font-semibold text-sm" style={{ color: entry.color }}>
                            {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}
                        </p>
                    )
                ))}
                {isFitted && (
                    <div className="mt-1 pt-1 border-t border-slate-200">
                        <p className="text-xs text-slate-400">Aktual: <span className="font-semibold text-slate-700">{p.pm25?.toFixed(1)}, {p.pm10?.toFixed(1)}, {p.co?.toFixed(1)}</span></p>
                        <p className="text-xs text-slate-400">Fitted: <span className="font-semibold text-violet-600">{p.pm25_fitted?.toFixed(1)}, {p.pm10_fitted?.toFixed(1)}, {p.co_fitted?.toFixed(1)}</span></p>
                    </div>
                )}
                <p className="text-xs text-slate-400 mt-1">
                    {isHistorical ? (isFitted ? 'Aktual vs Fitted' : 'Data Aktual') : 'Prediksi'}
                </p>
            </div>
        );
    }
    return null;
};

export default function ForecastDashboard() {
    const [mounted, setMounted] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [forecastData, setForecastData] = useState<ForecastPoint[]>([]);
    const [metadata, setMetadata] = useState<ForecastMetadata | null>(null);

    const fetchForecast = async () => {
        try {
            setLoading(true);
            setError(null);
            const res = await fetch('/api/forecast');
            const json = await res.json();
            if (json.error && !json.forecast?.length) {
                setError(json.error);
                return;
            }
            setForecastData(json.forecast ?? []);
            setMetadata(json.metadata ?? null);
        } catch {
            setError('Gagal memuat data prediksi');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        setMounted(true);
        fetchForecast();

        let timeoutId: NodeJS.Timeout;

        const channel = supabase
            .channel('realtime_forecast')
            .on(
                'postgres_changes',
                {
                    event: '*',
                    schema: 'public',
                    table: 'tb_prediksi_hourly',
                },
                () => {
                    clearTimeout(timeoutId);
                    timeoutId = setTimeout(() => {
                        fetchForecast();
                    }, 2000);
                }
            )
            .subscribe();

        return () => {
            clearTimeout(timeoutId);
            supabase.removeChannel(channel);
        };
    }, []);

    if (!mounted) return null;

    const trendIcon = metadata?.trendDirection === 'naik'
        ? <TrendingUp className="w-4 h-4 text-rose-500" />
        : metadata?.trendDirection === 'turun'
            ? <TrendingDown className="w-4 h-4 text-emerald-500" />
            : <Minus className="w-4 h-4 text-slate-400" />;

    const trendColor = metadata?.trendDirection === 'naik'
        ? 'text-rose-600' : metadata?.trendDirection === 'turun'
            ? 'text-emerald-600' : 'text-slate-500';

    const renderChart = (
        pollutant: 'pm25' | 'pm10' | 'co',
        colors: { hist: string; fore: string },
        label: string,
        unit: string,
    ) => {
        const upperKey = `${pollutant}_upper` as keyof ForecastPoint;
        const lowerKey = `${pollutant}_lower` as keyof ForecastPoint;

        return (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-3 h-[250px] bg-slate-50 rounded-xl p-4 border border-slate-100">
                    <h4 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: colors.hist }}>{label}</h4>
                    <ResponsiveContainer width="100%" height="90%">
                        <ComposedChart data={forecastData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                            <defs>
                                <linearGradient id={`hist-${pollutant}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={colors.hist} stopOpacity={0.25} />
                                    <stop offset="95%" stopColor={colors.hist} stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id={`fore-${pollutant}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={colors.fore} stopOpacity={0.2} />
                                    <stop offset="95%" stopColor={colors.fore} stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                            <XAxis dataKey="time" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={30} />
                            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
                            <Tooltip content={<CustomTooltip />} />

                            <Line type="monotone" dataKey={(d: ForecastPoint) => !d.isHistorical && d[upperKey] != null ? d[upperKey] : null} stroke={colors.fore} strokeWidth={1} strokeDasharray="3 3" dot={false} connectNulls={false} name="Upper Bound" />
                            <Line type="monotone" dataKey={(d: ForecastPoint) => !d.isHistorical && d[lowerKey] != null ? d[lowerKey] : null} stroke={colors.fore} strokeWidth={1} strokeDasharray="3 3" dot={false} connectNulls={false} name="Lower Bound" />

                            <Area type="monotone" dataKey={(d: ForecastPoint) => d.isHistorical ? d[pollutant] : null} stroke={colors.hist} strokeWidth={2} fillOpacity={1} fill={`url(#hist-${pollutant})`} name={`Aktual ${label}`} connectNulls={true} />
                            <Line type="monotone" dataKey={(d: ForecastPoint) => d.isFitted ? d[`${pollutant}_fitted` as keyof ForecastPoint] as number : null} stroke="#8b5cf6" strokeWidth={2} strokeDasharray="4 3" dot={false} connectNulls={true} name={`Fitted ${label}`} />
                            <Area type="monotone" dataKey={(d: ForecastPoint) => !d.isHistorical ? d[pollutant] : null} stroke={colors.fore} strokeWidth={2} strokeDasharray="5 3" fillOpacity={1} fill={`url(#fore-${pollutant})`} name={`Prediksi ${label}`} connectNulls={true} />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
                <div className="flex flex-col gap-3">
                    <div className="p-3 rounded-xl border" style={{ backgroundColor: `${colors.hist}10`, borderColor: `${colors.hist}20` }}>
                        <h4 className="text-xs font-semibold uppercase mb-1" style={{ color: colors.hist }}>{label} Saat Ini</h4>
                        <p className="text-2xl font-black" style={{ color: colors.hist }}>{(() => {
                            const v = pollutant === 'pm25' ? metadata?.latestPm25 : pollutant === 'pm10' ? metadata?.latestPm10 : metadata?.latestCo;
                            return v?.toFixed(2) ?? '—';
                        })()}</p>
                        <p className="text-xs opacity-60" style={{ color: colors.hist }}>{unit}</p>
                    </div>
                    <div className="bg-orange-50 p-3 rounded-xl border border-orange-100">
                        <h4 className="text-xs font-semibold text-orange-500 uppercase mb-1">Prediksi 1 Jam</h4>
                        <p className="text-2xl font-black text-orange-700">{(() => {
                            const v = pollutant === 'pm25' ? metadata?.forecastedIn30min : pollutant === 'pm10' ? metadata?.forecastedIn30minPm10 : metadata?.forecastedIn30minCo;
                            return v?.toFixed(2) ?? '—';
                        })()}</p>
                        <p className="text-xs text-orange-400">{unit}</p>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm overflow-hidden">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
                <div>
                    <h3 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
                        <Zap className="w-5 h-5 text-indigo-500" />
                        Forecasting & Analitik
                    </h3>
                    <p className="text-sm text-slate-500 mt-1">
                        Prediksi PM2.5, PM10, dan CO menggunakan{' '}
                        <span className={cn(
                            "font-medium",
                            (metadata as any)?.usingXGBoost ? "text-emerald-600" : "text-indigo-600"
                        )}>
                            {metadata?.method ?? 'Enhanced XGBoost'}
                        </span>{' '}
                        based on Supabase real-time data.
                    </p>
                </div>
                <div className={cn("flex items-center gap-1 text-sm font-medium", trendColor)}>
                    {trendIcon}
                    {metadata?.trendDirection === 'naik' ? 'Meningkat' : metadata?.trendDirection === 'turun' ? 'Menurun' : 'Stabil'}
                </div>
            </div>

            {loading ? (
                <div className="flex items-center justify-center h-[300px]">
                    <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-500/30 border-t-indigo-600" />
                </div>
            ) : error ? (
                <div className="flex flex-col items-center justify-center h-[300px] text-slate-400 gap-3">
                    <AlertTriangle className="w-10 h-10 text-amber-400" />
                    <p className="text-sm">{error}</p>
                    <button onClick={fetchForecast} className="text-sm text-indigo-600 hover:underline">Coba lagi</button>
                </div>
            ) : (
                <div className="space-y-8">
                    {renderChart('pm25', { hist: '#6366f1', fore: '#f97316' }, 'PM2.5', 'µg/m³')}
                    {renderChart('pm10', { hist: '#10b981', fore: '#f97316' }, 'PM10', 'µg/m³')}
                    {renderChart('co', { hist: '#f59e0b', fore: '#f97316' }, 'CO', 'ppb')}
                </div>
            )}
        </div>
    );
}
