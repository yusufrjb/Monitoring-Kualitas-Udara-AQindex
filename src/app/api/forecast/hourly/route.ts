import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import { execSync } from 'child_process';
import path from 'path';

const SCRIPT_PATH = path.resolve(process.cwd(), 'ml_model', 'predict_time_series.py');

const CAT_COLORS: Record<string, string> = {
    "Baik": "#10b981",
    "Sedang": "#3b82f6",
    "Tidak Sehat": "#f59e0b",
    "Sangat Tidak Sehat": "#ef4444",
    "Berbahaya": "#7c3aed",
};

function concToISPI(val: number, bp: number[][]): number {
    if (val <= 0) return 0;
    for (const [cl, ch, il, ih] of bp) {
        if (val <= ch) return il + (val - cl) / (ch - cl) * (ih - il);
    }
    return bp[bp.length - 1][3];
}

function getISPI(pm25: number, pm10: number, co: number): { ispu: number; label: string; dominant: string } {
    const BP_PM25: number[][] = [[0,15.5,0,50],[15.5,55.4,50,100],[55.4,150.4,100,200],[150.4,250.4,200,300],[250.4,500,300,500]];
    const BP_PM10: number[][] = [[0,50,0,50],[50,150,50,100],[150,350,100,200],[350,420,200,300],[420,500,300,500]];
    const BP_CO: number[][]   = [[0,4000,0,50],[4000,8000,50,100],[8000,15000,100,200],[15000,30000,200,300],[30000,45000,300,500]];

    // Hanya PM2.5, PM10, CO yang diprediksi - dominant hanya dari ketiganya
    const ispis: Record<string, number> = {
        "PM2.5": concToISPI(pm25, BP_PM25),
        "PM10": concToISPI(pm10, BP_PM10),
        "CO": concToISPI(co, BP_CO),
    };
    const maxISPU = Math.max(...Object.values(ispis));
    const label = maxISPU <= 50 ? "Baik" : maxISPU <= 100 ? "Sedang" : maxISPU <= 200 ? "Tidak Sehat" : maxISPU <= 300 ? "Sangat Tidak Sehat" : "Berbahaya";
    const dominant = Object.entries(ispis).reduce((a, b) => b[1] > a[1] ? b : a)[0];
    return { ispu: Math.round(maxISPU * 10) / 10, label, dominant };
}

function ispiToColor(label: string): string {
    return CAT_COLORS[label] ?? "#94a3b8";
}

function holtWintersForecast(series: number[], nSteps: number): number[] {
    const alpha = 0.3, beta = 0.1;
    let level = series[0], trend = series[1] - series[0];
    const forecasts: number[] = [];
    for (let i = 1; i < series.length; i++) {
        const newLevel = alpha * series[i] + (1 - alpha) * (level + trend);
        trend = beta * (newLevel - level) + (1 - beta) * trend;
        level = newLevel;
    }
    for (let i = 1; i <= nSteps; i++) {
        forecasts.push(Math.max(0, level + i * trend + (Math.random() - 0.5) * level * 0.05));
    }
    return forecasts;
}

export async function GET() {
    try {
        const since = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString();
        const { data: histData, error: histErr } = await supabase
            .from('tb_konsentrasi_gas')
            .select('pm25_ugm3, pm10_ugm3, co_ugm3, no2_ugm3, o3_ugm3, created_at')
            .gte('created_at', since)
            .order('created_at', { ascending: false })
            .limit(60);

        if (histErr) throw histErr;

        const rows = (histData || []).reverse();
        if (rows.length === 0) {
            return NextResponse.json({ error: 'No historical data' }, { status: 404 });
        }

        const lastRow = rows[rows.length - 1];
        const now = new Date();
        const historical = rows.map((r, i) => {
            const t = new Date(r.created_at);
            const pm25 = Number(r.pm25_ugm3) || 0;
            const pm10 = Number(r.pm10_ugm3) || 0;
            const co = Number(r.co_ugm3) || 0;
            const no2 = Number(r.no2_ugm3) || 0;
            const o3 = Number(r.o3_ugm3) || 0;
            const { ispu, label, dominant } = getISPI(pm25, pm10, co);
            return {
                target_at: t.toISOString(),
                pm25: Math.round(pm25 * 100) / 100,
                pm10: Math.round(pm10 * 100) / 100,
                co: Math.round(co * 100) / 100,
                isHistorical: true,
                ispu,
                category: label,
                dominant,
                color: ispiToColor(label),
            };
        });

        let forecast_result: { forecast: any[] } | null = null;
        let forecast_method = "";

        try {
            const output = execSync(`python "${SCRIPT_PATH}" --json`, {
                timeout: 60000,
                encoding: 'utf-8',
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: path.resolve(process.cwd()),
            });
            const parsed = JSON.parse(output.trim());
            if (parsed.forecast && parsed.forecast.length > 0) {
                forecast_result = parsed;
                forecast_method = parsed.method || "XGBoost + Pola Harian";
            }
        } catch {
            // fallback to Holt-Winters below
        }

        const pm25Series = rows.map(r => Number(r.pm25_ugm3) || 0);
        const pm10Series = rows.map(r => Number(r.pm10_ugm3) || 0);
        const coSeries = rows.map(r => Number(r.co_ugm3) || 0);
        const no2Latest = Number(lastRow.no2_ugm3) || 50;
        const o3Latest = Number(lastRow.o3_ugm3) || 40;

        const N_STEPS = 60;
        const forecast_pm25 = forecast_result?.forecast
            ? forecast_result.forecast.map(f => f.pm25)
            : holtWintersForecast(pm25Series, N_STEPS);
        const forecast_pm10 = forecast_result?.forecast
            ? forecast_result.forecast.map(f => f.pm10)
            : holtWintersForecast(pm10Series, N_STEPS);
        const forecast_co = forecast_result?.forecast
            ? forecast_result.forecast.map(f => f.co)
            : holtWintersForecast(coSeries, N_STEPS);
        const forecast = [];
        for (let i = 0; i < N_STEPS; i++) {
            const targetAt = new Date(now.getTime() + (i + 1) * 60000);
            const pm25Val = Math.round((forecast_pm25[i] || 0) * 100) / 100;
            const pm10Val = Math.round((forecast_pm10[i] || 0) * 100) / 100;
            const coVal = Math.round((forecast_co[i] || 0) * 100) / 100;

            const { ispu, label, dominant } = getISPI(pm25Val, pm10Val, coVal);
            const color = ispiToColor(label);

            forecast.push({
                target_at: targetAt.toISOString(),
                pm25: pm25Val,
                pm10: pm10Val,
                co: coVal,
                isHistorical: false,
                ispu,
                category: label,
                dominant,
                color,
            });
        }

        const latestISPU = historical[historical.length - 1]?.ispu ?? 0;
        const latestLabel = historical[historical.length - 1]?.category ?? "Sedang";
        const latestColor = historical[historical.length - 1]?.color ?? "#3b82f6";

        return NextResponse.json({
            historical,
            forecast,
            metadata: {
                latestPm25: pm25Series[pm25Series.length - 1],
                latestPm10: pm10Series[pm10Series.length - 1],
                latestCo: coSeries[coSeries.length - 1],
                latestISPU,
                latestCategory: latestLabel,
                latestColor,
                latestDominant: historical[historical.length - 1]?.dominant ?? "pm25",
                totalHistorical: historical.length,
                totalForecast: forecast.length,
                method: forecast_result ? forecast_method || "XGBoost + Pola Harian" : "Holt-Winters (fallback)",
            },
        });
    } catch (err) {
        console.error('[/api/forecast/hourly]', err);
        const msg = err instanceof Error ? err.message : 'Unknown error';
        return NextResponse.json({ error: 'Internal server error', detail: msg }, { status: 500 });
    }
}
