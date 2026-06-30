import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import { execSync } from 'child_process';
import path from 'path';

const SCRIPT_PATH = path.resolve(process.cwd(), 'ml_model', 'classify.py');

const CATEGORIES = [
    { min: 0, max: 50, label: 'Baik', color: '#10b981', bg: 'bg-emerald-500', bgLight: 'bg-emerald-50', text: 'text-emerald-700' },
    { min: 51, max: 100, label: 'Sedang', color: '#3b82f6', bg: 'bg-blue-500', bgLight: 'bg-blue-50', text: 'text-blue-700' },
    { min: 101, max: 200, label: 'Tidak Sehat', color: '#f59e0b', bg: 'bg-amber-500', bgLight: 'bg-amber-50', text: 'text-amber-700' },
    { min: 201, max: 300, label: 'Sangat Tidak Sehat', color: '#ef4444', bg: 'bg-red-500', bgLight: 'bg-red-50', text: 'text-red-700' },
    { min: 301, max: 500, label: 'Berbahaya', color: '#7c3aed', bg: 'bg-purple-500', bgLight: 'bg-purple-50', text: 'text-purple-700' },
];

function getCategory(ispu: number) {
    return CATEGORIES.find(c => ispu >= c.min && ispu <= c.max) ?? CATEGORIES[0];
}

function ispuFromFeatures(pm25: number, pm10: number, co: number): { ispu: number; subIspu: Record<string, number>; dominant: string } {
    type BP = number[][];
    const toISPI = (val: number, bp: BP): number => {
        if (val <= 0) return 0;
        for (const [cl, ch, il, ih] of bp) {
            if (val <= ch) return il + (val - cl) / (ch - cl) * (ih - il);
        }
        return bp[bp.length - 1][3];
    };

    const BP_PM25: BP = [[0,15.5,0,50],[15.5,55.4,50,100],[55.4,150.4,100,200],[150.4,250.4,200,300],[250.4,500,300,500]];
    const BP_PM10: BP = [[0,50,0,50],[50,150,50,100],[150,350,100,200],[350,420,200,300],[420,500,300,500]];
    const BP_CO: BP = [[0,4000,0,50],[4000,8000,50,100],[8000,15000,100,200],[15000,30000,200,300],[30000,45000,300,500]];

    const subIspu: Record<string, number> = {
        pm25: Math.round(toISPI(pm25, BP_PM25) * 10) / 10,
        pm10: Math.round(toISPI(pm10, BP_PM10) * 10) / 10,
        co: Math.round(toISPI(co, BP_CO) * 10) / 10,
    };

    const ispu = Math.round(Math.max(...Object.values(subIspu)) * 10) / 10;
    const dominant = Object.entries(subIspu).reduce((a, b) => b[1] > a[1] ? b : a)[0];
    return { ispu, subIspu, dominant };
}

export async function GET() {
    try {
        const { data, error } = await supabase
            .from('tb_konsentrasi_gas')
            .select('pm25_ugm3, pm10_ugm3, co_ugm3, no2_ugm3, o3_ugm3, created_at')
            .order('created_at', { ascending: false })
            .limit(1);

        if (error) throw error;
        if (!data || data.length === 0) {
            return NextResponse.json({ error: 'No data' }, { status: 404 });
        }

        const row = data[0];
        const pm25 = Number(row.pm25_ugm3) || 0;
        const pm10 = Number(row.pm10_ugm3) || 0;
        const co = Number(row.co_ugm3) || 0;
        const no2 = Number(row.no2_ugm3) || 0;
        const o3 = Number(row.o3_ugm3) || 0;

        // ── ISPU breakpoint (kategori resmi) ──
        const fallback = ispuFromFeatures(pm25, pm10, co);
        const fallbackCat = getCategory(fallback.ispu);

        // ── RF robustness layer (confidence + risk score) ──
        let mlConfidence = 0;
        let mlProbabilities: Record<string, number> = {};
        let mlCategory = fallbackCat.label;
        try {
            const py = execSync(
                `python "${SCRIPT_PATH}" ${pm25} ${pm10} ${co} ${no2} ${o3}`,
                { timeout: 15000, encoding: 'utf-8' }
            );
            const ml = JSON.parse(py.trim());
            if (!ml.error) {
                mlConfidence = ml.ml_confidence;
                mlProbabilities = ml.probabilities;
                mlCategory = ml.ml_category;
            }
        } catch (_e) {
            // ML gagal — lanjut tanpa robustness info
        }

        const result: Record<string, unknown> = {
            category: fallbackCat.label,
            ispu: fallback.ispu,
            color: fallbackCat.color,
            dominant: fallback.dominant,
            subIspu: fallback.subIspu,
            ml_confidence: mlConfidence,
            ml_category: mlCategory,
            probabilities: mlProbabilities,
            features: { pm25, pm10, co, no2, o3 },
            method: 'ISPU Breakpoint',
            robustness: 'Random Forest',
        };
        result.timestamp = row.created_at;
        return NextResponse.json(result);
    } catch (err) {
        console.error('[/api/classify]', err);
        const msg = err instanceof Error ? err.message : 'Unknown error';
        return NextResponse.json({ error: 'Internal server error', detail: msg }, { status: 500 });
    }
}
