import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import { classify } from '@/lib/rf-classifier';

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

        const ml = classify(pm25, pm10, co, no2, o3);
        const cat = getCategory(ml.bp_ispu);

        return NextResponse.json({
            category: cat.label,
            ispu: ml.bp_ispu,
            color: cat.color,
            dominant: ml.bp_dominant,
            subIspu: {
                pm25: Math.round(concISPI(pm25, [[0,15.5,0,50],[15.5,55.4,50,100],[55.4,150.4,100,200],[150.4,250.4,200,300],[250.4,500,300,500]]) * 10) / 10,
                pm10: Math.round(concISPI(pm10, [[0,50,0,50],[50,150,50,100],[150,350,100,200],[350,420,200,300],[420,500,300,500]]) * 10) / 10,
                co: Math.round(concISPI(co, [[0,4000,0,50],[4000,8000,50,100],[8000,15000,100,200],[15000,30000,200,300],[30000,45000,300,500]]) * 10) / 10,
            },
            ml_confidence: ml.ml_confidence,
            ml_category: ml.ml_category,
            probabilities: ml.probabilities,
            features: { pm25, pm10, co, no2, o3 },
            method: 'ISPU Breakpoint',
            robustness: 'Random Forest',
            timestamp: row.created_at,
        });
    } catch (err) {
        console.error('[/api/classify]', err);
        const msg = err instanceof Error ? err.message : 'Unknown error';
        return NextResponse.json({ error: 'Internal server error', detail: msg }, { status: 500 });
    }
}

function concISPI(val: number, bp: number[][]): number {
    if (val <= 0) return 0;
    for (const [cl, ch, il, ih] of bp) {
        if (val <= ch) return il + (val - cl) / (ch - cl) * (ih - il);
    }
    return bp[bp.length - 1][3];
}
