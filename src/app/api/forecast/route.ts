import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function GET() {
    try {
        // Fetch predictions from tb_prediksi_hourly - latest batch only
        const { data: latestForecast } = await supabase
            .from('tb_prediksi_hourly')
            .select('forecast_at')
            .order('forecast_at', { ascending: false })
            .limit(1);
        
        let predData: any[] = [];
        if (latestForecast && latestForecast.length > 0) {
            const { data: filteredPred } = await supabase
                .from('tb_prediksi_hourly')
                .select('*')
                .eq('forecast_at', latestForecast[0].forecast_at)
                .order('target_at', { ascending: true });
            predData = filteredPred || [];
        }

        // Fetch historical data - ambil 60 menit terakhir
        const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
        const { data: histData } = await supabase
            .from('tb_konsentrasi_gas')
            .select('pm25_ugm3, pm10_ugm3, co_ugm3, created_at')
            .gte('created_at', oneHourAgo)
            .order('created_at', { ascending: true })
            .limit(200);

        // Sort by timestamp to ensure proper ordering
        const sortedHist = (histData ?? []).sort((a, b) => 
            new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        );

        // Helper for historical - add 7 hours for WITA
        const extractTimeHistorical = (isoString: string) => {
            const match = isoString.match(/T(\d{2}):(\d{2})/);
            if (match) {
                let hour = parseInt(match[1]) + 7;
                if (hour >= 24) hour -= 24;
                return `${hour.toString().padStart(2, '0')}.${match[2]}`;
            }
            return '';
        };

        // Helper for predictions - convert UTC to WIB (+7)
        const extractTimePrediction = (isoString: string) => {
            const match = isoString.match(/T(\d{2}):(\d{2})/);
            if (match) {
                let hour = parseInt(match[1]) + 7;
                if (hour >= 24) hour -= 24;
                return `${hour.toString().padStart(2, '0')}.${match[2]}`;
            }
            return '';
        };

        const histPoints = sortedHist.map(d => ({
            time: extractTimeHistorical(d.created_at),
            pm25: Number(d.pm25_ugm3?.toFixed(2) || 0),
            pm10: Number(d.pm10_ugm3?.toFixed(2) || 0),
            co: Number(d.co_ugm3?.toFixed(2) || 0),
            isHistorical: true,
            timestamp: new Date(d.created_at).getTime(),
        }));

        // Predictions - tanpa konversi timezone
        const forecastPoints = (predData ?? []).map(r => ({
            time: extractTimePrediction(r.target_at),
            pm25: Number(r.pm25_pred?.toFixed(2) || 0),
            pm25_upper: r.pm25_upper ? Number(r.pm25_upper.toFixed(2)) : undefined,
            pm25_lower: r.pm25_lower ? Number(r.pm25_lower.toFixed(2)) : undefined,
            pm10: Number(r.pm10_pred?.toFixed(2) || 0),
            pm10_upper: r.pm10_upper ? Number(r.pm10_upper.toFixed(2)) : undefined,
            pm10_lower: r.pm10_lower ? Number(r.pm10_lower.toFixed(2)) : undefined,
            co: Number((r.co_pred || 0).toFixed(2)),
            co_upper: r.co_upper ? Number(r.co_upper.toFixed(2)) : undefined,
            co_lower: r.co_lower ? Number(r.co_lower.toFixed(2)) : undefined,
            isHistorical: false,
            timestamp: new Date(r.target_at).getTime(),
        }));

        // Fetch fitted values (in-sample predictions) & merge into historical points
        const { data: fittedData } = await supabase
            .from('tb_fitted_values')
            .select('timestamp, pm25_fitted, pm10_fitted, co_fitted')
            .gte('timestamp', oneHourAgo)
            .order('timestamp', { ascending: true });

        const fittedMap = new Map<string, { pm25_fitted: number; pm10_fitted: number; co_fitted: number }>();
        for (const f of (fittedData ?? [])) {
            const ts = new Date(f.timestamp).getTime().toString();
            fittedMap.set(ts, {
                pm25_fitted: Number(Number(f.pm25_fitted).toFixed(2) || 0),
                pm10_fitted: Number(Number(f.pm10_fitted).toFixed(2) || 0),
                co_fitted: Number(Number(f.co_fitted).toFixed(2) || 0),
            });
        }

        // Merge fitted into historical points where timestamps match
        for (const hp of histPoints) {
            const match = fittedMap.get(hp.timestamp.toString());
            if (match) {
                (hp as any).pm25_fitted = match.pm25_fitted;
                (hp as any).pm10_fitted = match.pm10_fitted;
                (hp as any).co_fitted = match.co_fitted;
                (hp as any).isFitted = true;
            }
        }

        const allPoints = [...histPoints, ...forecastPoints];

        // Get latest values
        const latestPm25 = histPoints.length > 0 ? histPoints[histPoints.length - 1].pm25 : 0;
        const latestPm10 = histPoints.length > 0 ? histPoints[histPoints.length - 1].pm10 : 0;
        const latestCo = histPoints.length > 0 ? Number(histPoints[histPoints.length - 1].co.toFixed(2)) : 0;
        
        const forecastedIn30Pm25 = forecastPoints.length >= 30 ? forecastPoints[29].pm25 : latestPm25;
        const forecastedIn30Pm10 = forecastPoints.length >= 30 ? forecastPoints[29].pm10 : latestPm10;
        const forecastedIn30Co = forecastPoints.length >= 30 ? forecastPoints[29].co : latestCo;

        // Trend calculation (based on PM2.5)
        const recentValues = histPoints.slice(-10).map(p => p.pm25);
        const avgStart = recentValues.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
        const avgEnd = recentValues.slice(5).reduce((a, b) => a + b, 0) / 5;
        const trendDirection = avgEnd - avgStart > 1 ? 'naik' : avgEnd - avgStart < -1 ? 'turun' : 'stabil';

        const method = predData && predData.length > 0 
            ? 'XGBoost Forecasting'
            : 'Fallback';

        return NextResponse.json({
            forecast: allPoints,
            metadata: {
                dataPoints: histPoints.length,
                latestPm25,
                latestPm10,
                latestCo,
                forecastedIn30min: forecastedIn30Pm25,
                forecastedIn30minPm10: forecastedIn30Pm10,
                forecastedIn30minCo: forecastedIn30Co,
                trendDirection,
                method,
                usingXGBoost: forecastPoints.length > 0,
            },
        });
    } catch (err) {
        console.error('[/api/forecast]', err);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}