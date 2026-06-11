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
            pm10: Number(r.pm10_pred?.toFixed(2) || 0),
            co: Number((r.co_pred || 0).toFixed(2)),
            isHistorical: false,
            timestamp: new Date(r.target_at).getTime(),
        }));

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