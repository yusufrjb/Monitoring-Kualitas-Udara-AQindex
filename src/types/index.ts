export interface RealtimeData {
    pm25: number;
    pm10: number;
    no2: number;
    co: number;
    o3: number;
    temperature: number;
    humidity: number;
    sensorOffline?: boolean;
}

export interface ForecastPoint {
    time: string;
    pm25: number;
    pm10: number;
    co: number;
    isHistorical: boolean;
    isFitted?: boolean;
    timestamp: number;
    pm25_upper?: number;
    pm25_lower?: number;
    pm10_upper?: number;
    pm10_lower?: number;
    co_upper?: number;
    co_lower?: number;
    pm25_actual?: number | null;
    pm10_actual?: number | null;
    co_actual?: number | null;
    pm25_fitted?: number;
    pm10_fitted?: number;
    co_fitted?: number;
}

export interface ForecastMetadata {
    dataPoints: number;
    latestPm25: number;
    latestPm10?: number;
    latestCo?: number;
    forecastedIn30min: number;
    forecastedIn30minPm10?: number;
    forecastedIn30minCo?: number;
    trendDirection: 'naik' | 'turun' | 'stabil';
    method: string;
}
