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
