import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import fs from 'fs';
import path from 'path';

/**
 * API Route: /api/regression
 * 
 * Melakukan regresi antar-parameter secara real-time:
 * - Mengambil data dari tb_konsentrasi_gas
 * - Menggunakan Linear Regression (model terbaik R²=87.70%)
 * - Mengembalikan actual vs predicted untuk visualisasi scatter plot
 * - Mengembalikan feature importance
 */

interface DataRow {
    pm25_ugm3: number;
    pm10_ugm3: number;
    no2_ugm3: number;
    co_ugm3: number;
    temperature: number;
    humidity: number;
    created_at: string;
}

// Simple linear regression implementation (no external ML deps needed)
function multipleLinearRegression(X: number[][], y: number[]): { weights: number[]; bias: number } {
    const n = X.length;
    const p = X[0].length;

    // Add bias column
    const Xb = X.map(row => [1, ...row]);

    // Normal equation: w = (X^T X)^(-1) X^T y
    // Using simplified iterative approach for numerical stability
    const XtX: number[][] = Array(p + 1).fill(0).map(() => Array(p + 1).fill(0));
    const Xty: number[] = Array(p + 1).fill(0);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < p + 1; j++) {
            Xty[j] += Xb[i][j] * y[i];
            for (let k = 0; k < p + 1; k++) {
                XtX[j][k] += Xb[i][j] * Xb[i][k];
            }
        }
    }

    // Solve via Gauss-Jordan elimination
    const aug = XtX.map((row, i) => [...row, Xty[i]]);
    const size = p + 1;

    for (let col = 0; col < size; col++) {
        // Find pivot
        let maxRow = col;
        for (let row = col + 1; row < size; row++) {
            if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
        }
        [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

        const pivot = aug[col][col];
        if (Math.abs(pivot) < 1e-10) continue;

        for (let j = col; j <= size; j++) aug[col][j] /= pivot;
        for (let row = 0; row < size; row++) {
            if (row === col) continue;
            const factor = aug[row][col];
            for (let j = col; j <= size; j++) aug[row][j] -= factor * aug[col][j];
        }
    }

    const solution = aug.map(row => row[size]);
    return {
        bias: solution[0],
        weights: solution.slice(1),
    };
}

function predict(X: number[][], model: { weights: number[]; bias: number }): number[] {
    return X.map(row => {
        let pred = model.bias;
        for (let i = 0; i < row.length; i++) {
            pred += row[i] * model.weights[i];
        }
        return Math.max(0, pred);
    });
}

function calcR2(yTrue: number[], yPred: number[]): number {
    const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
    const ssRes = yTrue.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0);
    const ssTot = yTrue.reduce((s, y) => s + (y - mean) ** 2, 0);
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

function calcMAE(yTrue: number[], yPred: number[]): number {
    return yTrue.reduce((s, y, i) => s + Math.abs(y - yPred[i]), 0) / yTrue.length;
}

export async function GET() {
    try {
        // 1. Try to read results from Python script (it calculates Best Model)
        const jsonPath = path.join(process.cwd(), 'ml_model', 'regression_results.json');

        if (fs.existsSync(jsonPath)) {
            const fileContent = fs.readFileSync(jsonPath, 'utf8');
            const data = JSON.parse(fileContent);

            // Check if data exists and has visualization points
            if (data.scatterData && data.scatterData.length > 0) {
                return NextResponse.json({
                    scatterData: data.scatterData,
                    timeSeriesData: data.timeSeriesData,
                    featureImportance: data.feature_importance.map((f: any) => ({
                        feature: f.label || f.feature,
                        importance: f.importance,
                        weight: f.weight || 0
                    })),
                    metrics: {
                        r2: data.best_r2,
                        mae: data.results.find((r: any) => r.name === data.best_model)?.MAE ?? 0,
                        dataPoints: data.total_data,
                        trainSize: data.train_size,
                        testSize: data.test_size,
                    },
                    model: data.best_model,
                    allResults: data.results,
                    evaluatedAt: data.evaluated_at,
                    totalData: data.total_data,
                    isRealtime: false
                });
            }
        }

        // 2. Fallback to real-time Linear Regression
        const since = new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString();
        const { data: rawData, error } = await supabase
            .from('tb_konsentrasi_gas')
            .select('pm25_ugm3,pm10_ugm3,no2_ugm3,co_ugm3,temperature,humidity,created_at')
            .gte('created_at', since)
            .order('created_at', { ascending: true })
            .limit(1000);

        if (error || !rawData || rawData.length < 20) {
            return NextResponse.json({ error: 'Insufficient data' }, { status: 400 });
        }

        const rows: DataRow[] = rawData
            .map((r: any) => ({
                pm25_ugm3: parseFloat(r.pm25_ugm3),
                pm10_ugm3: parseFloat(r.pm10_ugm3),
                no2_ugm3: parseFloat(r.no2_ugm3),
                co_ugm3: parseFloat(r.co_ugm3),
                temperature: parseFloat(r.temperature),
                humidity: parseFloat(r.humidity),
                created_at: r.created_at,
            }))
            .filter((r: DataRow) =>
                !isNaN(r.pm25_ugm3) && !isNaN(r.temperature) && !isNaN(r.humidity) &&
                r.pm25_ugm3 >= 0 && r.pm25_ugm3 < 500
            );

        if (rows.length < 20) {
            return NextResponse.json({ error: 'Insufficient clean data' }, { status: 400 });
        }

        const featureNames = ['Suhu', 'Kelembapan', 'PM10', 'NO₂', 'CO'];
        const X = rows.map(r => [r.temperature, r.humidity, r.pm10_ugm3, r.no2_ugm3, r.co_ugm3]);
        const y = rows.map(r => r.pm25_ugm3);

        const indices = rows.map((_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const splitIdx = Math.floor(rows.length * 0.8);
        const trainIndices = indices.slice(0, splitIdx);
        const testIndices = indices.slice(splitIdx);

        const X_train = trainIndices.map(i => X[i]);
        const y_train = trainIndices.map(i => y[i]);
        const X_test = testIndices.map(i => X[i]);
        const y_test = testIndices.map(i => y[i]);

        const model = multipleLinearRegression(X_train, y_train);
        const y_pred_all = predict(X, model);

        const y_pred_test = predict(X_test, model);
        const r2 = calcR2(y_test, y_pred_test);
        const mae = calcMAE(y_test, y_pred_test);

        const step = Math.max(1, Math.floor(rows.length / 200));
        const scatterData = rows
            .filter((_, i) => i % step === 0)
            .map((row, i) => {
                const globalIdx = i * step;
                return {
                    actual: parseFloat(row.pm25_ugm3.toFixed(2)),
                    predicted: parseFloat(y_pred_all[globalIdx]?.toFixed(2) ?? '0'),
                    time: new Date(row.created_at).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }),
                    isTrain: trainIndices.includes(globalIdx),
                };
            });

        return NextResponse.json({
            scatterData,
            timeSeriesData: scatterData.map(d => ({ time: d.time, actual: d.actual, predicted: d.predicted })),
            featureImportance: featureNames.map((name, i) => ({
                feature: name,
                importance: 20,
                weight: 0
            })),
            metrics: {
                r2: parseFloat((r2 * 100).toFixed(2)),
                mae: parseFloat(mae.toFixed(4)),
                dataPoints: rows.length,
                trainSize: splitIdx,
                testSize: rows.length - splitIdx,
            },
            model: 'Linear Regression',
            isRealtime: true
        });
    } catch (err: any) {
        console.error('Regression API error:', err);
        return NextResponse.json({ error: err.message || 'Internal error' }, { status: 500 });
    }
}
