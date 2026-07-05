import rfModel from './rf_model.json';

export interface ClassificationResult {
  bp_category: string;
  bp_ispu: number;
  bp_dominant: string;
  ml_category: string;
  ml_confidence: number;
  probabilities: Record<string, number>;
  features: Record<string, number>;
}

type TreeNode = {
  id: number;
  feature: number;
  threshold: number;
  children_left: number;
  children_right: number;
  value: number[][];
  n_node_samples: number;
  impurity: number;
};

type RFModel = {
  classes: string[];
  n_features: number;
  feature_names: string[];
  n_classes: number;
  n_estimators: number;
  trees: TreeNode[][];
};

const model = rfModel as unknown as RFModel;

function predictTree(tree: TreeNode[], features: number[]): number[] {
  let node = tree[0];
  while (node.children_left !== -1) {
    const val = features[node.feature] ?? 0;
    if (val <= node.threshold) {
      node = tree[node.children_left];
    } else {
      node = tree[node.children_right];
    }
  }
  return node.value[0];
}

const BP_PM25 = [[0, 15.5, 0, 50], [15.5, 55.4, 50, 100], [55.4, 150.4, 100, 200], [150.4, 250.4, 200, 300], [250.4, 500, 300, 500]];
const BP_PM10 = [[0, 50, 0, 50], [50, 150, 50, 100], [150, 350, 100, 200], [350, 420, 200, 300], [420, 500, 300, 500]];
const BP_CO = [[0, 4000, 0, 50], [4000, 8000, 50, 100], [8000, 15000, 100, 200], [15000, 30000, 200, 300], [30000, 45000, 300, 500]];
const BP_NO2 = [[0, 80, 0, 50], [80, 200, 50, 100], [200, 1130, 100, 200], [1130, 2260, 200, 300], [2260, 3000, 300, 500]];
const BP_O3 = [[0, 120, 0, 50], [120, 235, 50, 100], [235, 400, 100, 200], [400, 800, 200, 300], [800, 1000, 300, 500]];

const LABELS = ["Baik", "Berbahaya", "Sangat Tidak Sehat", "Sedang", "Tidak Sehat"];
const FEATURES = ["pm25_ugm3", "pm10_ugm3", "co_ugm3", "no2_ugm3", "o3_ugm3"];

function concToISPI(val: number, bp: number[][]): number {
  if (val <= 0) return 0;
  for (const [cl, ch, il, ih] of bp) {
    if (val <= ch) return il + (val - cl) / (ch - cl) * (ih - il);
  }
  return bp[bp.length - 1][3];
}

function ispiToLabel(ispi: number): string {
  if (ispi <= 50) return LABELS[0];
  if (ispi <= 100) return LABELS[1];
  if (ispi <= 200) return LABELS[2];
  if (ispi <= 300) return LABELS[3];
  return LABELS[4];
}

export function classify(pm25: number, pm10: number, co: number, no2: number, o3: number): ClassificationResult {
  const ispis = [
    concToISPI(pm25, BP_PM25),
    concToISPI(pm10, BP_PM10),
    concToISPI(co, BP_CO),
    concToISPI(no2, BP_NO2),
    concToISPI(o3, BP_O3),
  ];
  const maxISPU = Math.max(...ispis);
  const bpLabel = ispiToLabel(maxISPU);
  const dominant = FEATURES[ispis.indexOf(maxISPU)];

  const features = [pm25, pm10, co, no2, o3];
  let totalProb: number[] = new Array(model.n_classes).fill(0);

  for (const tree of model.trees) {
    const leafProb = predictTree(tree, features);
    for (let i = 0; i < model.n_classes; i++) {
      totalProb[i] += leafProb[i];
    }
  }

  const proba = totalProb.map(p => p / model.n_estimators);
  const maxProb = Math.max(...proba);
  const mlConfidence = Math.round(maxProb * 10000) / 10000;
  const mlLabelIdx = proba.indexOf(maxProb);
  const mlLabel = model.classes[mlLabelIdx];

  const probabilities: Record<string, number> = {};
  for (let i = 0; i < model.n_classes; i++) {
    probabilities[model.classes[i]] = Math.round(proba[i] * 10000) / 10000;
  }

  return {
    bp_category: bpLabel,
    bp_ispu: Math.round(maxISPU * 100) / 100,
    bp_dominant: dominant,
    ml_category: mlLabel,
    ml_confidence: mlConfidence,
    probabilities,
    features: {
      pm25_ugm3: pm25,
      pm10_ugm3: pm10,
      co_ugm3: co,
      no2_ugm3: no2,
      o3_ugm3: o3,
    },
  };
}
