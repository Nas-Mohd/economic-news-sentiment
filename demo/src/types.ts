export type SentimentLabel = 0 | 1 | 2; // 0 = Positive, 1 = Negative, 2 = Neutral

export interface DatasetRow {
  text: string;
  aspect: string;
  label: SentimentLabel;
}

export interface MetricCardData {
  title: string;
  value: string | number;
  subtitle?: string;
}

export interface AspectScore {
  aspect: string;
  score: number; // Cosine similarity or confidence score
}

export interface SentimentConfidence {
  positive: number;
  negative: number;
  neutral: number;
}

export interface ParseResult {
  id: number;
  text: string;
  aspect: string;
  sentiment: string;
  confidence: number;
  probabilities?: Record<string, number>;
}
