import { DatasetRow, AspectScore, SentimentConfidence } from "./types";

export const MACRO_ASPECTS = [
  "Monetary Financial",
  "Inflation Prices",
  "Real Economic Activity",
  "Labor Consumption",
  "Fiscal Government",
  "External Sector"
];

// High-fidelity fallback dataset
export const DEMO_DATASET: DatasetRow[] = [
  {
    text: "The Federal Reserve raised interest rates by 25 basis points to combat lingering persistent inflation.",
    aspect: "Monetary Financial",
    label: 1 // Negative
  },
  {
    text: "Consumer spending jumped significantly this quarter, defying initial market expectations of a major slowdown.",
    aspect: "Labor Consumption",
    label: 0 // Positive
  },
  {
    text: "The country's overall trade balance registered a record surplus due to explosive high tech exports.",
    aspect: "External Sector",
    label: 0 // Positive
  },
  {
    text: "Strict budgetary rules introduced in the finance act helped shrink the fiscal deficits by 12 percent.",
    aspect: "Fiscal Government",
    label: 0 // Positive
  },
  {
    text: "Industrial input costs and consumer energy tariffs surged across main manufacturing hubs last month.",
    aspect: "Inflation Prices",
    label: 1 // Negative
  },
  {
    text: "Domestic GDP numbers indicate full structural contraction as multiple critical manufacturing sectors stagnated.",
    aspect: "Real Economic Activity",
    label: 1 // Negative
  },
  {
    text: "Unemployment claims decreased drastically to multi-decade historic lows across key structural states.",
    aspect: "Labor Consumption",
    label: 0 // Positive
  },
  {
    text: "The Central Bank Governor announced clear plans to expand high volume asset purchasing loops in July.",
    aspect: "Monetary Financial",
    label: 2 // Neutral
  },
  {
    text: "Heavy investments in public infrastructure was backed by extensive borrowing packages.",
    aspect: "Fiscal Government",
    label: 2 // Neutral
  },
  {
    text: "Import tariffs raised cross-border shipping friction, hurting logistics supply lines globally.",
    aspect: "External Sector",
    label: 1 // Negative
  }
];

// Custom keyword matrices for robust local simulation of NLP models
const ASPECT_KEYWORDS: Record<string, string[]> = {
  "Monetary Financial": ["fed", "federal", "reserve", "rate", "interest", "rates", "cut", "easing", "monetary", "central bank", "governor", "liquidity", "yield", "bond", "borrowing", "policy"],
  "Inflation Prices": ["inflation", "cpi", "ppi", "prices", "tariff", "tariffs", "cost", "costs", "wage", "energy", "oil", "food", "commodity", "pricing", "consumer price", "expensive"],
  "Real Economic Activity": ["gdp", "growth", "manufacturing", "industrial", "contraction", "expansion", "recession", "output", "pmi", "production", "sectors", "throughput", "infrastructure"],
  "Labor Consumption": ["jobs", "payrolls", "spending", "unemployment", "consumers", "consumer", "hiring", "demand", "retail", "wage", "labor", "employment", "payroll"],
  "Fiscal Government": ["fiscal", "budget", "deficit", "deficits", "tax", "taxes", "borrowing", "government", "spending", "stimulus", "debt", "treasury", "budgetary", "finance act"],
  "External Sector": ["trade", "export", "exports", "import", "imports", "tariffs", "balance", "currency", "exchange", "dollar", "cross-border", "shipping", "logistics", "global"]
};

/**
 * Calculates simulated cosine similarity for pages based on keyword intersections with cosine bounds.
 */
export function calculateSemanticSimilarities(text: string): AspectScore[] {
  const normalizedText = text.toLowerCase();
  
  return MACRO_ASPECTS.map((aspect) => {
    const keywords = ASPECT_KEYWORDS[aspect] || [];
    let matchCount = 0;
    
    // Exact match weights
    keywords.forEach((keyword) => {
      if (normalizedText.includes(keyword)) {
        matchCount += 1.5;
        // Boost for multi-word or core terms
        if (keyword === "fed" || keyword === "monetary" || keyword === "cpi" || keyword === "gdp" || keyword === "unemployment") {
          matchCount += 1.0;
        }
      }
    });

    // Base mock baseline cosine score (usually sentence embed similarity ranges from 0.05 to 0.70)
    let baseScore = 0.08 + Math.random() * 0.05;
    
    if (matchCount > 0) {
      baseScore += Math.min(0.62, matchCount * 0.15);
    }
    
    return {
      aspect,
      score: Math.min(0.94, baseScore)
    };
  });
}

export interface DebertaPredictionResult {
  aspect: string;
  confidence: number;
  probabilities: Record<string, number>;
}

/**
 * Simulate DeBERTa domain classifier output with independent Sigmoid probabilities
 */
export function simulateDebertaPrediction(text: string): DebertaPredictionResult {
  const similarities = calculateSemanticSimilarities(text);
  // Find highest scoring aspect
  let best = similarities[0];
  similarities.forEach((s) => {
    if (s.score > best.score) {
      best = s;
    }
  });

  // Scale the similarity to confidence level between 65% and 98%
  const confidence = 0.65 + (best.score * 0.35);

  // Independent multi-label Sigmoid probabilities (do not add up to 1)
  const probabilities: Record<string, number> = {};
  similarities.forEach((item) => {
    let p = item.score;
    if (item.aspect === best.aspect) {
      p = Math.min(0.98, confidence + (Math.random() * 0.04 - 0.02));
    } else {
      // Map similarities to independent baseline noise
      p = Math.max(0.03, Math.min(0.45, p * 0.6 + (Math.random() * 0.06 - 0.03)));
    }
    probabilities[item.aspect] = p;
  });

  return {
    aspect: best.aspect,
    confidence: Math.min(0.985, confidence),
    probabilities
  };
}

/**
 * Simulate global finbert (unpaired text sentiment)
 */
export function simulateGlobalFinbert(text: string): SentimentConfidence {
  const txt = text.toLowerCase();
  
  const positiveWords = ["growth", "increase", "rise", "positive", "strong", "higher", "benefit", "expansion", "rebound", "surplus", "gains", "jumped", "robust", "defying"];
  const negativeWords = ["fall", "drop", "decline", "cut", "inflation", "deficit", "slowdown", "risk", "debt", "contraction", "stagnated", "friction", "restrictive", "overhead"];

  let posCount = 0;
  let negCount = 0;

  positiveWords.forEach((pw) => {
    if (txt.includes(pw)) posCount++;
  });

  negativeWords.forEach((nw) => {
    if (txt.includes(nw)) negCount++;
  });

  // Default neutral bias
  let rawPos = 0.1 + posCount * 0.35;
  let rawNeg = 0.1 + negCount * 0.35;
  let rawNeu = 0.4;

  // Add small random perturbation to make it dynamic
  rawPos += Math.random() * 0.05;
  rawNeg += Math.random() * 0.05;

  const total = rawPos + rawNeg + rawNeu;
  return {
    positive: rawPos / total,
    negative: rawNeg / total,
    neutral: rawNeu / total
  };
}

/**
 * Simulate Aspect-Based custom FinBERT sentiment (paired)
 */
export function simulateAspectSentiment(text: string, aspect: string): SentimentConfidence {
  const txt = text.toLowerCase();
  const asp = aspect.toLowerCase();

  let biasPos = 0.1;
  let biasNeg = 0.1;

  // Aspect-contextualized rules:
  if (asp.includes("inflation") || asp.includes("prices")) {
    // For inflation, price increases (surged, high, rise) are typically negative macro signs but increases in policy are neutral
    if (txt.includes("rise") || txt.includes("high") || txt.includes("surged") || txt.includes("rising") || txt.includes("elevated")) {
      biasNeg += 0.75;
    } else if (txt.includes("decreased") || txt.includes("shrink") || txt.includes("lower")) {
      biasPos += 0.6;
    }
  } else if (asp.includes("labor") || asp.includes("consumption")) {
    if (txt.includes("unemployment") || txt.includes("job cuts") || txt.includes("slowdown")) {
      biasNeg += 0.8;
    } else if (txt.includes("jumped") || txt.includes("robust") || txt.includes("hiring") || txt.includes("increased") || txt.includes("payrolls")) {
      biasPos += 0.75;
    }
  } else if (asp.includes("monetary")) {
    if (txt.includes("restrictive") || txt.includes("restrictive policy") || txt.includes("raised") || txt.includes("tightening")) {
      biasNeg += 0.55; // tightening is typically bearish or negative contextually
    } else if (txt.includes("easing") || txt.includes("stimulus") || txt.includes("cut")) {
      biasPos += 0.6;
    }
  } else if (asp.includes("real economic")) {
    if (txt.includes("contraction") || txt.includes("stagnated") || txt.includes("slowdown") || txt.includes("recession")) {
      biasNeg += 0.8;
    } else if (txt.includes("growth") || txt.includes("robust") || txt.includes("expansion")) {
      biasPos += 0.75;
    }
  } else if (asp.includes("external")) {
    if (txt.includes("friction") || txt.includes("tariffs") || txt.includes("hurting")) {
      biasNeg += 0.7;
    } else if (txt.includes("surplus") || txt.includes("exports") || txt.includes("exports jumped")) {
      biasPos += 0.7;
    }
  }

  // Fallback to general terms
  if (txt.includes("strong") || txt.includes("robust") || txt.includes("growth") || txt.includes("improving")) {
    biasPos += 0.3;
  }
  if (txt.includes("weak") || txt.includes("deficit") || txt.includes("stagnating")) {
    biasNeg += 0.3;
  }

  const rawPos = biasPos + Math.random() * 0.05;
  const rawNeg = biasNeg + Math.random() * 0.05;
  const rawNeu = 0.35 + Math.random() * 0.05;

  const total = rawPos + rawNeg + rawNeu;
  return {
    positive: rawPos / total,
    negative: rawNeg / total,
    neutral: rawNeu / total
  };
}
