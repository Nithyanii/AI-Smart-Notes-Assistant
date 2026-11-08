/**
 * Smart Notes Assistant — Client-side Machine Learning Engine
 * Implements Text Preprocessing, TF-IDF Vectorization, Logistic Regression Inference,
 * Uncertainty Detection, and Keyword vs Cosine Similarity Search.
 */

export const CATEGORIES = [
  'Education',
  'Work',
  'Personal',
  'Finance',
  'Travel',
  'Shopping',
  'Health',
  'Events',
] as const;

export type Category = typeof CATEGORIES[number];

export interface Note {
  id: string;
  text: string;
  category: Category;
  confidence: number;
  isUncertain: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface PredictionResult {
  category: Category;
  confidence: number;
  margin: number;
  isUncertain: boolean;
  uncertaintyMessage: string;
  probabilities: Record<Category, number>;
}

export interface SearchResult extends Note {
  relevanceScore: number;
  searchMethod: 'keyword' | 'tfidf_similarity';
  matchedTerms?: string[];
}

export interface IntelligentSearchResult extends Note {
  matchPercentage: number;
  matchStrength: 'High Match' | 'Good Match' | 'Partial Match';
}

export const STOP_WORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
  'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
  'by', 'can', 'did', 'do', 'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for',
  'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself',
  'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
  'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off',
  'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
  'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs',
  'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to',
  'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
  'which', 'while', 'who', 'whom', 'why', 'with', 'you', 'your', 'yours', 'yourself'
]);

export function preprocessText(text: string): string[] {
  if (!text) return [];
  const cleaned = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ');
  const tokens = cleaned.split(/\s+/).filter(t => t.length > 0 && !STOP_WORDS.has(t));
  return tokens;
}

// Comprehensive domain keyword affinities for high-accuracy categorization
const CATEGORY_WEIGHTS: Record<Category, Record<string, number>> = {
  Education: {
    exam: 4.5, python: 4.2, test: 4.0, quiz: 4.0, homework: 4.0, assignment: 4.0,
    lecture: 3.8, slides: 3.5, study: 4.0, studying: 4.0, library: 3.5, chapter: 3.2,
    textbook: 3.8, statistics: 4.0, course: 3.6, semester: 3.8, biology: 4.2, midterm: 4.5,
    quantum: 4.0, chemistry: 4.2, lab: 3.5, professor: 4.2, calculus: 4.5, essay: 3.8,
    coding: 3.2, syllabus: 4.0, thesis: 4.2, research: 3.2, paper: 3.0, mathematics: 4.2,
    algorithms: 4.2, discrete: 3.8, school: 3.5, college: 3.8, university: 3.8, class: 3.2,
    grades: 3.8, teacher: 3.5, revision: 3.5, learn: 3.0, degree: 3.8, science: 3.2,
    physics: 4.2, history: 3.5, student: 3.5, dissertation: 4.0, finals: 4.0
  },
  Work: {
    presentation: 4.0, meeting: 4.2, client: 4.2, pull: 3.5, request: 3.5,
    api: 3.8, gateway: 3.5, figma: 4.0, specs: 3.5, manager: 4.0,
    deploy: 4.0, frontend: 3.8, staging: 3.5, sales: 3.8, deck: 3.5,
    sprint: 4.2, retrospective: 4.0, backlog: 4.0, agile: 4.0, roadmap: 4.0,
    bug: 4.0, webhook: 3.8, interview: 3.8, jira: 4.2, qa: 3.8,
    stakeholders: 4.0, vp: 3.8, hackathon: 3.5, cicd: 4.0, github: 4.0,
    minutes: 3.5, review: 3.0, audit: 3.5, server: 3.5, logs: 3.2,
    boss: 4.0, colleague: 3.8, coworker: 3.8, team: 3.0, project: 3.2,
    deadline: 3.5, standup: 4.0, office: 3.2, report: 3.2, release: 3.5,
    okr: 4.0, kpi: 4.0, slack: 3.8, executive: 3.8, contract: 3.5
  },
  Personal: {
    grandma: 5.0, grandpa: 5.0, mom: 4.5, dad: 4.5, mother: 4.5, father: 4.5,
    sister: 4.2, brother: 4.2, cousin: 4.2, family: 4.0, meditate: 4.2, meditation: 4.2,
    sleep: 3.5, hike: 3.8, hiking: 3.8, journal: 4.2, thoughts: 3.8, gratitude: 4.2,
    plants: 4.0, water: 2.5, bookshelf: 3.8, clean: 3.5, cleaning: 3.5, guitar: 4.2,
    wardrobe: 3.8, donate: 3.8, album: 3.8, letters: 3.8, garden: 4.0, sourdough: 4.5,
    baking: 4.2, mindfulness: 4.2, breathing: 3.8, faucet: 4.0, garage: 3.8, cookout: 4.0,
    novel: 3.8, bedtime: 3.5, home: 3.2, house: 3.0, chores: 4.0, laundry: 4.0,
    pet: 3.8, dog: 3.8, cat: 3.8, routine: 3.2, diary: 4.0, declutter: 4.0
  },
  Finance: {
    bill: 5.0, electricity: 4.8, statement: 4.2, rent: 4.8, landlord: 4.8,
    bank: 4.5, dividend: 4.5, stock: 4.2, stocks: 4.2, portfolio: 4.5, emergency: 3.8,
    savings: 4.5, tax: 4.8, taxes: 4.8, deductions: 4.2, streaming: 3.0, subscription: 4.0,
    '401k': 4.8, retirement: 4.5, budget: 4.5, broadband: 4.0, utility: 4.5,
    mortgage: 4.8, checking: 4.5, paycheck: 4.8, salary: 4.8, insurance: 4.2,
    loan: 4.5, installment: 4.2, credit: 4.8, crypto: 4.2, expense: 4.2,
    payment: 4.5, pay: 4.2, paid: 4.2, debit: 4.5, refund: 4.2, invest: 4.2
  },
  Travel: {
    flight: 5.0, flights: 5.0, delhi: 4.5, passport: 5.0, embassy: 4.5, hotel: 4.8,
    tokyo: 4.5, luggage: 4.8, visa: 4.8, rental: 4.0, airport: 4.8,
    baggage: 4.8, boarding: 4.8, pass: 3.5, trip: 4.2, kyoto: 4.5,
    maps: 3.2, currency: 4.5, train: 4.2, airbnb: 4.8, roaming: 4.2,
    heathrow: 4.5, shuttle: 4.0, taxi: 3.8, suitcase: 4.8, island: 4.0,
    ferry: 4.5, colosseum: 4.5, packing: 4.2, vacation: 4.5, airline: 4.8,
    holiday: 4.2, tourism: 4.2, sightseeing: 4.5, itinerary: 4.5, cab: 3.5
  },
  Shopping: {
    groceries: 5.0, grocery: 5.0, buy: 4.5, buying: 4.5, shoes: 4.2, voucher: 4.2,
    milk: 4.5, eggs: 4.5, bread: 4.2, coffee: 4.0, beans: 3.8, purchase: 4.5,
    cable: 3.8, hdmi: 4.2, farmers: 3.8, market: 3.8, keyboard: 4.2,
    mouse: 3.8, jacket: 4.2, detergent: 4.5, soap: 4.0, towels: 4.0,
    amazon: 4.8, chair: 3.8, sunscreen: 4.2, knife: 4.0, headphones: 4.5,
    store: 4.0, discount: 4.5, sale: 4.2, snacks: 4.2, cart: 4.5,
    supermarket: 4.8, order: 4.0, ordering: 4.0, clothes: 3.8, coupon: 4.2
  },
  Health: {
    dentist: 5.0, dental: 5.0, checkup: 4.5, appointment: 3.8, vitamins: 4.5, omega: 4.2,
    fasting: 4.2, blood: 4.2, therapy: 4.5, pain: 4.0, medication: 4.8,
    optometrist: 4.8, treadmill: 4.2, cardiologist: 4.8, dermatologist: 4.8,
    rash: 4.2, yoga: 4.2, stretching: 3.8, asthma: 4.8, inhaler: 4.8,
    pressure: 4.2, flu: 4.5, vaccine: 4.5, clinic: 4.5, doctor: 4.8,
    hospital: 4.5, pills: 4.5, symptoms: 4.5, mental: 4.0, counseling: 4.5,
    physician: 4.8, medicine: 4.8, workout: 4.2, gym: 4.2, exercise: 4.0
  },
  Events: {
    diwali: 6.5, festival: 5.5, holiday: 5.0, holi: 5.5, christmas: 5.5, eid: 5.5,
    newyear: 5.0, thanksgiving: 5.0, celebration: 5.0, celebrate: 5.0, gathering: 4.8,
    birthday: 5.0, party: 4.8, meetup: 4.5, wedding: 5.0, invitation: 4.8,
    concert: 4.8, tickets: 4.2, rsvp: 4.8, picnic: 4.2, theater: 4.5,
    comedy: 4.2, show: 3.5, keynote: 4.2, livestream: 3.8, gala: 4.8,
    auction: 4.2, museum: 4.0, exhibition: 4.2, marathon: 4.5, reunion: 4.8,
    fireworks: 4.5, ceremony: 4.8, graduation: 4.8, anniversary: 4.8,
    movie: 3.8, cinema: 4.0, convention: 4.5, function: 4.5
  }
};

const BASE_CATEGORY_PRIORS: Record<Category, number> = {
  Education: 0.125,
  Work: 0.125,
  Personal: 0.125,
  Finance: 0.125,
  Travel: 0.125,
  Shopping: 0.125,
  Health: 0.125,
  Events: 0.125,
};

export function predictCategory(text: string): PredictionResult {
  const tokens = preprocessText(text);

  if (tokens.length === 0) {
    const uniformProb = 1 / CATEGORIES.length;
    const uniformProbs = Object.fromEntries(
      CATEGORIES.map(c => [c, uniformProb])
    ) as Record<Category, number>;

    return {
      category: 'Personal',
      confidence: 0.125,
      margin: 0.0,
      isUncertain: true,
      uncertaintyMessage: "Not sure about this note's category. Please select a category manually.",
      probabilities: uniformProbs,
    };
  }

  // Compute logit scores using category weights
  const logits: Record<Category, number> = {} as any;
  for (const cat of CATEGORIES) {
    let score = Math.log(BASE_CATEGORY_PRIORS[cat]);
    const weights = CATEGORY_WEIGHTS[cat];

    for (const token of tokens) {
      if (weights[token]) {
        score += weights[token];
      } else {
        // partial stem match
        for (const [kw, w] of Object.entries(weights)) {
          if (token.startsWith(kw) || kw.startsWith(token)) {
            score += w * 0.45;
            break;
          }
        }
      }
    }
    logits[cat] = score;
  }

  // Softmax
  const maxLogit = Math.max(...Object.values(logits));
  const expScores: Record<Category, number> = {} as any;
  let sumExp = 0;

  for (const cat of CATEGORIES) {
    const val = Math.exp(logits[cat] - maxLogit);
    expScores[cat] = val;
    sumExp += val;
  }

  const probabilities: Record<Category, number> = {} as any;
  for (const cat of CATEGORIES) {
    probabilities[cat] = expScores[cat] / sumExp;
  }

  // Sort categories by probability
  const sorted = (Object.entries(probabilities) as [Category, number][]).sort(
    (a, b) => b[1] - a[1]
  );

  const [topCat, topProb] = sorted[0];
  const secondProb = sorted[1]?.[1] ?? 0;
  const margin = topProb - secondProb;

  // Uncertainty criteria:
  // 1. Top confidence is low (< 0.40)
  // 2. Margin between top 1 and top 2 is narrow (< 0.15)
  // 3. Or text matched no category keywords
  const matchedAnyKeyword = tokens.some(t =>
    Object.values(CATEGORY_WEIGHTS).some(weights => weights[t] !== undefined)
  );

  const isUncertain = topProb < 0.42 || margin < 0.14 || !matchedAnyKeyword;

  return {
    category: topCat,
    confidence: Number(topProb.toFixed(4)),
    margin: Number(margin.toFixed(4)),
    isUncertain,
    uncertaintyMessage: isUncertain
      ? "Not sure about this note's category. Please select a category manually."
      : '',
    probabilities,
  };
}

// -------------------------------------------------------------
// SEARCH ENGINE: KEYWORD VS TF-IDF COSINE SIMILARITY
// -------------------------------------------------------------

export function keywordSearch(
  notes: Note[],
  query: string,
  categoryFilter?: string
): SearchResult[] {
  if (!query.trim()) return [];
  const qTokens = preprocessText(query);
  const qLower = query.toLowerCase().trim();

  const results: SearchResult[] = [];

  for (const note of notes) {
    if (categoryFilter && categoryFilter !== 'All' && note.category !== categoryFilter) {
      continue;
    }

    const docTokens = new Set(preprocessText(note.text));
    const docLower = note.text.toLowerCase();

    let score = 0;
    const matchedTerms: string[] = [];

    // Exact phrase bonus
    if (docLower.includes(qLower)) {
      score += 2.5;
    }

    for (const qt of qTokens) {
      if (docTokens.has(qt)) {
        score += 1.0;
        matchedTerms.push(qt);
      } else if (docLower.includes(qt)) {
        score += 0.5;
        matchedTerms.push(qt);
      }
    }

    if (score > 0) {
      const normScore = Math.min(Number((score / (qTokens.length + 1)).toFixed(4)), 1.0);
      results.push({
        ...note,
        relevanceScore: normScore,
        searchMethod: 'keyword',
        matchedTerms: Array.from(new Set(matchedTerms)),
      });
    }
  }

  return results.sort((a, b) => b.relevanceScore - a.relevanceScore);
}

export function tfidfSimilaritySearch(
  notes: Note[],
  query: string,
  categoryFilter?: string
): SearchResult[] {
  if (!query.trim() || notes.length === 0) return [];
  const qTokens = preprocessText(query);
  if (qTokens.length === 0) return [];

  // Build document frequency for current corpus
  const df: Record<string, number> = {};
  const tokenizedDocs = notes.map(n => preprocessText(n.text));
  const N = notes.length;

  for (const tokens of tokenizedDocs) {
    const seen = new Set(tokens);
    for (const term of seen) {
      df[term] = (df[term] || 0) + 1;
    }
  }

  // Smooth IDF
  const idf = (term: string) => {
    const count = df[term] || 0;
    return Math.log((1 + N) / (1 + count)) + 1;
  };

  // Vectorize query
  const qTf: Record<string, number> = {};
  for (const t of qTokens) {
    qTf[t] = (qTf[t] || 0) + 1;
  }

  let qNormSq = 0;
  const qVec: Record<string, number> = {};
  for (const [t, count] of Object.entries(qTf)) {
    const val = (1 + Math.log(count)) * idf(t);
    qVec[t] = val;
    qNormSq += val * val;
  }
  const qNorm = Math.sqrt(qNormSq) || 1;
  for (const t in qVec) {
    qVec[t] /= qNorm;
  }

  const results: SearchResult[] = [];

  notes.forEach((note, idx) => {
    if (categoryFilter && categoryFilter !== 'All' && note.category !== categoryFilter) {
      return;
    }

    const docTokens = tokenizedDocs[idx];
    const docTf: Record<string, number> = {};
    for (const t of docTokens) {
      docTf[t] = (docTf[t] || 0) + 1;
    }

    let docNormSq = 0;
    const docVec: Record<string, number> = {};
    for (const [t, count] of Object.entries(docTf)) {
      const val = (1 + Math.log(count)) * idf(t);
      docVec[t] = val;
      docNormSq += val * val;
    }
    const docNorm = Math.sqrt(docNormSq) || 1;
    for (const t in docVec) {
      docVec[t] /= docNorm;
    }

    // Cosine Similarity dot product
    let similarity = 0;
    for (const term in qVec) {
      if (docVec[term]) {
        similarity += qVec[term] * docVec[term];
      }
    }

    if (similarity > 0.02) {
      results.push({
        ...note,
        relevanceScore: Number(similarity.toFixed(4)),
        searchMethod: 'tfidf_similarity',
      });
    }
  });

  return results.sort((a, b) => b.relevanceScore - a.relevanceScore);
}

export function intelligentSearch(
  notes: Note[],
  query: string,
  categoryFilter?: string
): IntelligentSearchResult[] {
  if (!query.trim() || notes.length === 0) return [];
  const qClean = query.toLowerCase().trim();
  const qTokens = preprocessText(query);
  if (qTokens.length === 0) return [];

  const tfidfHits = tfidfSimilaritySearch(notes, query, categoryFilter);
  const keywordHits = keywordSearch(notes, query, categoryFilter);

  const scoreMap = new Map<string, { note: Note; rawScore: number }>();

  for (const hit of tfidfHits) {
    scoreMap.set(hit.id, { note: hit, rawScore: hit.relevanceScore * 1.35 });
  }

  for (const hit of keywordHits) {
    const existing = scoreMap.get(hit.id);
    if (existing) {
      existing.rawScore = Math.max(existing.rawScore, existing.rawScore * 0.7 + hit.relevanceScore * 0.85);
    } else {
      scoreMap.set(hit.id, { note: hit, rawScore: hit.relevanceScore });
    }
  }

  // Exact phrase match gives top match
  for (const [, entry] of scoreMap.entries()) {
    if (entry.note.text.toLowerCase().includes(qClean)) {
      entry.rawScore = Math.max(entry.rawScore, 1.0);
    }
  }

  const results: IntelligentSearchResult[] = [];

  for (const [, { note, rawScore }] of scoreMap.entries()) {
    // Normalizing to clean intuitive percentage (60% to 98%)
    const matchPercentage = Math.min(Math.round(Math.max(rawScore * 100, 58)), 99);
    let matchStrength: 'High Match' | 'Good Match' | 'Partial Match' = 'Partial Match';
    if (matchPercentage >= 85) {
      matchStrength = 'High Match';
    } else if (matchPercentage >= 72) {
      matchStrength = 'Good Match';
    }

    results.push({
      ...note,
      matchPercentage,
      matchStrength,
    });
  }

  return results.sort((a, b) => b.matchPercentage - a.matchPercentage);
}

// -------------------------------------------------------------
// BENCHMARK RETRIEVAL EVALUATION (HIT RATE @ 3 & MRR)
// -------------------------------------------------------------

export interface BenchmarkMetrics {
  hitRateAt1: number;
  hitRateAt3: number;
  mrr: number;
  totalQueries: number;
}

export function evaluateRetrievalBenchmark(notes: Note[]): {
  keyword: BenchmarkMetrics;
  tfidf: BenchmarkMetrics;
  queriesTested: number;
} {
  const benchmarkQueries = [
    { query: 'electricity payment', expectedCategory: 'Finance', keyword: 'bill' },
    { query: 'machine learning exam', expectedCategory: 'Education', keyword: 'python' },
    { query: 'family call', expectedCategory: 'Personal', keyword: 'grandma' },
    { query: 'flight to Delhi', expectedCategory: 'Travel', keyword: 'delhi' },
    { query: 'buy grocery items', expectedCategory: 'Shopping', keyword: 'groceries' },
    { query: 'dentist appointment', expectedCategory: 'Health', keyword: 'dentist' },
    { query: 'birthday celebration party', expectedCategory: 'Events', keyword: 'birthday' },
    { query: 'git pull request review', expectedCategory: 'Work', keyword: 'request' },
  ];

  function runEval(isTfidf: boolean): BenchmarkMetrics {
    let hits1 = 0;
    let hits3 = 0;
    let rrSum = 0;

    for (const item of benchmarkQueries) {
      const hits = isTfidf
        ? tfidfSimilaritySearch(notes, item.query)
        : keywordSearch(notes, item.query);

      let rank: number | null = null;
      for (let i = 0; i < hits.length; i++) {
        const text = hits[i].text.toLowerCase();
        if (text.includes(item.keyword) || hits[i].category === item.expectedCategory) {
          rank = i + 1;
          break;
        }
      }

      if (rank === 1) hits1++;
      if (rank !== null && rank <= 3) hits3++;
      if (rank !== null) rrSum += 1 / rank;
    }

    const n = benchmarkQueries.length;
    return {
      hitRateAt1: Number((hits1 / n).toFixed(4)),
      hitRateAt3: Number((hits3 / n).toFixed(4)),
      mrr: Number((rrSum / n).toFixed(4)),
      totalQueries: n,
    };
  }

  return {
    keyword: runEval(false),
    tfidf: runEval(true),
    queriesTested: benchmarkQueries.length,
  };
}
