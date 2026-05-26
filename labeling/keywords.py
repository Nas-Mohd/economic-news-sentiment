
# Keywords for matching
KEYWORDS: dict[str, list[str]] = {
    "monetary_policy": [
        "fed", "central bank", "interest rate", "rate cut", "rate hike",
        "monetary", "bnm", "overnight policy rate", "opr"
    ],
    "inflation": [
        "inflation", "cpi", "ppi", "price pressure", "cost of living", "consumer price", 'price', 'prices', 'domestic prices'
    ],
    "economic_growth": [
        "gdp", "recession", "economic growth", "economic expansion",
         "maintain growth", "sustain growth", "gdp growth", "growth trajectory"
        "productivity gain", "output growth", "quarterly growth", "annual growth"
    ],
    "labor_market": [
        "unemployment", "layoff", "job market", "labor participation", "wage",
        "employment", "labor", "hiring", "job creation", "minimum wage", "immigration",
        "hire", "layoff", "downsize", "recruit", "fire", "terminate","gig worker", "social security",
        "skills development", "training", "workforce", 'worker', 'employee', 'headcount'
    ],
    "consumer_activity": [
        "household debt", "credit card usage", "disposable income", "luxury good",
        "sale", "retail sale", "spending", "consumer confidence", "consumer sentiment",
        "consumer spending", "household spending", "consumer demand"
    ],
   "business_activity": [
    "manufacturing", "service", "pmis", "industrial output", "ceo", "automation",
    "corporation", "sme", "enterprise", "firm", "conglomerate", "multinational",
    "private sector", "commercial sector", "industrial base", "ecosystem","revenue", "profit", "earning", "profitability",
    "yield", "output", "capacity utilization", "business", "turnover", "margin",
    "investment", "business investment", "business expansion", "capex",
    "capital expenditure", "asset acquisition", "facility", "infrastructure rollout",
    "plant", "equipment", "r&d", "research and development", "property right", "robust", "expansion",
    "slowdown", "growth", "disruption", "supply chain", "supply network",
    "collapse", "bankruptcy", "insolvency", "default", "liquidation",
    "restructuring", "distress", "closure", "red tape",  "contraction", "resilient", "slump", "stagnation",
    ],
    "financial_markets": [
        "stock", "bond", "bursa", "equity", "yield", "index",
        "bourse", "indices", "klci", "fbmklci", "derivative", "derivatives", "futures market", "etf", "exchange-traded fund", "reit", "sukuk",
        "liquidation", "sell-off", "rally", "bull market", "bear market", "market capitalization", "market cap", "trading volume", "turnover value", "equities market",
        "ipo", "initial public offering", "treasury bills", "t-bills", "capital markets", "debt securities", "share price", "dividend yield", "stock", "bond", "bursa", "equity", "yield", "index",
        "bullish", "bearish", "ubs", "target price", "investor", "investors", "the market", "market sentiment"
    ],
    "trade_external": [
        "export", "import", "trade balance", "tariff", "trade war", "trade gap", 'supply chain', 'disruptions', 'logistics network',
        "exchange rate", "free trade", "devaluation", "trade deficit", "trade surplus", "current account",
        'usd','euro','ringgit','baht','peso','dollar','currency','myr','yen','yuan','php', "source market",
        "fx", "forex", "appreciation", "depreciation", "currency peg", "foreign reserves", "international reserves", "spot market", "cross-rate", "undervalued currency",
        "freight", "cargo", "shipping lane", "port congestion", "customs duty", "bill of lading", "maritime trade", "transshipment", "vessel", "container terminal"
    ],
    "fiscal_policy": [
        "government spending", "tax revenue", "budget deficit", "stimulus",
        "public debt", "infrastructure", "austerity", "tax base", "social security", "payroll tax",
        "budget revision", "price control", "targeted subsidy", "targeted subsidies",
        "diesel price", "fuel cost", "govt to allocate", "government allocation",
        "public spending", "subsidy rationalisation", "subsidy rationalization", 'subsidy',
        'economy ministry', 'ministry of economy', 'capital allocation', 'state-backed',
        "allocation", "funded by", "million allocation", "disbursement", "package",'supply'
    ],
    "energy_commodities": [
        "oil", "corn", "wheat", "palm oil", "solar", "wind", "nuclear", "electricity",
        "natural gas", "coal", "renewables", "metals", "price shock", "carbon", "supply disruption",
        "net exporter", "value chain", "semiconductor", "net importer", "importer", "exporter"
    ],
    "banking_credit": [
        "lending", "debt", "liquidity stress", "default", "interest rate", "bank",
        "mortgage", "loan", "central bank", "credit", "debit"
    ],
    "central_banks": [
        "bnm", "bank negara", "federal reserve", "fed", "ecb", "european central bank",
        "bank of england", "boe", "bank of japan", "boj", "peoples bank of china", "pboc",
        "reserve bank of australia", "rba", "bank of canada", "boc", "swiss national bank",
        "snb", "reserve bank of india", "rbi", "bank of korea", "banco de mexico", "ubs",
        "bank of brazil", "bcb", "monetary authority of singapore", "mas", "bank of russia", "cbr"
    ],
    "rate_actions": [
        "hike", "cut", "lower", "increase", "decrease", "maintain", "hold", "pause", "tighten", "raise"
    ],
    "advanced_context": [
        r"weather\s+the\s+(?:current\s+)?\w+\s+shock",   # Catches structural resilience (Real_Economic_Activity)
        r"supply\s+shock",                               # Supply chain disruptions (Real_Economic_Activity)
        r"global\s+(?:oil|commodity|energy|market|price)", # International transmission (External_Sector)
        r"war\s+in\s+\w+",                                # Geopolitical macro impact (External_Sector)
        r"external\s+(?:shock|demand|factor|environment)", # Direct external sector mention (External_Sector)
        # Should look for industry indicators coupled with directional market movement
        r"(slowdown|contraction|slump|boom|growth)\s+in\s+.*(sector|industry|real estate|construction)"
    ],
    'fiscal_actions': [
        'urge', 'introduce', 'implement', 'announce', 'adjust', 'reform', 'evaluate','evaluation', 'regard',
        'regulate', 'measure', 'table', 'approve', 'slash', 'hike', 'allocate', 'ensure', "fund", "allocate", "disburse", "spend", "grant", 
        "subsidize", "inject", "provide", "approve", "channel", "redirect", "reallocate", "distribute"
    ],
    'fiscal_instruments': [
        'subsidy', 'tax', 'taxation', 'budget', 'spending', 'expenditure', 'subsidy', 'allocations',
        'tariff', 'levy', 'excise', 'stimulus', 'allocation', 'package', 'supply', 'plan', "subsidies"
    ],
    'labor_actions': [
        'record', 'report', 'register', 'show', 'increase', 'rise'
    ]
}