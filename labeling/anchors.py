# anchors.py
from typing import Dict, List

SEMANTIC_THRESHOLD = 0.35


# Economic news anchor sentences
ECONOMIC_ANCHORS = [
    "Gold seen bearish, according to global investor sentiments as reported by central bank",
    "asia increased imports",
    "A high-ranking government minister faces official questioning and institutional scrutiny regarding state-backed investment deals, public venture funds, or corporate partnerships with multinational tech firms.",
    "Government entities and ministries face regulatory oversight and institutional scrutiny regarding public capital allocations for strategic foreign direct investments, sovereign tech venture funding, and semiconductor supply chain partnerships.",
    "Analysts warned that prolonged inflation could weigh further on economic recovery prospects.",
    "Sovereign nations utilizing international emergency credit lines must settle outstanding central banking obligations and institutional stabilization loans to restore external financial credibility.",
    "Financial media outlets and international institutions provide regular analytical reporting on global macroeconomic trends, national development milestones, and long-term fiscal performance.",
    "The central bank changed interest rates and job market policy.",
    "A looming sovereign default, national debt crisis, or balance-of-payments collapse forces governments to seek emergency international bailouts and strict fiscal restructuring packages.",
    "The finance ministry announced additional subsidies to offset rising fuel and food prices.",
    "Trade activity improved in April, resulting in an increase in exports.",
    "Government spending adjustments, fiscal restraint, and national budget revisions.",
    "The stock market grew 5 points this quarter, with TES (Tesla) being the biggest winner",
    "Global oil prices surged past key psychological thresholds due to energy market disruptions.",
    "Supply chain bottlenecks and commodity shortages rattled global trade networks.",
    "The US dollar (USD) gained value against the Malaysian Ringgit (MYR)",
    "National employment statistics tracking own-account workers, workforce participation rates, and monthly changes in headcount populations.",
    "The government formalized its alignment with emerging market trade blocs, aiming to diversify international commerce networks and enhance regional investment cooperation.",
    "The Prime Minister defended the nation's economic track record, affirming that robust institutional policy frameworks would keep the investment ecosystem stable despite regional volatility.",
    "The international monetary fund updated its macroeconomic outlook for developing markets, citing mixed structural adjustments.",
    "The government has announced a multi-million ringgit financial allocation to fund new economic and national talent development initiatives."
]

# Aspect sentence anchors
# Monetary Policy
MONETARY_ANCHORS = [
    "central bank raised interest rates",
    "benchmark interest rate hike",
    "Sovereign nations utilizing international emergency credit lines must settle outstanding central banking obligations and institutional stabilization loans to restore external financial credibility.",
    "liquidity tightening measures",
    "The monetary policy committee signaled a potential reduction in reserve requirements to stimulate credit expansion."
]
# Inflation
INFLATION_ANCHORS = [
    "cost of living has gone up",
    "Global supply chain shocks and shipping network disruptions are driving up raw material commodity input prices.",
    "As skyrocketing diesel costs ripple through the supply chain, economists warn that unmitigated fuel hikes are stoking broader inflation, leaving both commercial enterprises and household consumers vulnerable to a sharp increase in the cost of living.",
    "prices continue to climb year-on-year",
    "inflationary pressures remain elevated",
    "Producer price indices ticked upward, indicating that rising raw material costs are beginning to pass through to wholesale market pricing and end-consumer goods."
]
# Economic Growth : GDP CPI Recession
ECONOMIC_GROWTH_ANCHORS = [
    "recession fears are mounting",
    "economy grew faster than expected",
    "the economy contracted sharply",
    "National policymakers must implement long-term structural reforms and fiscal adjustments to sustain macroeconomic growth and ensure future economic stability.",
    "Surging manufacturing output and strong industrial production cycles are expected to significantly boost quarterly economic growth."
    ]
# Labor Market : Unemployment rate, minimum wage
LABOR_ANCHORS = [
    "layoffs increased amid cost cuts",
    "labor market remains tight",
    "National employment statistics tracking own-account workers, workforce participation rates, and monthly changes in headcount populations.",
    "Workforce training programs, social security protections, human capital development, and gig worker welfare.",
    "Labor unions accused management of wage theft, unfair contract breaches, and retaliatory terminations during the worker dispute.",
    "The department of statistics reported that the seasonally adjusted unemployment rate held steady as payroll additions balanced out market entries."
]
# Consumer Activity: demands,
CONSUMER_ANCHORS = [
    "household demand remains resilient",
    "more people are buying goods",
    "retail sales rose strongly this month",
    'Private consumption expenditures showed robust quarter-on-quarter acceleration, outstripping conservative retail forecasts'
]
 # Business Activity: Investments
BUSINESS_ANCHORS = [
    "A sharp contraction or operational slowdown in major industrial sectors, including real estate, construction, and infrastructure, heavily disrupts business activity, project completions, and contract worker utilization.",
    "manufacturing activity expanded this month",
    "A systemic collapse or widespread bankruptcy of small and medium enterprises (SMEs) poses a severe threat to the national business environment and industrial production infrastructure.",
    "Domestic enterprises signaled increased capital allocation toward building new distribution centers and scaling local production facilities.",
    "ceo reports increased profits",
    "A high-ranking government minister faces official questioning and institutional scrutiny regarding state-backed investment deals, public venture funds, or corporate partnerships with multinational tech firms.",
    "Industrial manufacturing plants and corporate entities are modifying their core operating models, production schedules, and factory inventory practices."
]
# Financial Markets: Stocks, Bonds, Instruments
FINANCIAL_ANCHORS = [
    "stock markets rallied on strong earnings",
    "investors have bullish expectations",
    "Investors and financial market participants are closely monitoring forward-looking macroeconomic data to price in upcoming risk factors.",
    "google report record high profits, 3 points",
    "The S&P 500 shed 1.24% to end at 7,408.50, while the Nasdaq Composite slipped 1.54% to 26,225.14.",
    "At the close of trade, the stock index finished points higher.",
    "The benchmark index slipped down percentage points following intraday trading.",
    "Sovereign bond yields curve shifted as equity markets faced selloffs.",
    "The local currency opened higher against the US dollar ahead of market closing."
]
# Trade : Imports Exports
TRADE_ANCHORS = [
    "palm oil exports increased",
    "The Malaysian ringgit fluctuated in volatile trading, gaining ground against the Singapore dollar and edging up against the US dollar in the regional currency market.",
    "The domestic currency experienced cross-border valuation shifts, strengthening or weakening against major regional trading partner exchange rates in the global FX spot market.",
    "asia increased imports",
    "The country operates as a net energy exporter, shielding its trade balance from global fossil fuel and crude oil market shocks.",
    "Changes in regional energy production and hydrocarbon export volumes heavily influence macroeconomic stability and commodity revenues.",
    "trade war between them continues",
    'Cross-border shipments rebounded sharply, narrowing the merchandise trade deficit as bilateral commerce normalized'
]
 # Fiscal Policy: Government spending and budget
FISCAL_ANCHORS = [
    # --- Government Spending & Subsidies ---
    "The finance ministry announced a major restructuring of targeted fuel and electricity subsidies.",
    "Government economic resilience packages, financial stimulus programs, and national emergency funding initiatives.",
    "The Prime Minister announced a national regulation plan to ensure the security  of strategic commodity and energy supplies.",
    "Government leadership emphasized that maintaining national economic growth requires difficult policy decisions regarding subsidy re-evaluation and fiscal restructuring.",
    # --- Taxation & Revenue Generation ---
    "Tax cuts introduced this year aim to boost disposable income and corporate reinvestment rates.",
    "Parliament debated raising corporate windfall taxes and expanding the scope of the consumption tax.",
    "The inland revenue board stepped up enforcement to curb tax evasion and broaden the national revenue base.",

    # --- Budgetary Frameworks & Deficits ---
    "The government outlined its revised budgetary framework, targeting a reduction in the fiscal deficit through streamlined public expenditures.",
    "The annual budget allocation prioritized healthcare, education, and rural development civil projects.",

    # --- Sovereign Debt & Fiscal Discipline ---
    "Economists warned that rising statutory debt levels could prompt international rating agencies to reassess the sovereign outlook.",
    "The treasury tap-issued new government investment issues (MGS/GII) to fund the fiscal deficit.",

    "A high-ranking government minister faces official questioning and institutional scrutiny regarding state-backed investment deals, public venture funds, or corporate partnerships with multinational tech firms."
]
 # Energy & Commodities; Oil, Solar, Energy, Metal
ENERGY_COMMODITIES_ANCHORS = [
    "Brazilian soy exports are projected to surge next month, boosting global port activity.",
    "As global coal mining hubs enter peak production season, maritime commodity imports are expected to climb sharply throughout the second quarter."
    "natural gas prices spiked during winter demand",
    "electricity and household utility bills",
    "Industrial metals like copper and iron ore faced downward price pressure as global warehouse inventories reached multi-month highs"
]
# Banking and Credit: Loan rates, mortgage, credit
BANKING_CREDIT_ANCHORS = [
    "household credit expanded steadily",
    "banks increased provisions for bad loans",
    "Commercial banks tightened their credit underwriting standards, leading to a marginal slowdown in corporate loan approvals."
]
# Corporate & Investment Climate:
CORPORATE_CLIMATE_ANCHORS = [
    "Inbound foreign direct investment surged as international conglomerates finalized cross-border capital allocations for regional infrastructure projects.",
    "firms expanded operations into new markets",
    "capital expenditure plans were delayed"
]

# --- Build dictionary of string anchors (optional, keep for reference) ---
ANCHORS: Dict[str, List[str]] = {
    "monetary_policy": MONETARY_ANCHORS,
    "inflation": INFLATION_ANCHORS,
    "economic_growth": ECONOMIC_GROWTH_ANCHORS,
    "labor_market": LABOR_ANCHORS,
    "consumer_activity": CONSUMER_ANCHORS,
    "business_activity": BUSINESS_ANCHORS,
    "financial_markets": FINANCIAL_ANCHORS,
    "trade_external": TRADE_ANCHORS,
    "fiscal_policy": FISCAL_ANCHORS,
    "energy_commodities": ENERGY_COMMODITIES_ANCHORS,
    "banking_credit": BANKING_CREDIT_ANCHORS,
    "corporate_climate": CORPORATE_CLIMATE_ANCHORS,
    "economic_news": ECONOMIC_ANCHORS
}