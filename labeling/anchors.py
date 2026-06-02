# anchors.py
from typing import Dict, List

SEMANTIC_THRESHOLD = 0.55


# Economic news anchor sentences
ECONOMIC_ANCHORS = [
    "Gold seen bearish, according to global investor sentiments as reported by central bank",
    "asia increased imports",
    "Political manifestos, election campaigns, and party platforms frequently address sovereign economic stability, household purchasing power, and domestic workforce protection policies.",
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
    "The monetary policy committee emphasized data-dependent decision making.",
    "Central bank officials signaled caution regarding future policy adjustments.",
    "Meeting minutes revealed concerns about inflation persistence.",
    "The governor highlighted risks to the inflation outlook during the press conference.",
    "The monetary policy committee signaled a potential reduction in reserve requirements to stimulate credit expansion."
]
# Inflation
INFLATION_ANCHORS = [
    "cost of living has gone up",
    "Vulnerable households require targeted financial buffers to mitigate the direct economic impact of rising consumer prices and escalating cost of living pressures.",
    "Global supply chain shocks and shipping network disruptions are driving up raw material commodity input prices.",
    "As skyrocketing diesel costs ripple through the supply chain, economists warn that unmitigated fuel hikes are stoking broader inflation, leaving both commercial enterprises and household consumers vulnerable to a sharp increase in the cost of living.",
    "prices continue to climb year-on-year",
    "Inflation slowed considerably compared with the previous quarter.",
    "Price growth moderated as supply conditions improved.",
    "Consumer prices remained largely stable throughout the reporting period.",
    "Deflationary pressures emerged amid weakening demand.",
    "The pace of price increases has begun to ease.",
    "Rising energy prices are placing upward pressure on overall consumer inflation.",
    "Electricity tariffs and fuel costs contributed significantly to the latest inflation reading.",
    "Oil price increases are feeding through into transportation and production costs.",
    "Energy-driven inflation remains a key concern for policymakers.",
    "inflationary pressures remain elevated",
    "Food prices rose sharply due to adverse weather conditions affecting agricultural output.",
    "Households are spending a larger share of their income on groceries as food inflation persists.",
    "Higher agricultural commodity prices are contributing to inflationary pressures.",
    "Retail food costs continue to increase across major urban centers.",
    "Producer price indices ticked upward, indicating that rising raw material costs are beginning to pass through to wholesale market pricing and end-consumer goods."
]
# Economic Growth : GDP CPI Recession
ECONOMIC_GROWTH_ANCHORS = [
    "recession fears are mounting",
    "Market analysts and research houses maintain a cautious outlook on national growth momentum and macro economic performance indicators following recent policy announcements",
    "Changes in regulatory pricing mechanisms structurally impact aggregate domestic demand, altering the operating costs of commercial businesses and household consumption patterns.",
    "economy grew faster than expected",
    "the economy contracted sharply",
    "National policymakers must implement long-term structural reforms and fiscal adjustments to sustain macroeconomic growth and ensure future economic stability.",
    "Expanded manufacturing plant capacity and baseline industrial output volumes support stable long-term quarterly GDP expansions."
    ]
# Labor Market : Unemployment rate, minimum wage
LABOR_ANCHORS = [
    "layoffs increased amid cost cuts",
    "labor market remains tight",
    "Populist parties and opposition leaders are gaining voters by promising to defend jobs, boost wages and improve purchasing power.",
    "Downturns in labor-intensive industries heavily impact vulnerable workforces, contract personnel, and migrant laborers reliant on manual employment cycles.",
    "Industrial expansions and local development projects stimulate robust job creation, generating significant direct employment and downstream workforce opportunities",
    "Sovereign economic interventions and national contingency plans aim to safeguard systemic employment stability and mitigate widespread job losses during market crises.",
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
    "consumer spending increased",
    "household expenditure remained strong",
    "private consumption contributed to GDP growth",
    "retail sales rose strongly",
    "consumer demand remained robust",
    "households reduced discretionary spending",
    "consumer confidence improved",
    "consumer confidence deteriorated",
    "household purchasing activity strengthened",
    "weaker consumption weighed on economic growth",
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
    "The domestic currency experienced cross-border valuation shifts, strengthening or weakening against major regional trading partner exchange rates in the global FX spot market.",
    "As part of a broader regional trade strategy, several emerging market economies significantly increased their inbound import volumes of foreign goods over the tracking period.",
    "Geopolitical threats and military posturing targeting strategic maritime chokepoints jeopardize regional infrastructure corridors and disrupt international commercial networks",
    "The country operates as a net energy exporter, shielding its trade balance from global fossil fuel and crude oil market shocks.",
    "Changes in regional energy production and hydrocarbon export volumes heavily influence macroeconomic stability and commodity revenues.",
    "trade war between them continues",
    'Cross-border shipments rebounded sharply, narrowing the merchandise trade deficit as bilateral commerce normalized'
]
 # Fiscal Policy: Government spending and budget
FISCAL_ANCHORS = [
    # --- Government Spending & Subsidies ---
    "The finance ministry announced structural adjustments to national public expenditure programs and targeted state subsidy frameworks.",
    "Government economic resilience packages, financial stimulus programs, and national emergency funding initiatives.",
    "The Prime Minister announced a national regulation plan to ensure the security  of strategic commodity and energy supplies.",
    "Government leadership emphasized that maintaining national economic growth requires difficult policy decisions regarding subsidy re-evaluation and fiscal restructuring.",
    # --- Taxation & Revenue Generation ---
    "Tax cuts introduced this year aim to boost disposable income and corporate reinvestment rates.",
    "Parliament debated raising corporate windfall taxes and expanding the scope of the consumption tax.",
    "The inland revenue board stepped up enforcement to curb tax evasion and broaden the national revenue base.",
    # --- Fiscal Transfers & Social Protection ---
    "The government expanded its cash transfer programme, increasing monthly payouts to low-income households under the national aid scheme.",
    "Budget allocations for social safety nets were raised to cushion vulnerable groups from rising cost of living pressures.",
    "The welfare ministry announced an upward revision to the minimum household assistance threshold for aid eligibility.",

    # --- Government Revenue & Fiscal Balance ---
    "The treasury reported a narrowing of the fiscal deficit as tax collection exceeded projections amid stronger corporate earnings.",
    "Dividend income from state-owned enterprises contributed significantly to government revenue, partially offsetting expenditure growth.",
    "The government recorded a primary surplus for the first time in five years, driven by improved tax compliance and spending discipline.",

    # --- Public Investment & Development Spending ---
    "The government earmarked funds for large-scale infrastructure development under the national development plan.",
    "State capital expenditure on public transportation and utilities was increased to stimulate economic activity.",
    "The administration fast-tracked approval for publicly funded industrial zones and special economic corridors.",

    # --- Regulatory & Policy Intervention ---
    "New regulations mandating price controls on essential goods were introduced to protect consumers from supply-driven inflation.",
    "The government issued directives capping profit margins for fuel distributors and essential commodity traders.",
    "Authorities tightened compliance requirements for government procurement to reduce leakage in public spending.",

    # --- Monetary-Fiscal Coordination ---
    "The central bank cautioned that expansionary fiscal policy could complicate efforts to bring inflation back to target.",
    "Policymakers discussed coordinating fiscal stimulus with monetary easing to support economic recovery without overheating.",

    # --- Crisis Fiscal Response ---
    "The government unveiled an economic rescue package worth billions to cushion businesses and workers from the impact of the crisis.",
    "Emergency supplementary budgets were tabled in parliament to fund crisis relief measures and stabilise the economy.",
    "The finance minister outlined contingency spending plans to address shortfalls in revenue caused by the external shock.",
    # --- Budgetary Frameworks & Deficits ---
    "The government outlined its revised budgetary framework, targeting a reduction in the fiscal deficit through streamlined public expenditures.",
    "The annual budget allocation prioritized healthcare, education, and rural development civil projects.",
    "The administration is launching a state-funded development initiative backed by a multi-million dollar public capital allocation.",
    # --- Sovereign Debt & Fiscal Discipline ---
    "Economists warned that rising statutory debt levels could prompt international rating agencies to reassess the sovereign outlook.",
    "The treasury tap-issued new government investment issues (MGS/GII) to fund the fiscal deficit.",

    "A high-ranking government minister faces official questioning and institutional scrutiny regarding state-backed investment deals, public venture funds, or corporate partnerships with multinational tech firms."
]
 # Energy & Commodities; Oil, Solar, Energy, Metal
ENERGY_COMMODITIES_ANCHORS = [
    "Regional agricultural commodity export volumes are projected to surge, boosting international maritime shipping traffic and cross-border commercial port activity",
    "As global coal mining hubs enter peak production season, maritime commodity imports are expected to climb sharply throughout the second quarter."
    "natural gas prices spiked during winter demand",
    "electricity and household utility bills",
    "crude oil prices rose following supply disruptions",
    "natural gas prices spiked during winter demand",
    "electricity and household utility bills increased significantly",
    "agricultural commodity prices increased due to poor harvest conditions",
    "copper prices increased on expectations of stronger industrial activity",
    "iron ore demand strengthened due to construction sector growth",
    "renewable energy investment accelerated across the region",
    "coal production increased to meet rising electricity demand",
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
    "capital expenditure plans were delayed",
    "business investment increased amid improving economic conditions",
    "foreign direct investment inflows exceeded expectations",
    "companies announced significant expansion projects",
    "business sentiment improved as market conditions stabilized",
    "firms increased spending on productive assets",
    "companies postponed investment decisions",
    "capital expenditure plans were scaled back",
    "uncertainty weighed on private sector investment decisions",
    "corporate confidence strengthened during the quarter",
    "private sector investment supported economic growth"
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