PROJECT: Adaptive GHG Emission Inversion System
STATUS: Starting from scratch, clean build
LOCATION: D:\patent_carbon_emission

═══════════════════════════════════════════════════
WHAT THIS SYSTEM IS
═══════════════════════════════════════════════════

I am building a patent-grade atmospheric greenhouse 
gas emission estimation system for Hyderabad, India.

The core patent claim is:

  A closed-loop adaptive inversion system that 
  stabilizes GHG emission estimates under real-world 
  sensor network failures, proven by:

  S = Var(E_static) / Var(E_adaptive) > 1.2
  consistently across dropout scenarios.

This is NOT a machine learning system.
This is NOT a black box.
This is a mathematical feedback loop that is 
interpretable, reproducible, and patent-defensible.

═══════════════════════════════════════════════════
THE SCIENCE IN PLAIN TERMS
═══════════════════════════════════════════════════

PROBLEM:
  We have 8-12 air quality sensors across Hyderabad
  (from CPCB via data.gov.in API).
  
  We want to estimate CO2-equivalent emissions from 
  different zones — traffic corridors, industrial 
  areas, residential zones.
  
  The forward model is:
    y = H * x + noise
  where:
    y = what sensors measure (pollutant observations)
    H = transport matrix (how emissions reach sensors)
    x = what we want (emission rates per zone)
  
  Problem 1 — TOPOLOGY BIAS:
    70% of sensors are near roads.
    Industrial zones are under-monitored.
    Static inversion over-estimates traffic emissions.
  
  Problem 2 — DROPOUT INSTABILITY:
    Sensors go offline randomly.
    Static inversion estimate swings wildly.
    Var(E_static) is high.
  
  Problem 3 — CONVERSION BIAS:
    Pollutant-to-CO2 conversion factors (theta) 
    are wrong at the start.
    Residuals never go to zero with fixed theta.

SOLUTION (THE PATENT):
  A three-part closed feedback loop:

  PART 1 — Leverage Weights R_i:
    Compute diagonal of hat matrix:
    R_i = diag(A(A'A + gamma*I)^-1 * A')
    This scores each sensor by how much geometric 
    pull it has on the inversion result.
    Normalize to [0,1]. Clip outliers at 3-sigma.

  PART 2 — Theta Adaptation:
    Weighted loss: J(theta) = sum(R_i * r_i^2)
    where r = y - H*c(theta)*x_hat
    Conversion factors: c_j = c0_j * exp(theta_j)
    Update: theta = theta - eta * grad(J)
             with momentum alpha for stability

  PART 3 — Closed Loop:
    R_i --> inversion --> residuals --> 
    theta update --> new inversion --> R_i
    Repeat until norm(delta_theta) < 1e-5
    
  RESULT:
    Var(E_adaptive) < Var(E_static)
    S = Var(E_static)/Var(E_adaptive) > 1.2
    Patent claim proven.

═══════════════════════════════════════════════════
REAL DATA SOURCES (ALL FREE, ALL PUBLIC)
═══════════════════════════════════════════════════

SOURCE 1 — CPCB via data.gov.in (PRIMARY)
  What: Hourly NO2, CO, PM2.5 per station
  Fields: country, state, city, station, 
          latitude, longitude, last_update,
          pollutant_id, pollutant_min, 
          pollutant_max, pollutant_avg
  API: data.gov.in real-time AQI dataset
  City filter: Hyderabad
  Granularity: Hourly
  Note: Raw live data, may have errors and 
        dropouts — this IS our dropout scenario
  Role in system: y vector (observations)

SOURCE 2 — ERA5 via Copernicus CDS API (WIND)
  What: u-wind, v-wind, boundary layer height
  Grid: 0.25 degree, hourly
  Bounding box: 17.0N-17.8N, 78.2E-78.8E
  Access: cds.climate.copernicus.eu (free key)
  Library: cdsapi Python package
  Role in system: builds physically correct H matrix

SOURCE 3 — EDGAR v8.0 (PRIORS)
  What: Gridded GHG emissions by sector
  URL: edgar.jrc.ec.europa.eu/dataset_ghg80
  Format: gridded CSV
  Sectors: transport, industrial, residential
  Role in system: x0 prior emission estimate

SOURCE 4 — OpenStreetMap Overpass API (LAND USE)
  What: Highway, industrial, residential zones
  URL: overpass-turbo.eu
  Query: 500m radius around each CPCB station
  Role in system: station zone classification
                  (determines network bias type)

═══════════════════════════════════════════════════
EXACT FOLDER STRUCTURE TO CREATE
═══════════════════════════════════════════════════

patent_carbon_emission/
│
├── data/
│   ├── loaders.py          # CPCB, ERA5, EDGAR, OSM
│   ├── raw/                # downloaded files go here
│   └── processed/          # cleaned outputs go here
│
├── core/
│   ├── dispersion.py       # Gaussian plume H matrix
│   ├── inversion.py        # Tikhonov solver
│   ├── residuals.py        # r = y - H*x_hat
│   ├── weights.py          # leverage scores R_i
│   └── adaptive.py         # THE PATENT LOOP
│
├── simulation/
│   ├── synthetic.py        # biased network generator
│   └── dropout.py          # stress tests, S metric
│
├── pipeline/
│   ├── run_pipeline.py     # master end-to-end runner
│   ├── run_realdata.py     # real CPCB data runner
│   └── parameter_sweep.py  # find best S config
│
├── dashboard/
│   └── app.py              # Streamlit 4-tab UI
│
├── results/
│   ├── figures/            # plots saved here
│   ├── patent_evidence_table.csv
│   └── parameter_sweep.csv
│
├── utils/
│   └── helpers.py          # logger, timer, normalize
│
├── requirements.txt
└── README.md

═══════════════════════════════════════════════════
THE SYNTHETIC NETWORK DESIGN (CRITICAL)
═══════════════════════════════════════════════════

Station placement must be intentionally biased
to mirror real Hyderabad sensor distribution:

  70% — road corridor
         y between 0.4 and 0.6
         full x range 0.0 to 1.0
         represents NH65, ORR, city highways

  20% — residential cluster
         x between 0.0 and 0.3
         y between 0.0 and 0.3
         represents Jubilee Hills, Banjara Hills

  10% — industrial cluster
         x between 0.7 and 1.0
         y between 0.7 and 1.0
         represents Patancheru, Nacharam

Source grid: 10x10 regular grid = 100 zones

Emission hotspots: concentrated at industrial corner
Initial conversion factors DELIBERATELY WRONG:
  traffic:     40% too high  (theta_traffic = +0.4)
  industrial:  30% too low   (theta_industrial = -0.3)
  residential: 10% too low   (theta_residential = -0.1)

═══════════════════════════════════════════════════
THE TARGET — DO NOT PROCEED WITHOUT THIS
═══════════════════════════════════════════════════

This table must be filled with real numbers 
before building dashboard or README:

Scenario              | Var Static | Var Adaptive | S
----------------------|------------|--------------|------
Targeted top-3 dropout|            |              | >1.2
Random 30% dropout    |            |              | >1.2
Road corridor removal |            |              | >1.3
Mean across 50 trials |            |              | >1.2

S > 1.2 consistently = patent claim proven.
S < 1.2 = tune parameters, do not move forward.

═══════════════════════════════════════════════════
CANONICAL PARAMETERS (STARTING POINT)
═══════════════════════════════════════════════════

n_stations:    30 (synthetic), 8-12 (real CPCB)
n_sources:     100 (10x10 grid)
lambda:        0.01 (Tikhonov regularization)
gamma:         0.01 (leverage weight regularization)
eta:           0.1  (learning rate, tune if S low)
alpha:         0.3  (momentum, prevents oscillation)
max_iter:      25   (adaptive loop max)
noise_std:     0.05 (synthetic observation noise)
dropout_frac:  0.30 (30% stations removed per trial)
n_trials:      50   (Monte Carlo dropout trials)
convergence:   1e-5 (norm delta theta threshold)

═══════════════════════════════════════════════════
BUILD ORDER — FOLLOW THIS EXACTLY
═══════════════════════════════════════════════════

PHASE 1 — FOUNDATION
  Session 1: folder structure + utils/helpers.py
  Session 2: simulation/synthetic.py 
             (biased network, wrong theta start)
  Session 3: core/dispersion.py + core/inversion.py
  Session 4: core/weights.py + core/residuals.py

PHASE 2 — PATENT CORE
  Session 5: core/adaptive.py
             (THE most important file — review 
              every line before accepting)
  Session 6: simulation/dropout.py
             (targeted + random + cluster dropout)

PHASE 3 — PROVE THE CLAIM
  Session 7: pipeline/parameter_sweep.py
             (find config where S > 1.2)
  Session 8: pipeline/run_pipeline.py
             (full end-to-end, generate evidence table)
  
  *** STOP HERE UNTIL S > 1.2 IS CONFIRMED ***

PHASE 4 — REAL DATA
  Session 9:  data/loaders.py
              (CPCB API, ERA5, EDGAR, OSM stubs)
  Session 10: pipeline/run_realdata.py
              (same pipeline on real Hyderabad data)

PHASE 5 — PRESENTATION
  Session 11: dashboard/app.py (Streamlit 4 tabs)
  Session 12: README.md + patent evidence table

═══════════════════════════════════════════════════
RULES FOR EVERY SESSION
═══════════════════════════════════════════════════

1. One session = one file or one tightly 
   related pair of files. Never mix phases.

2. After every session ask Claude Code:
   "Explain what the core function does and 
   why it reduces variance under dropout."
   If it cannot explain — the code is wrong.

3. Commit after every working session:
   git add . && git commit -m "session-N: description"

4. Never accept code with hardcoded S values.
   Every S number must come from actual runs.

5. Dashboard and README are last.
   They depend on proven S numbers.
   Do not build them on unproven claims.

═══════════════════════════════════════════════════
FIRST TASK FOR CLAUDE CODE
═══════════════════════════════════════════════════

Read everything above carefully.

Then do exactly this and nothing else:

1. Confirm you understand the patent claim 
   and what S > 1.2 means scientifically.

2. Confirm you understand why static inversion 
   becomes unstable when sensors cluster near 
   roads and a high-leverage sensor drops out.

3. Confirm you understand what R_i weights do 
   differently from raw residuals in the 
   theta update step.

4. Create the full folder structure listed above 
   with empty placeholder files.

5. Write utils/helpers.py with:
   - get_logger() with timestamped format
   - normalize_minmax() 
   - normalize_zscore()
   - array_summary() printer
   - Timer context manager
   - save_array() and load_array() for .npy files

6. Show me the folder structure when done 
   and confirm it matches exactly.

Do not write any other files yet.
Wait for my confirmation before Session 2.