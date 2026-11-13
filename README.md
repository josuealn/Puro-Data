# Low Water Crossing Risk – San Antonio

Built for the Dell Technologies Social Impact AI Datathon at UT San Antonio, November 13, 2026.

**Presentation:** [docs/presentation.pdf](docs/presentation.pdf)

San Antonio has 156 low water crossings, where roads dip through creeks that flood during heavy rain. The warning signs around them (failing drains, uninspected inlets, sewer overflows on the creek) are spread across separate city and state datasets. This project joins them into one table, ranks every crossing by those warning signs, and uses a local AI model on a Dell Pro Max GB10 to write a plain-language warning in English and Spanish for each one.

The question it answers: **before the next storm, which crossings should the city check first, and why?**

## Results

- **The highest-risk crossings are the least watched:** 9 of the 10 highest-risk crossings have no public USGS stream gauge within a mile. Only 8 of all 156 crossings do, and 2 of those gauges have no current reading.
- **Different crossings rank high for different reasons:** #1 is far outside the city's drain network, #2 and #3 sit on Leon Creek with 19 sewer overflows since 2021, and #5 has 4 failed or poor inlets.
- **K-means found four kinds of crossing,** including the Leon Creek pair with heavy overflows and a group whose channel restoration is already planned.
- **Checked against real rescues:** the San Antonio Fire Department's #1 rescue site, Old Seguin at Salado Creek (46 rescues, 2015–2025), is #9 in our ranking. Most other rescue sites rank low, because rescues depend on where drivers enter the water, not only on drainage.
- **We tested traffic as an exposure measure,** using TxDOT counts. Public counts only reached 4 of the 10 rescue sites, and the match didn't improve, so the drainage ranking stays the main one.

## How it works

1. **Clean:** load the city and state files, keep Bexar County, and use one coordinate system (Texas State Plane, in feet).
2. **Join:** one row per crossing, with counts of storm inlets within 1,500 ft, planned drainage projects within ½ mile, and sewer overflows on the same creek.
3. **Live data:** the nearest USGS stream gauge and its current reading, and the nearest TxDOT traffic count.
4. **Score:** each feature becomes a z-score (how unusual it is among the 156 crossings), and the weighted z-scores are added into a risk score. Higher means more warning signs. It is a ranking, not a flood forecast.
5. **Cluster:** k-means, an unsupervised method, groups crossings with similar warning signs into 4 types.
6. **Priority score:** the risk score plus a traffic z-score, as a separate ranking of exposure.
7. **AI warnings:** the app sends one crossing's facts to Qwen3-VL-8B-Instruct, served by vLLM on the GB10, which writes the warning. Only facts go to the model, and it is told not to invent anything.

All data processing runs on a laptop. Only the AI step uses the GB10, so city data never leaves the local network.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── vlm_config.py              # GB10 model server address
├── lwc_pipeline.ipynb         # the whole data pipeline, run top to bottom
├── app.py                     # Streamlit demo app
├── docs/
│   └── presentation.pdf
└── data/
    ├── raw/                   # downloaded files (not committed, see "Data")
    └── processed/
        ├── crossings.csv      # one row per crossing; the app reads this
        ├── risk_map.html      # interactive map
        ├── briefs.csv         # AI warnings saved by the app
        └── figures/           # charts for the slides
```

Earlier notebooks (`01_clean.ipynb`, `03_charts.ipynb`, `04_traffic.ipynb`) are replaced by `lwc_pipeline.ipynb`.

## Data

Download these into `data/raw/` with their original file names (CSV where offered). The notebook finds each file by the start of its name.

| Dataset | Source | Used for |
| --- | --- | --- |
| Low Water Crossings | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/4089adb69af14608acfb02703b1dd439_0/about) | The 156 crossings |
| Storm Inlets | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/CoSAGIS::storminlets/about) | Drains near each crossing and their condition |
| IMP Drainage | [Open Data SA](https://data.sanantonio.gov/dataset/imp-drainage) | Planned debris removal and restoration, 2026–2030 |
| Sanitary Sewer Overflows | [TCEQ](https://data.texas.gov/dataset/Texas-Commission-on-Environmental-Quality-Sanitary/8kc5-95uk/about_data) | Overflows into each crossing's creek since 2021 |
| Storm Channels | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/CoSAGIS::stormchannels/about) | Not used: the CSV has no locations |

Downloaded automatically when the notebook runs (skipped if offline):

| Dataset | Source | Used for |
| --- | --- | --- |
| Stream gauges | USGS Water Services API | Nearest gauge and its current water level |
| Traffic counts | TxDOT AADT layers (state roads and city streets) | Vehicles per day near each crossing |

Rescue counts for validation come from the San Antonio Fire Department's list of its 10 locations with the most high-water rescues, 2015–2025, released May 2026 and reported by KSAT.

## Setup

```sh
conda create --name gb10 python=3.11 pip -y
conda activate gb10
python -m pip install -r requirements.txt
```

## Run the pipeline

```sh
python -m jupyterlab
```

Open `lwc_pipeline.ipynb` and use **Kernel → Restart Kernel and Run All Cells**. It writes everything in `data/processed/`. Rerun it whenever the data or the score weights change.

## Run the app

1. On the GB10, check that the model server is running: `docker ps` should list `qwen-vl-server`.
2. In `vlm_config.py`, set `BASE_URL = "http://GB10_IP:8000/v1"` (or `http://localhost:8000/v1` on the GB10 itself).
3. Test the connection: `curl http://GB10_IP:8000/v1/models`
4. Start the app: `streamlit run app.py`, then open `http://localhost:8501`. No account or login is needed.

The app shows the risk map with live USGS gauges, a details panel for each crossing, a button that writes an AI brief (for residents or a Public Works crew, in English, Spanish or both) and a chat box for questions about the selected crossing. If the GB10 or USGS can't be reached, it shows a message and keeps working with saved data.

## Limitations

- **The risk score ranks warning signs; it is not a flood forecast.** There is no public record of every flood at every crossing to train a predictive model.
- **Rescue counts exist for only 10 sites,** which is enough to check the ranking but not to train on.
- **Sewer overflows have no coordinates,** so they only reach the 18 crossings whose name includes a creek.
- **Rural crossings outside the city's drain network** rank high partly because there are no drains on record near them.
- **Public traffic counts cover few of the crossings,** so traffic is kept as a separate priority score.
- **The distances used (1,500 ft, ½ mile) are judgment calls** and change the ranking somewhat.
- **AI warnings are generated text** and should be reviewed before anything is shared with the public.

## Next steps

- Full fire department rescue history, to train a model that predicts where people get into danger
- City street traffic counts, to measure exposure properly
- Live crossing sensor data from BexarFlood (San Antonio River Authority)
- Storm channel locations and exact coordinates for sewer overflow reports

## Team

- Josue: data cleaning, joins, risk score, clustering, map, charts
- Larry: AI warnings on the GB10, demo app, slides
