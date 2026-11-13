# Low Water Crossing Risk - San Antonio

Built for the Dell Technologies Social Impact AI Datathon at UT San Antonio, November 13, 2026.

San Antonio's low water crossings are where flash floods turn deadly, and the warning signs around them are scattered across separate city datasets. This project ranks every crossing by flood risk. A local AI model running on a Dell Pro Max GB10 then writes a plain-language warning in English and Spanish, plus a maintenance suggestion, for each crossing.

## How it works

1. **Clean:** load the city and state datasets, keep Bexar County, and put everything in the same coordinate system.
2. **Combine:** build one table with a row per crossing. Columns count nearby storm inlets, planned drainage projects, sewer overflow reports and the nearest USGS gauge.
3. **Score and cluster:** rank the crossings with a transparent risk score and group them with k-means into named risk types.
4. **Write briefs:** send each crossing's facts to Qwen3-VL-8B-Instruct, served by vLLM on the GB10, which writes the bilingual warning.
5. **Show it:** an interactive map colored by risk, where clicking a crossing shows its numbers and its brief.

Data processing runs on a laptop. Batch brief generation and optional dashboard questions use the GB10, and crossing data stays on the local network.

## Repository layout

```
.
|-- README.md
|-- requirements.txt
|-- vlm_config.py         # shared GB10 connection settings
|-- 01_clean.ipynb        # cleaning, joins, scoring, clustering and gauge association
|-- generate_briefs.py    # resumable Qwen batch generation
|-- app.py                # Streamlit display and selected-crossing interaction
`-- data/
    |-- raw/              # downloaded source files
    `-- processed/        # crossings.csv, briefs.csv, interactive_briefs.csv, risk_map.html
```

## Data

The data files are not committed. Download each one into `data/raw/`, choosing CSV or GeoJSON where offered.

| Dataset | Source | Used for |
| --- | --- | --- |
| Low Water Crossings | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/4089adb69af14608acfb02703b1dd439_0/about) | One row per crossing |
| Storm Inlets | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/CoSAGIS::storminlets/about) | Drainage near each crossing |
| Storm Channels | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/CoSAGIS::stormchannels/about) | Downloaded but not scored because the CSV lacks coordinates |
| IMP Drainage | [Open Data SA](https://data.sanantonio.gov/dataset/imp-drainage) | Planned FY 2026-2030 projects |
| Sanitary Sewer Overflows | [TCEQ](https://data.texas.gov/dataset/Texas-Commission-on-Environmental-Quality-Sanitary/8kc5-95uk/about_data) | Overflow reports nearby |
| Street Light Sensor Data (optional) | [Open Data SA](https://data.sanantonio.gov/dataset/street-light-sensor-data) | Flood-level readings |

## Setup

```sh
conda create --name gb10 python=3.11 pip -y
conda activate gb10
python -m pip install -r requirements.txt
```

## Connect to the GB10 model server

On the GB10, check that the model server is running:

```sh
docker ps    # look for qwen-vl-server
```

The shared defaults are in `vlm_config.py`. They can be overridden without editing code:

```powershell
$env:VLLM_BASE_URL = "http://GB10_IP_ADDRESS:8000/v1"
$env:VLLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
$env:VLLM_API_KEY = "EMPTY"
```

Test it from your laptop:

```sh
curl http://GB10_IP_ADDRESS:8000/v1/models
```

## Run it

1. Start Jupyter from the activated environment with `python -m jupyterlab`.
2. Run `01_clean.ipynb` top to bottom. It writes `data/processed/crossings.csv` and `risk_map.html`.
3. Run `python generate_briefs.py`. It writes `data/processed/briefs.csv`. Later runs reuse unchanged model-generated rows; use `python generate_briefs.py --force` for a complete refresh.
4. Launch the demo with `streamlit run app.py`.

The dashboard reads the prepared standard briefs and calls the model only for a custom selected-crossing brief or chat question. Custom briefs are stored separately in `data/processed/interactive_briefs.csv`.

## Limitations

- The risk score ranks crossings by nearby warning signs. It is not a flood probability and has not been checked against recorded flood events.
- The distance used to count nearby features (500 m to start) is a judgment call that changes the rankings.
- The AI briefs are generated text. A person should review them before anything is shared with the public.

## Team

- Josue: data cleaning, combined table, risk score, clustering, map
- Larry: Qwen briefs, Streamlit app, slides
