# Low Water Crossing Risk – San Antonio

Built for the Dell Technologies Social Impact AI Datathon at UT San Antonio, November 13, 2026.

San Antonio's low water crossings are where flash floods turn deadly, and the warning signs around them are scattered across separate city datasets. This project ranks every crossing by flood risk. A local AI model running on a Dell Pro Max GB10 then writes a plain-language warning in English and Spanish, plus a maintenance suggestion, for each crossing.

## How it works

1. **Clean:** load the city and state datasets, keep Bexar County, and put everything in the same coordinate system.
2. **Combine:** build one table with a row per crossing. Columns count the storm inlets, drainage maintenance jobs and sewer overflows nearby, plus the distance to the nearest storm channel.
3. **Score and cluster:** rank the crossings with a transparent risk score and group them with k-means into named risk types.
4. **Write briefs:** send each crossing's facts to Qwen3-VL-8B-Instruct, served by vLLM on the GB10, which writes the bilingual warning.
5. **Show it:** an interactive map colored by risk, where clicking a crossing shows its numbers and its brief.

All data processing runs on a laptop. Only step 4 uses the GB10, and the data never leaves the local network.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── vlm_config.py         # GB10 model server address
├── 01_clean.ipynb        # cleaning, combined table, risk score, clusters, map
├── 02_briefs.ipynb       # Qwen prompt and brief generation
├── app.py                # Streamlit demo app
└── data/                 # not committed, see "Data" below
    ├── raw/              # downloaded files
    └── processed/        # crossings.csv, briefs.csv, risk_map.html
```

## Data

The data files are not committed. Download each one into `data/raw/`, choosing CSV or GeoJSON where offered.

| Dataset | Source | Used for |
| --- | --- | --- |
| Low Water Crossings | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/4089adb69af14608acfb02703b1dd439_0/about) | One row per crossing |
| Storm Inlets | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/CoSAGIS::storminlets/about) | Drainage near each crossing |
| Storm Channels | [COSA GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/CoSAGIS::stormchannels/about) | Distance to nearest channel |
| IMP Drainage | [Open Data SA](https://data.sanantonio.gov/dataset/imp-drainage) | Recent maintenance |
| Sanitary Sewer Overflows | [TCEQ](https://data.texas.gov/dataset/Texas-Commission-on-Environmental-Quality-Sanitary/8kc5-95uk/about_data) | Overflow reports nearby |
| Street Light Sensor Data (optional) | [Open Data SA](https://data.sanantonio.gov/dataset/street-light-sensor-data) | Flood-level readings |

## Setup

```sh
conda create --name gb10 python=3.11 pip -y
conda activate gb10
python -m pip install -r requirements.txt
```

`requirements.txt`:

```
jupyterlab>=4,<5
openai>=1,<4
pandas>=2,<4
geopandas
scikit-learn>=1.3,<2
matplotlib>=3.7,<4
folium
streamlit
```

## Connect to the GB10 model server

On the GB10, check that the model server is running:

```sh
docker ps    # look for qwen-vl-server
```

In `vlm_config.py`, set the server address:

```python
BASE_URL = "http://GB10_IP_ADDRESS:8000/v1"   # or http://localhost:8000/v1 when running on the GB10
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
API_KEY = "NONE"
```

Test it from your laptop:

```sh
curl http://GB10_IP_ADDRESS:8000/v1/models
```

## Run it

1. Start Jupyter from the activated environment with `python -m jupyterlab`.
2. Run `01_clean.ipynb` top to bottom. It writes `data/processed/crossings.csv` and `risk_map.html`.
3. Run `02_briefs.ipynb`. It writes `data/processed/briefs.csv`.
4. Launch the demo with `streamlit run app.py`.

## Limitations

- The risk score ranks crossings by nearby warning signs. It is not a flood probability and has not been checked against recorded flood events.
- The distance used to count nearby features (500 m to start) is a judgment call that changes the rankings.
- The AI briefs are generated text. A person should review them before anything is shared with the public.

## Team

- Josue: data cleaning, combined table, risk score, clustering, map
- Larry: Qwen briefs, Streamlit app, slides
