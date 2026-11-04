# PAUSA 후속 연구: 의도 기반 반사실 궤적 생성

PAUSA(패스 타이밍 평가)는 패스가 없었을 경우의 궤적을 등속 가정으로 생성합니다.
이 레포는 PAUSA 코드에 그 궤적을 학습 기반 모델로 대체하는 후속 연구를 더한 것입니다. 후속 연구의 설명과 코드는 [`intent_gen/`](intent_gen/)에 있습니다.

| 폴더 | 내용 | 출처 |
|---|---|---|
| [`intent_gen/`](intent_gen/) | 의도 기반 반사실 궤적 생성 (Ours) | 후속 연구 |
| [`pausa/`](pausa/) | OBSO, pitch control, 데이터 로더 | PAUSA 원본 |
| [`notebooks/`](notebooks/) | PAUSA 계산과 분석 노트북 | PAUSA 원본 |
| [`elastic/`](elastic/) | 이벤트와 트래킹 데이터 동기화 | ELASTIC 원본 |
| [`data/`](data/) | 정적 모델과 예시 결과 | PAUSA 원본 |
| [`docs/`](docs/) | 아키텍처 다이어그램 | PAUSA 원본 |

아래는 PAUSA 원본 README입니다. 원본 코드는 Apache License 2.0을 따르며(`LICENSE`, `NOTICE`), 폴더 정리에 맞춰 파일 위치와 import 경로를 고쳤습니다.

---

# Valuing La Pausa
This is the repository for valuing optimal pass timing for football players using OBSO (Off-Ball Scoring Opportunity).

---

## La Pausa Example
The optimal pass is executed slightly later than the actual timing, allowing off-ball spaces to fully develop. The animation shows the full play context, while the snapshots compare the actual and delayed pass moments.

<p align="center">
    <img  src="data/dfl/obso/DFL-MAT-J03WMX/animation/output/animation.gif" controls width="500">
</p>

<table align="center">
  <tr>
    <th>
      <p align="center"><font size="4">Actual Timing</b></font></p>
    </th>
    <th>
      <p align="center"><font size="4">Optimal Timing (Slower)</b></font></p>
    </th>
  </tr>
  <tr>
    <td align="center">
      <img src="data/dfl/obso/DFL-MAT-J03WMX/animation/actual_pass.png" width="420"/>
    </td>
    <td align="center">
      <img src="data/dfl/obso/DFL-MAT-J03WMX/animation/optimal_pass.png" width="420"/>
    </td>
  </tr>
</table>

---

## Installation and Setup
- This Paper was developed with Python 3.11.10.
```
git clone https://github.com/leemingo/mitssac-pausa.git
cd value-pausa
pip install -r requirements.txt
```

## Project Structure

```
🗂️ mitssac-pausa/
├── 🗂️ data
│   ├── 🗂️ dfl # DFL DataSet
        ├── 🗂️ raw
            ├── 🗂️ DFL-MAT-J03WMX # Match ID
            │   ├── 📄 DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml       # Match metadata
            │   ├── 📄 DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml             # Event data
            │   ├── 📄 DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml # Tracking data
            ├── 🗂️ Other Match ID ... 
        ├── 🗂️ elastic
            ├── 🗂️ DFL-MAT-J03WMX # Match ID
            │   ├── 📄 meta_data.parquet             
            │   ├── 📄 event.parquet
            │   ├── 📄 tracking.parquet
            │   ├── 📄 raw_tracking.parquet
            ├── 🗂️ Other Match ID ... 
        ├── 🗂️ obso     
            ├── 🗂️ DFL-MAT-J03WMX # Match ID
                ├── 🗂️ event
                    │   ├── 📄 obso.pkl   
                    │   ├── 📄 ... 
                ├── 🗂️ trace 
                    │   ├── 📄 obso.pkl   
                    │   ├── 📄 ... 
                ├── 🗂️ virtual  # compute PAUSA
                    │   ├── 📄 pass_events.pkl 
                    │   ├── 📄 virtual_obso_maps.pkl  # Transition, scoring, pitch control maps
                    │   ├── 📄 virtual_obso.pkl 
                    │   ├── 📄 ... 
            ├── 🗂️ Other Match ID ... 
│   └── 🗂️ static # Static data (e.g., Transition.csv, EPV_grid.csv, xT_grid.json)
├── 🗂️ elastic                # Synchronization of tracking and event data
│   ├── 🗂️ sync               # Synchronization Logic
│   ├── 🗂️ tools              # Unified Spadl format
│   ├── 📄 convert_elastic.py # Main synchronization script
│   └── ...
├── 🗂️ notebooks               
│   ├──  1-visualize-obso.ipynb # OBSO visualization
│   ├── 2-calculate-pausa.ipynb # PAUSA computation and visualization
│   ├──  3-analysis-pausa.ipynb # Player- and team-level PAUSA analysis
├── 🗂️ pausa
│   ├── 📄 calculate_obso.py # Main OBSO computation
│   ├── 📄 loader.py         # Customized dataloader for PAUSA
│   ├── 📄 obso.py           # OBSO computation logic
│   ├── 📄 pitch_control.py  # Pitch control model
│   ├── 📄 xthreat.py        # Expected Threat (xT) computation
│   └── ...
├── 🗂️ intent_gen            # Follow-up: intent-based counterfactual trajectories
└── ...
```

## Architecture

Open [`docs/architecture.html`](docs/architecture.html) in a browser to explore the C4 architecture diagrams (System Context, Container).

## Data Usage
- Unfortunately, we cannot share the full DFL dataset. However, we provide seven publicly available matches, which are sufficient to reproduce the main experiments in this repository.
- Public dataset: [DFL DataSet](https://springernature.figshare.com/articles/dataset/-An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177).
- Download the public DFL matches from the link above. Organize each match using its match ID as the directory name.

## Running the Code

### 1. Elastic Data Preprocessing
- [The Elastic Framework](https://github.com/hyunsungkim-ds/elastic) operates on data converted into a SPADL format.
```bash
python elastic/convert_elastic.py --data_dir ~/geonhee/Data/dfl/raw --save_dir ./data/dfl/elastic --n_jobs -1
```

### 2. Caculate OBSO

```bash
# Event-level OBSO
python pausa/calculate_obso.py --provider dfl --unit event --game_id DFL-MAT-J03WMX --data_dir ./data/dfl/elastic --output_dir ./data/dfl/obso --n_jobs -1

# Tracking-level OBSO
python pausa/calculate_obso.py --provider dfl --unit trace --game_id DFL-MAT-J03WMX --data_dir ./data/dfl/elastic --output_dir ./data/dfl/obso --n_jobs -1

# Virtual trajectory-based OBSO
# This mode generates counterfactual trajectories to estimate OBSO under all possible timing scenarios.
python pausa/calculate_obso.py --provider dfl --unit virtual --game_id DFL-MAT-J03WMX --data_dir ./data/dfl/elastic --output_dir ./data/dfl/obso --n_jobs -1
```

### 3. Run Analysis [Notebook 3](https://github.com/leemingo/mitssac-pausa/blob/main/notebooks/3-analysis-pausa.ipynb)
- Compute PAUSA, xT, POP, and related metrics.
- Perform quantitative analysis (e.g., correlations between metrics).
- Perform qualitative analysis, including player- and team-level rankings

## License

[Apache License 2.0](LICENSE) — see [NOTICE](NOTICE) for third-party attribution.

## Research
please consider citing the following papers:
- Minho Lee, Geonhee Jo, Miru Hong, Pascal Bauer, and Sang-Ki Ko. **Valuing La Pausa: Quantifying Optimal Pass Timing Beyond Speed.** In 2026 MIT Sloan Sports Analytics Conference. <br/>[ [paper](https://www.sloansportsconference.com/conference/2026-conference) | [github](https://github.com/leemingo/mitssac-pausa.git) ]

## Affiliations
This research is conducted by **[Korea AI Research Society for Sports (KAISports)](https://sites.google.com/view/kaisport)**.  
