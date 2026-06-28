# Learner-Stage Aware AI Tutor Improves Learning Processes: Initial Evidence from a Field Experiment

This repository contains sample data, code and AI tutor configuration materials for the paper "Learner-Stage Aware AI Tutor Improves Learning Processes: Initial Evidence from a Field Experiment". Accepted for publication as a long paper at Artificial Intelligence in Education (AIED) 2026. 

## Folder Descriptions

| Path | Contents |
|---|---|
| :file_folder: `replication/analysis/` | R scripts to reproduce the statistical analyses reported in the paper. |
| :file_folder: `replication/figures/` | Python scripts to reproduce the figures reported in the paper. |
| :file_folder: `replication/mock_data/` | Mock datasets that mirror the structure of the real data. |
| `replication/mock_data/grades_mock.csv` | Mock grade data. |
| `replication/mock_data/survey_mock.csv` | Mock endline survey data. |
| `replication/mock_data/chatbot_mock.csv` | Mock chatbot interaction data, including message-level annotations. |
| :file_folder: `system prompts/` | Full texts of the meta-prompts used for the four AI tutor versions (stage-aware conditions + baseline). |

> ❗ **Note:** All files under :file_folder: `mock_data/` are synthetic and do not contain real participant data. They are included solely to demonstrate the expected data structure and to allow the analysis and figures scripts to run for verification purposes. Real participant data is accessible upon request — please contact the authors at maria.poiaganova@business.uzh.ch

## How to Replicate

### Requirements

- **R** version `4.4` or later, with `data.table`, `ggplot2`, `BayesFactor`, `effects`, `brms`, `rmarkdown`, `knitr`.

- **Python** version `3.9` or later, with `pandas`, `numpy`, `plotly`.

### Steps

1. **Clone the repository**
```bash
   git clone https://github.com/mpoiaganova/stage-aware-AI-tutor.git
   cd stage-aware-AI-tutor
```

2. **Run the analysis**

   Render the analysis report with:

```bash
   Rscript -e 'rmarkdown::render("replication/analysis/report.Rmd")
```

   Alternatively, open ```report.Rmd``` in ```RStudio``` and click ```Knit```.

   The rendered output will be saved as ```report.html``` in the same folder.

4. **Generate the figures**
```bash
   pip install -r requirements.txt
   python replication/figures/<script_name>.py
```

4. **Check outputs**

   Results from the mock data will **not** numerically match the published findings, but the code should run end-to-end without errors.




