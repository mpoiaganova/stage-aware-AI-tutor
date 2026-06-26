# Learner-Stage Aware AI Tutor Improves Learning Processes: Initial Evidence from a Field Experiment

This repository contains sample data, code and AI tutor configuration materials for the paper "Learner-Stage Aware AI Tutor Improves Learning Processes: Initial Evidence from a Field Experiment". Accepted for publication at AIED 2026. 

## Folder Descriptions

| Path | Contents |
|---|---|
| `system_prompts/` | Full texts of the meta-prompts used for the four AI tutor versions (stage-aware conditions + baseline). |
| `replication/analysis/` | R scripts to reproduce the statistical analyses reported in the paper. |
| `replication/figures/` | Python scripts to reproduce the figures reported in the paper. |
| `replication/mock_data/` | Mock datasets that mirror the structure of the real data (column names, types, formats) without containing actual participant data. Provided so the analysis/figure scripts can be run end-to-end for verification. |
| `replication/mock_data/grades_mock.csv` | Mock grade data, matching the structure of the real grades dataset. |
| `replication/mock_data/survey_mock.csv` | Mock endline survey data, matching the structure of the real survey dataset. |
| `replication/mock_data/chatbot_mock.csv` | Mock chatbot interaction data, including message-level annotations, matching the structure of the real chatbot logs. |

> **Note:** All files under `mock_data/` are synthetic and do not contain real participant data. They are included solely to demonstrate the expected data structure and to allow the analysis/figure scripts to run for verification purposes.

## AI Tutor design

![AI Tutor design](./system%20prompts/tutor_design.png)

The repository is to be updated shortly. 

