# ESS11 LLM Prediction Study 🇪🇺🤖

This project evaluates whether Large Language Models (LLMs) can act as synthetic respondents in survey research, using real-world data from the **European Social Survey Round 11 (ESS11)**. 
The experiment tests if an LLM can accurately predict individual survey responses based on a person's other answers — using a structured leave-one-out approach.

---

## 💡 Project Summary

- **Data**: ESS Round 11, Edition 3.0 (focus: Germany + cross-country subset)
- **Model**: GPT-5 Nano via OpenAI API (zero-shot; no fine-tuning)
- **Design**: Leave-one-out — 34 variables per respondent; one held out and predicted using the remaining 33

---

## 🧠 Key Components

### `/data/`
- `ess11_de_filtered.csv`: Preprocessed dataset (Germany)
- `ess11_full_filtered.csv`: 7-country sample (300 respondents each)
- `ess11_de_variables.csv`: Variable list with labels and response options
- `ess11_llm_predictions_wide.csv`: All model predictions (Germany)
- `ess11_full_llm_predictions_wide.csv`: All model predictions (Multi-country)
- `ess11_llm_eval_summary.csv`: Evaluation results per variable

---

## 🧪 Prediction Pipeline

- **Script**: `run_predictions_super_parallel.py`
- For each person:
  - Constructs a prompt containing all answers **except** the target variable
  - Sends structured prompts to OpenAI's GPT-5 Nano API
  - Collects predictions for all 34 variables
- **Prompt structure**:
  - System: role, constraints, allowed response values
  - User: full list of known responses in readable form

For multi-country prediction, use `run_predictions_super_parallel_all_countries.py`.

---

## 🌍 Countries in Multi-Country Phase

- Germany (DE)  
- Sweden (SE)  
- Poland (PL)  
- Spain (ES)  
- Italy (IT)  
- Hungary (HU)  
- Netherlands (NL)

Each country includes ~300 randomly sampled respondents.


