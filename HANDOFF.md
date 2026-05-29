# Уляне: что взять и сделать

Сделай `git pull` (ветка `emotion-encoder`). Ниже три блока, по приоритету.

## 1. Разметка (вместе, главное)
Решаем, кто ближе к людям — энкодер или LLM-судья.
- Файл: `results/human_label_sheet.csv` (300 строк). В каждой строке текст и одна
  эмоция. Оцени интенсивность этой эмоции **0–100** в колонке `human_0_100`.
- Бери вторую половину (item_0150 … item_0299), я размечу первую.
- НЕ открывай `results/human_label_key.csv` — там оценки моделей, чтобы не якорило.
- Когда заполним: `python -m emotion.label_correlation --sheet results/human_label_sheet.csv --key results/human_label_key.csv`

## 2. Третий судья (deepseek)
- Прогнать на тех же ответах: `python -m emotion.judge_specificity --csv results/steer_spec_gemma_saefeat_n56.csv --model <deepseek> --out-wide results/judge_spec_n56_deepseek_wide.csv`
- Согласие 3 судей: `python -m emotion.judge_agreement` (см. `--help`).
- Судейский промпт берём наш — он уже в `emotion/judge_steered.py` и `emotion/pairwise_judge.py`.

## 3. Твой трек — «традиционные методы»
- Берёшь на себя, чтобы не пересекаться. Расширение по моделям/слоям тоже на тебе.
- Стиринг/специфичность параметризованы: `emotion/steer_specificity.py` (`--model_name`, `--layer`).

Контекст и общая рамка — в `REPORT.md` и `IMPROVEMENT_PLAN.md` (Фаза 5).
