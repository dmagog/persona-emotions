# Уляне: что взять и сделать

Сделай `git pull` (ветка `emotion-encoder`). Ниже три блока, по приоритету.

## Что нового за сегодня (29.05)

Закрыли на нашей стороне (чтобы ты не дублировала):
- **W2 coeff-matched raw vs SAE:** при равном эффекте диагонали SAE-вектор **не**
  специфичнее сырого (на высокой силе даже протекает больше). Прежняя видимость «SAE
  меньше протекает» — артефакт магнитуды (`results/coeffmatch_gemma.md`).
- **Механизм (§7б):** декод top-k фич почти так же запутан в residual, как сырой вектор
  (cos +0.61 vs +0.66), и держит ~половину общей оси. Расщепление есть на уровне индексов
  фич (Jaccard 0.25), но не переходит в специфичность стиринга (`results/mechanism_ablation_gemma.md`).
- **A+V (§5.1):** корреляция векторов эмоций = одна общая ось «эмоциональности» (~80%
  дисперсии) + валентностный остаток (остаточная PC1 ↔ валентность Spearman +0.70 на Qwen
  и Gemma); joy анти-коррелирует с негативами (`results/valence_arousal_qwen.md`).
- **Off-distribution (§9а):** на фактологических промптах стиринг почти не наводит эмоцию
  (coeff 8 → ~0, кроме joy); больший coeff перебивает инструкцию, но ломает связность —
  «эмоция + связность» одновременно недостижимы (`results/random_prompts_gemma.md`,
  `results/random_coeff_sweep_gemma.md`).
- Отчёт финализирован: `REPORT.md` + `REPORT_humanized.md` в паритете. Формулировку
  «энкодер vs судья» смягчили — спор решает разметка (блок 1 ниже).

Что осталось за тобой: разметка (блок 1), deepseek-судья (блок 2), слои/модели и
«традиционные методы» (блок 3).

## 1. Разметка (вместе, главное)
Решаем, кто ближе к людям — энкодер или LLM-судья.
- Файл: `results/human_label_sheet.csv` (300 строк). В каждой строке текст и одна
  эмоция. Оцени интенсивность этой эмоции **0–100** в колонке `human_0_100`.
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
