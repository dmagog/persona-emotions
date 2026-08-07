# persona-emotions: эмоциональные steering-векторы для открытых LLM

Конвейер, который для любой открытой instruct-модели извлекает векторы семи
эмоций (ISEAR: anger, disgust, fear, guilt, joy, sadness, shame), находит
рабочий слой и коэффициент наведения и измеряет специфичность эффекта двумя
независимыми измерителями. Метод — contrastive activation steering по мотивам
[Persona Vectors (arXiv 2507.21509)](https://arxiv.org/abs/2507.21509),
перенесённый с черт личности на эмоции.

Посчитано на 11 моделях шести семейств (Qwen 2.5/3, Llama 3.2, Gemma 2/3,
Falcon 3, IBM granite, OLMo 2) по единому протоколу; судейские вердикты
проверены панелью из трёх LLM-судей трёх вендоров. Готовые артефакты всех
прогонов лежат в `runs/`, сводная таблица — `runs/summary.csv`.

## Быстрый старт

```bash
pip install -r requirements-inference.txt
export HF_TOKEN=...                                  # для gated-моделей
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1  # судья

python -m emotion.run_model_chain --config configs/models/qwen3-1.7b.yaml
python -m emotion.run_eval_chain  --slug Qwen3-1.7B --judge
python -m emotion.collect_results
```

Новая модель добавляется одним yaml в `configs/models/` — без правок кода.
Требования: одна GPU, для моделей до 3B хватает 8 ГБ.

## Документация

- **[RUNBOOK.md](RUNBOOK.md)** — как запускать, что делает каждая стадия,
  ловушки, которые стоили нам прогонов, и почему готовое не пересчитывается
  (отпечатки протокола у артефактов).
- **`python -m emotion.protocol`** — карточка протокола: каждое решение рядом
  с тем, как делает базовая статья, отклонения помечены.
- **`python -m emotion.protocol --check runs`** — проверка, что строки готовых
  прогонов сопоставимы между собой.
- **`python -m emotion.selftest`** — самопроверка логики конвейера, без GPU,
  за секунды. Гоняется перед каждым коммитом и в начале каждой ночной очереди.

## Структура

```
emotion/            конвейер: цепочка прогона, оценка, судьи, протокол, штампы
configs/models/     по одному yaml на модель
runs/<slug>/        артефакты прогонов: матрицы, свипы, судейские вердикты, манифесты
eval_emotion/       пары pos/neg (генерируются самой моделью)
emotion_vectors/    извлечённые векторы
data_generation/    сценарии и инструкции семи эмоций (extract/eval сплиты)
demo/               сборка демо-страницы из прогона
docs/legacy/        документы первого семестра и upstream-репозитория
```

Каталоги `eval/`, `NPTI/`, `analyze/`, `scripts/`, `training.py`, `sft.py` —
наследие upstream-репозитория ([xcfcode/persona](https://github.com/xcfcode/persona),
черты личности); эмоциональная ветка использует из них только
`eval/run_emotion_inference_batch.py` и `eval/model_utils.py`.
