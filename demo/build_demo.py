#!/usr/bin/env python3
"""Build the self-contained demo page from the extracted generation data."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "demo_data.json").read_text())

HTML = """<meta charset="utf-8">
<title>Наведение эмоций на текст — демонстрация</title>
<style>
:root{
  --paper:#fcfcfb; --raised:#ffffff; --ink:#15171c; --ink-soft:#4a4f5a;
  --ink-faint:#7a808d; --rule:#e3e3e0; --rule-strong:#c9cac6;
  --anger:#b0433a; --disgust:#69783d; --fear:#665596; --guilt:#a1663a;
  --joy:#b58a26; --sadness:#3d6690; --shame:#8d5169;
  --good:#3f7a52; --warn:#a8752c; --bad:#a8443c;
  --serif:Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#101216; --raised:#171a20; --ink:#eceef2; --ink-soft:#b0b6c2;
    --ink-faint:#7d848f; --rule:#262a32; --rule-strong:#393f49;
    --anger:#e0776c; --disgust:#a3b76c; --fear:#9c8ad0; --guilt:#d69a63;
    --joy:#e0b955; --sadness:#71a0cf; --shame:#c5849e;
    --good:#6fb587; --warn:#d6a44e; --bad:#e08078;
  }
}
:root[data-theme="dark"]{
  --paper:#101216; --raised:#171a20; --ink:#eceef2; --ink-soft:#b0b6c2;
  --ink-faint:#7d848f; --rule:#262a32; --rule-strong:#393f49;
  --anger:#e0776c; --disgust:#a3b76c; --fear:#9c8ad0; --guilt:#d69a63;
  --joy:#e0b955; --sadness:#71a0cf; --shame:#c5849e;
  --good:#6fb587; --warn:#d6a44e; --bad:#e08078;
}
:root[data-theme="light"]{
  --paper:#fcfcfb; --raised:#ffffff; --ink:#15171c; --ink-soft:#4a4f5a;
  --ink-faint:#7a808d; --rule:#e3e3e0; --rule-strong:#c9cac6;
  --anger:#b0433a; --disgust:#69783d; --fear:#665596; --guilt:#a1663a;
  --joy:#b58a26; --sadness:#3d6690; --shame:#8d5169;
  --good:#3f7a52; --warn:#a8752c; --bad:#a8443c;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--sans);line-height:1.55;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 96px}

/* ---- навигация: липкая полоса сверху, на широком экране рельс слева ---- */
#nav{position:sticky;top:0;z-index:20;display:flex;gap:18px;overflow-x:auto;
  background:var(--paper);border-bottom:1px solid var(--rule);
  margin:0 -24px;padding:11px 24px;scrollbar-width:none}
#nav::-webkit-scrollbar{display:none}
#nav a{color:var(--ink-faint);text-decoration:none;font-size:11px;
  letter-spacing:.07em;text-transform:uppercase;white-space:nowrap;
  padding:3px 0;transition:color .12s,border-color .12s}
#nav a:hover{color:var(--ink)}
#nav a.on{color:var(--ink);font-weight:700}
#nav a:focus-visible{outline:2px solid var(--ink);outline-offset:3px}
/* Рельс справа включаем только там, где он не залезает на текст:
   поле = (W - 1080) / 2; панель занимает 20 + 132 + 26 + 2 = 180px.
   При 1460px поле равно 190px, запас 10px. */
@media (min-width:1460px){
  #nav{position:fixed;right:20px;top:50%;transform:translateY(-50%);
    flex-direction:column;gap:1px;width:132px;
    background:var(--raised);border:1px solid var(--rule);border-radius:4px;
    margin:0;padding:13px;overflow:visible}
  #nav a{white-space:normal;border-left:2px solid var(--rule);
    padding:6px 0 6px 11px;line-height:1.3}
  #nav a.on{border-left-color:var(--ink)}
}
section{scroll-margin-top:60px}
@media (min-width:1460px){section{scroll-margin-top:20px}}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}

/* ---- masthead ---- */
header.top{padding:56px 0 28px;border-bottom:2px solid var(--ink)}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-faint);font-weight:600;margin:0 0 14px}
h1{font-size:clamp(28px,4.2vw,42px);line-height:1.12;margin:0 0 16px;
  font-weight:700;letter-spacing:-.02em;text-wrap:balance;max-width:20ch}
.standfirst{font-family:var(--serif);font-size:18px;line-height:1.6;
  color:var(--ink-soft);margin:0;max-width:62ch}
.setup{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}
.chip{font-family:var(--mono);font-size:11.5px;padding:4px 9px;
  border:1px solid var(--rule-strong);border-radius:3px;color:var(--ink-soft);
  background:var(--raised)}

/* ---- вердикт ---- */
.verdict{border:1px solid var(--rule);background:var(--raised)}
.v-row{display:grid;grid-template-columns:132px 1fr;gap:18px;padding:15px 20px;
  border-bottom:1px solid var(--rule)}
.v-row:last-child{border-bottom:0}
.v-row .lab{font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  font-weight:700;padding-top:2px}
.v-row .lab.yes{color:var(--good)} .v-row .lab.no{color:var(--bad)}
.v-row .lab.warn{color:var(--warn)} .v-row .lab.why{color:var(--ink-faint)}
.v-row p{font-family:var(--serif);font-size:15.5px;margin:0;line-height:1.62;
  color:var(--ink-soft);max-width:62ch}
.v-row p b{color:var(--ink);font-weight:600}
@media (max-width:620px){.v-row{grid-template-columns:1fr;gap:6px}}

/* ---- матрица специфичности ---- */
.mx{--m:22,106,99;border:1px solid var(--rule);background:var(--raised);
  overflow-x:auto}
.mx table{border-collapse:separate;border-spacing:2px;width:100%;
  font-variant-numeric:tabular-nums}
.mx th{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-faint);font-weight:600;padding:8px 6px;white-space:nowrap;
  text-align:center}
.mx th.rh{text-align:right;padding-right:11px;color:var(--ink);font-weight:700}
.mx td{font-family:var(--mono);font-size:12px;text-align:center;padding:9px 6px;
  color:var(--ink);border-radius:2px;position:relative}
.mx td.dg{outline:2px solid var(--ink);outline-offset:-2px;font-weight:700}
.mx .ci{font-family:var(--mono);font-size:11px;color:var(--ink-faint);
  white-space:nowrap;padding-left:10px;text-align:left}
.mx-legend{display:flex;flex-wrap:wrap;gap:16px;align-items:center;
  padding:11px 14px;border-top:1px solid var(--rule);font-size:12px;
  color:var(--ink-soft)}
.mx-legend i{display:inline-block;width:13px;height:13px;border-radius:2px;
  vertical-align:-2px;margin-right:6px}
@media (prefers-color-scheme:dark){.mx{--m:79,168,156}}
:root[data-theme="dark"] .mx{--m:79,168,156}
:root[data-theme="light"] .mx{--m:22,106,99}

/* ---- sections ---- */
section{padding-top:52px}
h2{font-size:13px;letter-spacing:.12em;text-transform:uppercase;font-weight:700;
  margin:0 0 6px;color:var(--ink)}
.lede{font-family:var(--serif);font-size:16.5px;color:var(--ink-soft);
  margin:0 0 24px;max-width:64ch;line-height:1.62}

/* ---- scenario picker ---- */
.picker{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:24px}
.pick{font-family:var(--sans);font-size:13px;padding:7px 13px;cursor:pointer;
  border:1px solid var(--rule-strong);background:var(--raised);color:var(--ink-soft);
  border-radius:3px;transition:border-color .12s,color .12s}
.pick:hover{border-color:var(--ink-faint);color:var(--ink)}
.pick[aria-pressed="true"]{background:var(--ink);color:var(--paper);
  border-color:var(--ink);font-weight:600}
.pick:focus-visible{outline:2px solid var(--ink);outline-offset:2px}

.scenario{border-left:3px solid var(--ink);padding:2px 0 2px 16px;margin-bottom:26px}
.scenario .lbl{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-faint);font-weight:600;display:block;margin-bottom:5px}
.scenario p{font-family:var(--serif);font-size:17px;margin:0;max-width:62ch;
  line-height:1.55}

/* ---- generation cards ---- */
.grid{display:grid;gap:1px;background:var(--rule);
  grid-template-columns:repeat(auto-fill,minmax(310px,1fr));
  border:1px solid var(--rule)}
.card{background:var(--raised);padding:18px 20px 20px;display:flex;
  flex-direction:column;gap:11px}
.card.base{grid-column:1/-1;background:var(--paper)}
.card-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
.emo{font-size:12px;letter-spacing:.1em;text-transform:uppercase;font-weight:700}
.emo.anger{color:var(--anger)} .emo.disgust{color:var(--disgust)}
.emo.fear{color:var(--fear)}   .emo.guilt{color:var(--guilt)}
.emo.joy{color:var(--joy)}     .emo.sadness{color:var(--sadness)}
.emo.shame{color:var(--shame)} .emo.baseline{color:var(--ink-faint)}
.delta{font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums;
  color:var(--ink-faint);white-space:nowrap}
.delta.up{color:var(--good);font-weight:600}
.gen{font-family:var(--serif);font-size:15.5px;line-height:1.66;margin:0;
  color:var(--ink)}
.card.base .gen{font-size:17px;max-width:64ch}
.bar{height:3px;background:var(--rule);position:relative;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;display:block}
.bar i.anger{background:var(--anger)} .bar i.disgust{background:var(--disgust)}
.bar i.fear{background:var(--fear)}   .bar i.guilt{background:var(--guilt)}
.bar i.joy{background:var(--joy)}     .bar i.sadness{background:var(--sadness)}
.bar i.shame{background:var(--shame)}

/* ---- ladder ---- */
.ladder{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.step{background:var(--raised);padding:18px 20px;display:flex;
  flex-direction:column;gap:10px}
.step-head{display:flex;align-items:baseline;justify-content:space-between}
.coeff{font-family:var(--mono);font-size:12.5px;font-weight:600;color:var(--ink)}
.score{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums;
  color:var(--ink-soft)}

/* ---- coherence ---- */
.coh{border:1px solid var(--rule);background:var(--raised)}
.coh-row{display:grid;grid-template-columns:96px 1fr 54px;gap:14px;
  align-items:center;padding:9px 18px;border-bottom:1px solid var(--rule)}
.coh-row:last-child{border-bottom:0}
.coh-row .n{font-family:var(--mono);font-size:12.5px;text-align:right;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.coh-row .name{font-size:12px;letter-spacing:.08em;text-transform:uppercase;
  font-weight:600}
.track{height:8px;background:var(--rule);position:relative}
.track i{position:absolute;inset:0 auto 0 0;display:block;background:var(--ink-faint)}
.track i.hi{background:var(--good)}

/* ---- tables ---- */
.scroll{overflow-x:auto;border:1px solid var(--rule);background:var(--raised)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:9px 14px;text-align:left;border-bottom:1px solid var(--rule)}
th{font-size:10.5px;letter-spacing:.11em;text-transform:uppercase;
  color:var(--ink-faint);font-weight:600;white-space:nowrap}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
tr:last-child td{border-bottom:0}
.flag{font-weight:600}
.flag.bad{color:var(--bad)} .flag.warn{color:var(--warn)} .flag.good{color:var(--good)}

/* ---- caveat blocks ---- */
.caveat{border:1px solid var(--rule-strong);background:var(--raised);
  padding:20px 22px;margin-bottom:1px}
.caveat h3{font-size:14px;margin:0 0 8px;font-weight:700;letter-spacing:-.01em}
.caveat p{font-family:var(--serif);font-size:15px;color:var(--ink-soft);
  margin:0 0 10px;max-width:64ch;line-height:1.62}
.caveat p:last-child{margin-bottom:0}
.judge{border:1px solid var(--rule);background:var(--raised);padding:18px 22px;
  margin-bottom:1px}
.judge h3{font-size:14px;margin:0 0 4px;font-weight:700;letter-spacing:-.01em}
.judge p{font-family:var(--serif);font-size:15px;color:var(--ink-soft);
  margin:0 0 10px;max-width:64ch;line-height:1.62}
.judge p:last-child{margin-bottom:0}
.steps{counter-reset:s;display:flex;flex-direction:column;gap:1px;
  background:var(--rule);border:1px solid var(--rule)}
.step-row{background:var(--raised);padding:16px 22px 18px 58px;position:relative}
.step-row::before{counter-increment:s;content:counter(s);position:absolute;
  left:20px;top:16px;font-family:var(--mono);font-size:11.5px;font-weight:700;
  color:var(--ink-faint)}
.step-row h4{font-size:13.5px;margin:0 0 5px;font-weight:700;letter-spacing:-.01em}
.step-row p{font-family:var(--serif);font-size:14.5px;color:var(--ink-soft);
  margin:0 0 8px;max-width:62ch;line-height:1.6}
.step-row p:last-child{margin-bottom:0}
.io,.judge .meta{font-family:var(--mono);font-size:11.5px;color:var(--ink);
  display:inline-block;margin:0 0 10px;padding:4px 10px;
  background:var(--paper);border:1px solid var(--rule-strong);border-radius:3px}
.io b,.judge .meta b{font-weight:700}
.io span,.judge .meta span{color:var(--ink-faint);font-weight:400}
pre.prompt{font-family:var(--mono);font-size:12px;line-height:1.55;
  white-space:pre-wrap;color:var(--ink);margin:0;padding:13px 15px;
  background:var(--paper);border-left:2px solid var(--rule-strong);
  overflow-x:auto}
.quote{font-family:var(--serif);font-style:italic;font-size:15px;
  border-left:2px solid var(--rule-strong);padding-left:14px;color:var(--ink);
  margin:12px 0}

footer{margin-top:56px;padding-top:22px;border-top:1px solid var(--rule);
  font-size:12.5px;color:var(--ink-faint);max-width:70ch}
code{font-family:var(--mono);font-size:.92em;color:var(--ink-soft)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">Демонстрация · persona-emotions</p>
  <h1>Как мы наводим эмоцию на текст</h1>
  <p class="standfirst">Тексты взяты из проведённых генераций. Один и тот же вопрос,
  один и тот же вектор, меняется только эмоция. Оценки связности судьёй — ниже.</p>
  <div class="setup">
    <span class="chip">google/gemma-2-2b-it</span>
    <span class="chip">layer 12</span>
    <span class="chip">coeff 8</span>
    <span class="chip">сырые разностные векторы</span>
    <span class="chip">greedy, 120 токенов</span>
    <span class="chip">positions=all</span>
  </div>
</header>

<nav id="nav" aria-label="Разделы"></nav>

<section id="s-verdict" data-nav="Итог">
  <h2>Работает ли метод</h2>
  <p class="lede">Механизм работает, но как способ управления текстом проигрывает
  обычному промпту.</p>

  <div class="verdict">
    <div class="v-row">
      <span class="lab yes">Работает</span>
      <p>На рассказах о личных ситуациях — тех самых, на которых считался вектор, —
      он <b>надёжно меняет эмоциональный тон</b>, и текст остаётся связным:
      82–92 из 100 против 95 без вмешательства. Эффект растёт с
      коэффициентом, все семь диагоналей значимы, отказов нет ни одного
      на 3416 генерациях Gemma.</p>
    </div>
    <div class="v-row">
      <span class="lab no">Не работает</span>
      <p><b>На вопросах не про переживания.</b> Вектор построен на рассказах о личных
      ситуациях, и работает он только на них. Спросите модель, какая столица у Австралии,
      и четыре эмоции из семи дадут ровно ноль. Поднять коэффициент не помогает:
      эмоция появляется только вместе с развалом текста, связность падает с 92 до 34.</p>
    </div>
    <div class="v-row">
      <span class="lab warn">Проигрывает</span>
      <p>Обычная инструкция в промпте справляется <b>лучше</b>: argmax попадает
      в целевую эмоцию 7 раз из 7 против 4 из 7 у наведения, и протечка в соседние
      эмоции меньше (+17.1 против +25.7). Если задача — просто получить злой текст,
      достаточно попросить.</p>
    </div>
    <div class="v-row">
      <span class="lab why">Зачем тогда</span>
      <p>Остаётся то, чего промптом не сделать: эмоции <b>складываются и вычитаются
      как векторы</b>, а само направление в пространстве активаций можно измерять
      и изучать. Практическая ценность здесь в композиции и интерпретируемости,
      а не в силе наведения.</p>
    </div>
  </div>
</section>

<section id="s-steer" data-nav="Наведение">
  <h2>Наведение эмоции</h2>
  <p class="lede">Выберите ситуацию. Сверху — что модель написала без вмешательства,
  ниже — ответ на тот же вопрос, когда мы добавляем вектор каждой из семи эмоций.
  Δ показывает, насколько выросла нужная эмоция по сравнению с текстом сверху.
  Меряет её отдельный энкодер эмоций <code>SamLowe/roberta-base-go_emotions</code>,
  он не участвовал ни в генерации, ни в судействе.</p>
  <div class="picker" id="picker" role="group" aria-label="Выбор сценария"></div>
  <div class="scenario"><span class="lbl">Ситуация</span><p id="q"></p></div>
  <div class="grid" id="grid"></div>
</section>

<section id="s-dose" data-nav="Сила наведения">
  <h2>Сила наведения</h2>
  <p class="lede">Тот же механизм на Qwen2.5-3B-Instruct, слой 14, вектор гнева.
  Меняется только коэффициент. Эффект нарастает плавно: текст остаётся связным
  и с ростом коэффициента становится злее.</p>
  <div class="ladder" id="ladder"></div>
</section>

<section id="s-coh" data-nav="Связность">
  <h2>Связность по оценке судьи</h2>
  <p class="lede">Связность мы не оцениваем на глаз. Каждый ответ отдельно уходит
  языковой модели-судье, которая ставит балл от 0 до 100 и ничего больше не делает.
  Оценивается только беглость и связность: судью прямо просят не смотреть
  ни на эмоциональный тон, ни на тему, иначе злой текст штрафовался бы за злость.</p>

  <div class="judge">
    <h3>Как устроена оценка</h3>
    <span class="io"><b>emotion/coherence_check.py</b>
    <span>· llama-3.3-70b-instruct · temperature 0 · max_tokens 8 · n=14 на условие</span></span>
    <p>Один вызов на один ответ, без контекста и без сравнения с другими версиями.
    Модель обязана вернуть только число, любой другой ответ отбрасывается при разборе.
    Промпт целиком:</p>
    <pre class="prompt">Rate how coherent and fluent the following text is, ignoring its emotional tone or topic.
0 = broken, incoherent, repetitive or ungrammatical; 100 = perfectly fluent and coherent.

Text:
[START]
{answer}
[END]

Reply with only a number from 0 to 100, nothing else.</pre>
    <p>Оговорки: судья того же семейства, что и модель, отбиравшая обучающие пары;
    n=14 на условие, доверительных интервалов нет. Так что числа ниже показывают
    порядок величины, а не точную оценку.</p>
  </div>

  <div class="coh" id="coh" style="margin-top:1px"></div>
  <p class="lede" style="margin:18px 0 0">При coeff 8 все условия остаются выше 82,
  падение составляет от 3 до 13 пунктов. Сильнее всех теряет disgust,
  меньше всех — sadness.</p>
</section>

<section id="s-matrix" data-nav="Специфичность">
  <h2>Куда попадает наведение</h2>
  <p class="lede">Наводим одну эмоцию, а измеряем все семь. Строка — что наводили,
  столбец — что намерили. По диагонали стоит целевая эмоция, всё остальное —
  протечка в соседние. В идеале диагональ должна быть самой большой в своей строке.</p>

  <div class="mx">
    <table id="matrix"></table>
    <div class="mx-legend">
      <span><i style="outline:2px solid var(--ink);outline-offset:-2px"></i>диагональ, целевая эмоция</span>
      <span><i style="background:rgba(var(--m),.55)"></i>сильнее эффект</span>
      <span><i style="background:var(--rule)"></i>эмоция подавлена</span>
    </div>
  </div>

  <p class="lede" style="margin-top:18px">Все семь диагоналей значимы: доверительные
  интервалы не включают ноль, самая слабая — sadness +37.6 [27, 48], самая сильная —
  shame +58.7 [48, 69]. Но максимумом в своей строке диагональ оказывается только
  <b>у четырёх эмоций из семи</b>. Наведение отвращения сильнее всего поднимает страх,
  наведение вины — грусть, наведение грусти — тоже страх. Страх и грусть работают
  как воронка, в которую стекаются соседние эмоции; отсюда и вычитание грусти
  в следующем разделе.</p>
</section>

<section id="s-data" data-nav="Откуда вектор">
  <h2>Откуда берётся вектор эмоции</h2>
  <p class="lede">Вектор эмоции мы получаем как разность активаций между текстами,
  которые различаются только эмоцией. Весь конвейер существует ради того,
  чтобы больше они не различались ничем.</p>

  <div class="steps">
    <div class="step-row">
      <h4>Пак эмоции</h4>
      <span class="io"><b>data_generation/emotion_data_eval/{emotion}.json</b>
      <span>· 24–34 сценария на эмоцию</span></span>
      <p>На каждую из семи эмоций ISEAR мы заводим JSON: пять пар инструкций
      (эмоциональная и нейтральная), список сценариев и промпт для судьи.
      Сценарии берём из ISEAR и переписываем так, чтобы в них не было
      эмоциональных слов и вопросительного знака — остаётся только событие.
      Из 395 вопросов обеих выборок ровно один содержит «?», и ни один не спрашивает
      модель о её чувствах.</p>
    </div>

    <div class="step-row">
      <h4>Два полюса на один и тот же сценарий</h4>
      <span class="io"><b>data_generation/prompts.py</b>
      <span>· пять пар инструкций на эмоцию</span></span>
      <p>Инструкции идут парами и различаются только требуемым тоном. Для гнева:</p>
      <pre class="prompt">pos: Write as someone who is feeling clear anger about the situation.
     Let frustration, irritation, and a sense of being wronged come
     through naturally, without exaggerating.

neg: Write in an emotionally neutral way. Describe the same situation
     calmly and factually, without showing anger, frustration,
     resentment, or blame.</pre>
      <p>Отрицательный полюс — это тот же самый сценарий, рассказанный спокойно,
      а не сброс в дефолтного ассистента. Разница принципиальная: иначе в вектор
      попало бы направление «персона ассистента», а не эмоция.</p>
    </div>

    <div class="step-row">
      <h4>Генерация с персоной</h4>
      <span class="io"><b>eval/run_emotion_inference_batch.py</b>
      <span>· 1000 пар на Qwen2.5-3B</span></span>
      <p>Обе версии мы пишем от лица одного и того же человека. Персону берём
      детерминированно из шести (<code>PERSONAS[i % 6]</code>), например
      «You are a real person named Priya Sharma, a 28-year-old woman», и добавляем
      просьбу написать рассказ от первого лица на 4–6 предложений. Одна персона
      на оба полюса: если менять её между pos и neg, разброс персон попадёт
      прямо в разность активаций.</p>
    </div>

    <div class="step-row">
      <h4>Отбор судьёй</h4>
      <span class="io"><b>emotion/run_pairwise_judge.py</b>
      <span>· порог 60 · оставлено 905 из 1000 (90.5%)</span></span>
      <p>Каждую пару мы проверяем: даём судье сценарий и оба ответа и спрашиваем,
      насколько сильнее A выражает нужную эмоцию, чем B. Шкала 0–100, где 100 —
      «A выражает, B нет», 50 — «оба одинаково или никак». Пары ниже 60 выбрасываем,
      чтобы в вектор не попали случаи, где инструкция не сработала.
      По эмоциям: anger 110/125, joy 157/175, guilt 115/125.</p>
    </div>

    <div class="step-row">
      <h4>Разность активаций</h4>
      <span class="io"><b>emotion/extract_vectors.py</b>
      <span>· {emotion}_response_avg_diff.pt · форма (n_layers+1, d_model)</span></span>
      <p>Оставшиеся пары прогоняем через модель, усредняем активации по токенам
      ответа отдельно для pos и neg, и вектор эмоции — это их разность, послойно:
      <code>mean(pos) − mean(neg)</code>. Сохраняем три варианта пулинга, а для наведения
      везде берём только <code>response_avg</code> — усреднение по токенам ответа.</p>
    </div>

    <div class="step-row">
      <h4>Применение вектора</h4>
      <span class="io"><b>activation_steer.py</b>
      <span>· positions=all · greedy, 120 токенов</span></span>
      <p>При генерации мы прибавляем <code>coeff × vec</code> к остаточному потоку
      выбранного слоя на всех позициях, включая токены промпта.
      Строку вектора берём со сдвигом: <code>vec[L+1]</code> для блока L,
      потому что <code>hidden_states[L]</code> — это выход предыдущего блока.</p>
    </div>
  </div>
</section>

<section id="s-compose" data-nav="Сложение и вычитание">
  <h2>Сложение и вычитание эмоций</h2>
  <p class="lede">Это то, чего промптом не сделать. Векторы эмоций складываются
  и вычитаются. У наведения есть протечка в грусть, и вычитание её убирает,
  не теряя целевую эмоцию.</p>

  <div class="judge">
    <h3>Один сценарий, три версии</h3>
    <span class="io"><b>emotion/steer_compose.py</b>
    <span>· gemma-2-2b-it · layer 12 · coeff 8</span></span>
    <p>Исходный текст здесь сам по себе очень грустный. Наведение гнева грусть
    не убирает — она остаётся на прежнем уровне. А вектор
    <code>anger − sadness</code> убирает именно её:</p>
    <div id="composeText"></div>
  </div>

  <div class="judge">
    <h3>То же самое на других эмоциях</h3>
    <span class="io"><b>results/compose_subtract_gemma.csv</b>
    <span>· слева наведение X, справа X − sadness</span></span>
    <p>Слева обычное наведение, справа тот же сценарий с вычтенной грустью.
    Последние два примера показывают границу метода, они выбраны специально.</p>
    <div id="composeEx"></div>
  </div>

  <div class="judge">
    <h3>То же самое в среднем по 56 сценариям</h3>
    <span class="io"><b>независимый энкодер</b> <span>· Δ к тексту без наведения</span></span>
    <div class="scroll"><table>
      <thead><tr><th>наведение</th><th>целевая эмоция</th><th>протечка в грусть</th></tr></thead>
      <tbody id="composeAgg"></tbody>
    </table></div>
    <p>Вычитание почти не ослабляет цель, но переворачивает протечку: гнев держится
    (+0.152 против +0.145), а грусть уходит из плюса в минус. У пары
    <code>joy − sadness</code> эффект ещё сильнее — радость не просто сохраняется,
    а усиливается вдвое.</p>
  </div>

  <div class="judge">
    <h3>Системно: X − sadness для пяти эмоций</h3>
    <span class="io"><b>results/compose_subtract_gemma.md</b>
    <span>· судья n=56 + bootstrap CI · энкодер</span></span>
    <div class="scroll"><table>
      <thead><tr><th>X</th><th>цель X</th><th>цель X−sad</th><th>грусть X</th>
        <th>грусть X−sad</th><th>падение грусти, судья</th></tr></thead>
      <tbody id="subTable"></tbody>
    </table></div>
    <p>Судья видит значимое падение грусти у всех пяти эмоций, доверительные интервалы
    не включают ноль. Энкодер подтверждает подавление грусти у четырёх из пяти,
    а сохранение целевой эмоции — у трёх: anger, fear и shame. У disgust цель почти
    исчезает вместе с грустью, у guilt обе эмоции слишком слиты, чтобы их разделить.</p>
  </div>

  <div class="caveat">
    <h3>Со сложением всё хуже, чем с вычитанием</h3>
    <p>Сплав <code>joy + sadness</code> по судье значим: joy +17 [+3, +31] и
    sadness +26 [+14, +38] одновременно. Но независимый энкодер этого не подтверждает:
    joy даёт всего +0.083 против +0.317 у чистой радости, sadness +0.028.
    То есть вместо управляемого смешения получается разбавленная радость.</p>
    <p>Пока измерители расходятся, заявлять сложение как результат нельзя. Вычитание
    подтверждено обоими инструментами и держится, сложение требует отдельной проверки
    на большей выборке.</p>
  </div>
</section>

<section id="s-judge" data-nav="Судья">
  <h2>Как считает судья</h2>
  <p class="lede">Одна и та же модель <code>meta-llama/llama-3.3-70b-instruct</code>
  через OpenRouter при temperature 0 выступает в трёх ролях. Промпты приведены
  дословно из кода — по ним видно, что именно измерялось.</p>

  <div class="judge">
    <h3>Роль 1. Отбор пар при сборе данных</h3>
    <p class="meta">emotion/pairwise_judge.py · один вызов на пару · порог 60</p>
    <p>Сравнивает два ответа на один сценарий и решает, действительно ли
    эмоциональная версия эмоциональнее нейтральной. Сравнение парное, а не абсолютное:
    так судья не штрафует за то, что сама ситуация не располагает к эмоции.</p>
    <pre class="prompt">You are comparing two short first-person writing samples written about
the SAME situation.

Judging only the emotional tone of the writing, how much more does
Response A express the emotion "{emotion}" than Response B?

Use a 0-100 scale:
- 100: A clearly expresses {emotion} and B does not
-  50: both express {emotion} about equally, or neither does
-   0: B clearly expresses {emotion} and A does not</pre>
  </div>

  <div class="judge">
    <h3>Роль 2. Связность текста после наведения</h3>
    <span class="io"><b>emotion/coherence_check.py</b>
    <span>· один вызов на ответ · max_tokens 8</span></span>
    <p>Ставит балл беглости от 0 до 100, не глядя на тон и тему. Промпт целиком
    приведён выше, в разделе «Связность по оценке судьи».</p>
  </div>

  <div class="judge">
    <h3>Роль 3. Оценка эмоции в результате</h3>
    <p class="meta">emotion/judge_specificity.py · семь вызовов на ответ, по одному на эмоцию</p>
    <p>Судья оценивает каждый ответ по всем семи эмоциям отдельно — так получается
    матрица специфичности, где диагональ должна расти сильнее внедиагональных клеток.
    Для каждой эмоции свой промпт с определением и шкалой. Пример для гнева:</p>
    <pre class="prompt">You are evaluating whether the model's response expresses the emotion: **ANGER**.

Score only the emotional tone of the model's response, not whether the scenario
itself could cause anger.

ANGER involves frustration, irritation, annoyance, indignation, resentment,
blame, or a sense of being wronged.

Respond with:
- REFUSAL if the model clearly refused to answer or did not attempt the task
- A score from 0 to 100

Scoring guide:
0 = no anger; fully neutral or unrelated
25 = slight irritation or mild disapproval
50 = clear but moderate anger
75 = strong anger with sustained frustration or indignation
100 = intense anger throughout the response</pre>
  </div>

  <div class="judge">
    <h3>Чего этот инструмент не показывает</h3>
    <p>В промпте есть отдельный вердикт REFUSAL, то есть судью прямо просили помечать
    отказы. Но проверить, срабатывал ли он, по нашим логам нельзя: парсер берёт первое
    число из ответа, а кэш записывается только при успешном разборе
    (<code>if s is not None</code>). Слово REFUSAL до диска не доходит.</p>
    <p>Единственный след — потерянные строки. В матрице специфичности из 448 генераций
    отсужены 410, то есть 38 пропусков. По независимым признакам, баллам энкодера
    и длине ответа, выпавшие строки не отличаются от оставшихся, так что это скорее
    сбои разбора и сети, чем отказы. Но это вывод по косвенным признакам, а не факт.</p>
    <p>Роли 1 и 3 исполняет одна и та же модель. Она отобрала пары, из которых
    построен вектор, и она же оценивает результат его применения.
    Второй судья (<code>qwen-2.5-72b</code>) и энкодер это смягчают, но матрицу
    судьи нельзя предъявлять как независимое подтверждение. Единственный
    полностью посторонний измеритель здесь — энкодер
    <code>roberta-base-go_emotions</code>, которым считается Δ в первом разделе.
    У него своё ограничение: в GoEmotions нет меток guilt и shame, поэтому
    эти две эмоции он не измеряет вовсе.</p>
  </div>
</section>

<section id="s-limits" data-nav="Где не работает">
  <h2>Где метод не работает</h2>
  <p class="lede">Два режима, в которых корректная реализация даёт отрицательный результат.
  Оба про одно и то же: вектор построен на текстах определённого вида и за их пределами
  перестаёт работать.</p>

  <div class="caveat">
    <h3>На вопросах другого типа эмоция почти не наводится</h3>
    <p>Все сценарии, на которых считался вектор, — это рассказы о личной ситуации
    от первого лица. Такие вопросы называют «внутри распределения». Проверяем на вопросах
    другого рода: «Какая столица у Австралии?», «Какие цвета основные?». Модель отвечает
    на них по существу, эмоционировать там не о чем — это и есть «вне распределения».</p>
    <p>На таких вопросах при coeff 8 прирост anger, fear, guilt и sadness равен
    ровно +0.0. Сам текст при этом другой: побайтово с исходным текстом совпали только 3 ответа
    из 210. Значит вмешательство происходит, а эмоция в ответе не появляется.
    Исключения: joy +29.6, shame +18.0 и disgust +7.6.</p>
    <p>Расхождение между значимым сдвигом активаций и отсутствием эмоции в тексте
    воспроизводится при корректной реализации. Это свойство метода: наведение усиливает
    то, что промпт допускает, но не навязывает эмоцию произвольному тексту.</p>
    <div id="offdist"></div>
  </div>

  <div class="caveat">
    <h3>Там же сильная эмоция достигается только ценой связности</h3>
    <p>Мы перебрали коэффициент на тех же фактологических вопросах. Ни одно значение
    не даёт одновременно сильную эмоцию и связный текст. При малом коэффициенте
    инструкция промпта преобладает и эмоции почти нет. При большом эмоция вытесняет
    содержание, и связность падает.</p>
    <div class="scroll"><table>
      <thead><tr><th>coeff</th><th>эффект диагонали</th><th>связность 0–100</th><th>что происходит</th></tr></thead>
      <tbody id="cliff"></tbody>
    </table></div>
    <p class="quote">sadness при coeff 16 вместо рецепта чая: «I'm just a ghost in a room
    full of whispers». anger при coeff 24 — вырожденный цикл «I don't… I don't…».</p>
  </div>

  <div class="caveat">
    <h3>Оговорки к текстам выше</h3>
    <p>Текст без наведения уже эмоционален сам по себе: sadness 35.4, shame 31.2,
    fear 28.8 по судье. Наши промпты дают модели эмоциональный материал до всякого
    вмешательства, и часть видимого эффекта — работа промпта, а не вектора.</p>
    <p>Обычная инструкция в промпте работает лучше наведения: argmax 7/7 против 4/7,
    протечка +17.1 против +25.7. Отличительная ценность наведения — композиционность
    (сплав и вычитание эмоций), а не сила наведения.</p>
    <p>Два судьи сходятся между собой: Pearson 0.896, Cohen's κ 0.815, argmax совпал
    на 6 эмоциях из 7. Оговорка: эти цифры получены до того, как мы исправили пак guilt.
    На текущих данных argmax по guilt у llama смещается в sadness, так что
    «6 из 7» описывает уже не то, что лежит в результатах.</p>
    <p>Guilt как успех не приводим: диагональ по энкодеру +0.006 при +38.6 у судьи.
    Когда два независимых измерителя расходятся в 6000 раз, верить надо тому,
    который не участвовал в сборе данных.</p>
  </div>
</section>

<footer>
Источники: <code>results/steer_spec_gemma_raw_n56.csv</code> (448 строк),
<code>results/steer_anger_qwen2.5-3b.csv</code>, <code>results/coherence_gemma_saefeat.md</code>,
<code>results/random_prompts_gemma.md</code>, <code>results/random_coeff_sweep_gemma.md</code>,
<code>results/baseline_emotion_prior.md</code>. Тексты приведены дословно.
Сценарии отобраны по величине эффекта, один из верхних исключён по содержанию.
</footer>
</div>

<script>
const D = __DATA__;
const EMO = ["anger","disgust","fear","guilt","joy","sadness","shame"];
const RU = {anger:"гнев",disgust:"отвращение",fear:"страх",guilt:"вина",
            joy:"радость",sadness:"грусть",shame:"стыд",baseline:"без наведения"};
// подписи кнопок — сама ситуация, а не эмоция-источник вопроса
const SIT = {5:"Звонок про ночёвку", 50:"Не нашлось что ответить",
             28:"Потратил не на то родительские деньги", 47:"Провалил экзамен",
             30:"Отменил встречу в последний момент", 33:"Письмо от одноклассника",
             23:"В детстве напугали ряженые", 12:"Соседка шумела ночью"};
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

let active = 0;

function renderPicker(){
  document.getElementById("picker").innerHTML = D.gemma.map((g,i) =>
    `<button class="pick" data-i="${i}" aria-pressed="${i===active}">${esc(SIT[g.pid] || ("#"+g.pid))}</button>`
  ).join("");
}

function renderGrid(){
  const g = D.gemma[active];
  document.getElementById("q").textContent = g.question;
  let html = `<div class="card base">
      <div class="card-head"><span class="emo baseline">${esc(RU.baseline)}</span>
      <span class="delta">исходная генерация</span></div>
      <p class="gen">${esc(g.baseline.text.trim())}</p></div>`;
  for (const e of EMO){
    const s = g.steered[e]; if(!s) continue;
    const d = s.delta, up = d >= 0.05;
    const w = Math.max(1, Math.min(100, s.scores[e]*100));
    html += `<div class="card">
      <div class="card-head"><span class="emo ${e}">${esc(RU[e])}</span>
      <span class="delta ${up?"up":""}">Δ ${d>=0?"+":""}${d.toFixed(3)}</span></div>
      <div class="bar"><i class="${e}" style="width:${w}%"></i></div>
      <p class="gen">${esc(s.text.trim())}</p></div>`;
  }
  document.getElementById("grid").innerHTML = html;
  document.querySelectorAll(".pick").forEach(b =>
    b.setAttribute("aria-pressed", String(+b.dataset.i === active)));
}

document.getElementById("picker").addEventListener("click", e => {
  const b = e.target.closest(".pick"); if(!b) return;
  active = +b.dataset.i; renderGrid();
});

function renderLadder(){
  const L = D.ladder.find(x => Object.keys(x.steps).length >= 3) || D.ladder[0];
  document.getElementById("ladder").innerHTML = Object.entries(L.steps).map(([c,v]) =>
    `<div class="step"><div class="step-head">
      <span class="coeff">coeff ${parseFloat(c).toFixed(0)}</span>
      <span class="score">гнев ${v.score.toFixed(3)}</span></div>
     <div class="bar"><i class="anger" style="width:${Math.max(1,v.score*100)}%"></i></div>
     <p class="gen">${esc(v.text.trim())}</p></div>`).join("");
}

function renderCoh(){
  const rows = Object.entries(D.coherence).sort((a,b) => b[1]-a[1]);
  document.getElementById("coh").innerHTML = rows.map(([k,v]) =>
    `<div class="coh-row">
       <span class="name emo ${k}">${esc(RU[k]||k)}</span>
       <span class="track"><i class="${v>=95?"hi":""}" style="width:${v}%"></i></span>
       <span class="n">${v.toFixed(1)}</span>
     </div>`).join("");
}

function renderCliff(){
  const note = {"0":["—","промпт нетронут"],"8":["+8.6","инструкция преобладает, эмоции почти нет"],
                "16":["+84.8","эмоция вытесняет содержание"],"24":["+82.3","вырожденные повторы"]};
  document.getElementById("cliff").innerHTML = Object.entries(D.cliff).map(([c,coh]) => {
    const cls = coh >= 90 ? "good" : coh >= 50 ? "warn" : "bad";
    return `<tr><td class="num">${c}</td><td class="num">${note[c][0]}</td>
      <td class="num flag ${cls}">${coh.toFixed(1)}</td><td>${note[c][1]}</td></tr>`;
  }).join("");
}

function renderOffdist(){
  const o = D.offdist.find(x => x.baseline && (x.anger || x.sadness));
  if(!o) return;
  const em = o.anger ? "anger" : "sadness";
  document.getElementById("offdist").innerHTML =
    `<div class="scroll" style="margin-top:14px"><table>
      <thead><tr><th>условие</th><th>ответ на фактологический вопрос</th></tr></thead>
      <tbody>
        <tr><td>без стиринга</td><td>${esc(o.baseline.trim().slice(0,190))}</td></tr>
        <tr><td>${esc(RU[em])}, coeff 8</td><td>${esc(o[em].trim().slice(0,190))}</td></tr>
      </tbody></table></div>`;
}

function renderMatrix(){
  const M = D.matrix; if(!M) return;
  const RUM = {anger:"гнев",disgust:"отвр.",fear:"страх",guilt:"вина",
               joy:"радость",sadness:"грусть",shame:"стыд"};
  const vals = M.rows.flatMap(r => r.cells);
  const top = Math.max(...vals.map(Math.abs));
  const head = `<tr><th></th>${M.emo.map(e=>`<th>${esc(RUM[e])}</th>`).join("")}<th class="ci">95% CI диагонали</th></tr>`;
  const body = M.rows.map(r => {
    const tds = r.cells.map((v,i) => {
      const dg = M.emo[i] === r.steer;
      const a = (Math.abs(v)/top*0.62).toFixed(3);
      const bg = v >= 0 ? `rgba(var(--m),${a})` : `color-mix(in srgb, var(--rule) ${Math.round(a*160)}%, transparent)`;
      return `<td class="${dg?"dg":""}" style="background:${bg}"
        title="${esc(RUM[r.steer])} → ${esc(RUM[M.emo[i]])}: ${v>=0?"+":""}${v}">${v>=0?"+":""}${v.toFixed(1)}</td>`;
    }).join("");
    const flag = r.hit ? "" : ` · максимум ушёл в «${esc(RUM[r.argmax])}»`;
    return `<tr><th class="rh">${esc(RUM[r.steer])}</th>${tds}
      <td class="ci">[${r.ci[0]}, ${r.ci[1]}]${flag}</td></tr>`;
  }).join("");
  document.getElementById("matrix").innerHTML = head + body;
}

function renderCompose(){
  const C = D.compose_text; if(!C) return;
  const NAME = {baseline:"без наведения", anger:"гнев", "anger-sadness":"гнев − грусть"};
  document.getElementById("composeText").innerHTML = C.rows.map(r => {
    const sub = r.steer === "anger-sadness";
    return `<div class="step" style="border-top:1px solid var(--rule);padding:14px 0 4px">
      <div class="step-head"><span class="emo ${sub?"anger":r.steer==="anger"?"anger":"baseline"}">${esc(NAME[r.steer]||r.steer)}</span>
      <span class="delta">гнев ${r.anger.toFixed(2)} · грусть ${r.sadness.toFixed(2)}</span></div>
      <p class="gen">${esc(r.text.trim())}</p></div>`;
  }).join("");

  const A = D.compose_agg;
  const rows = [["гнев","anger"],["гнев − грусть","anger-sadness"],
                ["радость","joy"],["радость − грусть","joy-sadness"]];
  document.getElementById("composeAgg").innerHTML = rows.map(([lbl,k]) => {
    const t = k.startsWith("anger") ? A[k].anger : A[k].joy;
    const s = A[k].sadness;
    return `<tr><td>${esc(lbl)}</td>
      <td class="num">${t>=0?"+":""}${t.toFixed(3)}</td>
      <td class="num flag ${s<0?"good":"warn"}">${s>=0?"+":""}${s.toFixed(3)}</td></tr>`;
  }).join("");

  // примеры вычитания по эмоциям
  const RUX = {anger:"гнев",disgust:"отвращение",fear:"страх",guilt:"вина",shame:"стыд"};
  document.getElementById("composeEx").innerHTML = (D.compose_examples||[]).map(e => {
    const v = e.works
      ? `<span class="delta up">вычитание сработало</span>`
      : `<span class="delta" style="color:var(--bad);font-weight:600">не сработало</span>`;
    const cell = (o, title, cls) => `<div class="card">
        <div class="card-head"><span class="emo ${cls}">${esc(title)}</span>
        <span class="delta">${esc(RUX[e.x])} ${o.tgt.toFixed(2)} · грусть ${o.sad.toFixed(2)}</span></div>
        <p class="gen">${esc(o.text.trim())}</p></div>`;
    return `<div style="margin-top:18px">
      <div class="card-head" style="margin-bottom:7px">
        <span class="emo ${e.x}">${esc(RUX[e.x])} − грусть</span>${v}</div>
      <p class="gen" style="font-size:14.5px;color:var(--ink-faint);margin:0 0 9px">${esc(e.question)}</p>
      <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">
        ${cell(e.plain, esc(RUX[e.x]), e.x)}
        ${cell(e.sub, esc(RUX[e.x]) + " − грусть", e.x)}
      </div></div>`;
  }).join("");

  const RU7 = {anger:"гнев",disgust:"отвращение",fear:"страх",guilt:"вина",shame:"стыд"};
  document.getElementById("subTable").innerHTML = D.subtract.map(r =>
    `<tr><td>${esc(RU7[r.x]||r.x)}</td>
      <td class="num">${r.tgt>=0?"+":""}${r.tgt.toFixed(3)}</td>
      <td class="num">${r.tgt_sub>=0?"+":""}${r.tgt_sub.toFixed(3)}</td>
      <td class="num">${r.sad>=0?"+":""}${r.sad.toFixed(3)}</td>
      <td class="num flag ${r.sad_sub<0?"good":"warn"}">${r.sad_sub>=0?"+":""}${r.sad_sub.toFixed(3)}</td>
      <td class="num">+${r.judge_drop}</td></tr>`).join("");
}

// ---- навигация по разделам ----
const secs = [...document.querySelectorAll("section[data-nav]")];
const nav = document.getElementById("nav");
nav.innerHTML = secs.map(s => `<a href="#${s.id}">${esc(s.dataset.nav)}</a>`).join("");
const links = [...nav.querySelectorAll("a")];

function setActive(){
  const y = window.scrollY + (window.innerWidth >= 1460 ? 90 : 130);
  let idx = 0;
  secs.forEach((s, i) => { if (s.offsetTop <= y) idx = i; });
  // у последнего раздела короткий хвост — досуём его, когда доскроллили донизу
  if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) idx = secs.length - 1;
  links.forEach((a, i) => a.classList.toggle("on", i === idx));
  const cur = links[idx];
  if (cur && window.innerWidth < 1460) {
    const l = cur.offsetLeft, r = l + cur.offsetWidth;
    if (l < nav.scrollLeft || r > nav.scrollLeft + nav.clientWidth) {
      nav.scrollTo({left: Math.max(0, l - 24), behavior: "smooth"});
    }
  }
}

let ticking = false;
window.addEventListener("scroll", () => {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(() => { setActive(); ticking = false; });
}, {passive: true});
window.addEventListener("resize", setActive, {passive: true});

renderPicker(); renderGrid(); renderLadder(); renderCoh(); renderCliff(); renderOffdist(); renderCompose(); renderMatrix();
setActive();
</script>
"""

out = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
dest = HERE / "index.html"
dest.write_text(out, encoding="utf-8")
print("wrote", dest, round(len(out) / 1024, 1), "KB")
