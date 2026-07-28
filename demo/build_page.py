"""Сборка страницы демо из данных прогона.

Прежний build_demo.py собирал страницу под одну модель: имя, слой и числа были
вписаны в текст, подписи сценариев привязаны к номерам промптов конкретного
прогона, а данные брались из вручную подготовленного JSON.

Здесь всё приходит из `collect_demo_data.py`. Разделы, для которых данных нет,
не рисуются вовсе — вместо того чтобы показывать чужие числа.

Usage:
    python demo/collect_demo_data.py --run runs/Qwen3-1.7B
    python demo/build_page.py --data demo/demo_data_Qwen3-1.7B.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RU = {"anger": "гнев", "disgust": "отвращение", "fear": "страх", "guilt": "вина",
      "joy": "радость", "sadness": "грусть", "shame": "стыд"}


def short_label(question: str, words: int = 4) -> str:
    """Короткая подпись кнопки из самого вопроса.

    Прежняя версия держала подписи списком, привязанным к номерам промптов
    одного прогона: на другой модели номера те же, а сценарии другие, и
    подписи начинали врать.
    """
    q = re.sub(r"^You (are|have|receive|learn|tell|gave|failed|used|canceled|got)\b", "",
               question.strip(), flags=re.I).strip()
    q = re.sub(r"\s+", " ", q).strip(" .,:")
    parts = q.split(" ")
    lab = " ".join(parts[:words])
    return (lab[:1].upper() + lab[1:]) if lab else question[:28]


def section_verdict(d: dict) -> str:
    """Итог: что говорят числа этого прогона. Без данных — без раздела."""
    m = d.get("matrix")
    coh = d.get("coherence") or {}
    cliff = d.get("cliff") or {}
    if not m:
        return ""
    hits = sum(1 for r in m["rows"] if r["hit"])
    diag = sum(r["diag"] for r in m["rows"]) / len(m["rows"])
    rows = [("yes", "Работает",
             f"Наведение поднимает целевую эмоцию в среднем на {diag:+.3f} по независимому "
             f"классификатору. Максимум строки совпадает с целью у {hits} эмоций из 7.")]
    if coh:
        base = coh.get("baseline")
        others = [v for k, v in coh.items() if k != "baseline"]
        if base and others:
            rows.append(("warn", "Цена",
                         f"Связность падает с {base:.1f} до {min(others):.1f}–{max(others):.1f} "
                         "по оценке судьи."))
    if cliff:
        worst = max(cliff.items(), key=lambda kv: kv[1]["degen"])
        rows.append(("no", "Граница",
                     f"При коэффициенте {float(worst[0]):g} вырожденных ответов "
                     f"{worst[1]['degen']:.0f}%: сильнее наводить нечем, текст рассыпается."))
    body = "".join(
        f'<div class="v-row"><span class="lab {c}">{t}</span><p>{p}</p></div>'
        for c, t, p in rows)
    return f'''<section id="s-verdict" data-nav="Итог">
  <h2>Что показывает прогон</h2>
  <div class="verdict">{body}</div>
</section>'''


def section_steer(d: dict) -> str:
    if not d.get("scenarios"):
        return ""
    return '''<section id="s-steer" data-nav="Наведение">
  <h2>Наведение эмоции</h2>
  <p class="lede">Выберите ситуацию. Сверху — что модель написала без вмешательства,
  ниже — ответ на тот же вопрос при добавлении вектора каждой из семи эмоций.
  Δ показывает, насколько выросла нужная эмоция. Меряет её отдельный энкодер
  <code>SamLowe/roberta-base-go_emotions</code>, он не участвовал ни в генерации,
  ни в судействе.</p>
  <div class="picker" id="picker" role="group" aria-label="Выбор ситуации"></div>
  <div class="scenario"><span class="lbl">Ситуация</span><p id="q"></p></div>
  <div class="grid" id="grid"></div>
</section>'''


def section_ladder(d: dict) -> str:
    if not d.get("ladder"):
        return ""
    return '''<section id="s-dose" data-nav="Сила наведения">
  <h2>Сила наведения</h2>
  <p class="lede">Один и тот же вопрос, меняется только коэффициент. Видно, где
  наведение усиливается, а где начинает разрушать текст.</p>
  <div class="ladder" id="ladder"></div>
</section>'''


def section_cliff(d: dict) -> str:
    if not d.get("cliff"):
        return ""
    return '''<section id="s-cliff" data-nav="Где граница">
  <h2>Где проходит граница</h2>
  <p class="lede">Свип коэффициента на одной эмоции. Балл целевой эмоции растёт,
  пока текст держится, и падает, когда генерация вырождается в повтор: повторяющийся
  текст энкодер эмоциональным не считает.</p>
  <div class="scroll"><table>
    <thead><tr><th>коэффициент</th><th>балл эмоции</th><th>вырожденных</th></tr></thead>
    <tbody id="cliff"></tbody>
  </table></div>
</section>'''


def section_matrix(d: dict) -> str:
    if not d.get("matrix"):
        return ""
    return '''<section id="s-matrix" data-nav="Специфичность">
  <h2>Куда попадает наведение</h2>
  <p class="lede">Наводим одну эмоцию, а измеряем все семь. Строка — что наводили,
  столбец — что намерили. По диагонали целевая эмоция, остальное — протечка.
  В идеале диагональ должна быть максимумом своей строки.</p>
  <div class="mx"><table id="matrix"></table>
    <div class="mx-legend">
      <span><i style="outline:2px solid var(--ink);outline-offset:-2px"></i>диагональ</span>
      <span><i style="background:rgba(var(--m),.55)"></i>сильнее эффект</span>
      <span><i style="background:var(--rule)"></i>эмоция подавлена</span>
    </div>
  </div>
  <p class="lede" id="mxnote" style="margin-top:18px"></p>
</section>'''


def section_coherence(d: dict) -> str:
    if not d.get("coherence"):
        return ""
    return '''<section id="s-coh" data-nav="Связность">
  <h2>Связность по оценке судьи</h2>
  <p class="lede">Судья оценивает беглость каждого ответа от 0 до 100, не глядя
  на эмоциональный тон и тему. Чем сильнее наведение, тем ниже оценка.</p>
  <div class="coh" id="coh"></div>
</section>'''


def build(d: dict) -> str:
    css = (HERE / "_style.css.txt").read_text(encoding="utf-8")
    meta = d.get("meta", {})
    model = meta.get("model", "модель")
    layer = meta.get("layer")
    coeff = meta.get("coeff")
    env = meta.get("env", {})

    chips = [model]
    if layer is not None:
        chips.append(f"слой {layer}")
    if coeff is not None:
        chips.append(f"coeff {coeff:g}" if isinstance(coeff, (int, float)) else f"coeff {coeff}")
    chips.append(f"вариант: {meta.get('variant', 'raw')}")
    if env.get("dtype"):
        chips.append(str(env["dtype"]).replace("torch.", ""))
    if env.get("git_sha"):
        chips.append(f"код {env['git_sha']}")

    sections = "\n\n".join(x for x in (
        section_verdict(d), section_steer(d), section_ladder(d),
        section_cliff(d), section_matrix(d), section_coherence(d)) if x)

    return f'''<meta charset="utf-8">
<title>Наведение эмоций: {model}</title>
{css}
<div class="wrap">
<header class="top">
  <p class="eyebrow">Прогон · {meta.get("slug", "")}</p>
  <h1>Наведение эмоций на текст</h1>
  <p class="standfirst">Все тексты ниже взяты из этого прогона дословно. Один и тот же
  вопрос, один и тот же вектор, меняется только эмоция.</p>
  <div class="setup">{"".join(f'<span class="chip">{c}</span>' for c in chips)}</div>
</header>

<nav id="nav" aria-label="Разделы"></nav>

{sections}

<footer>
Собрано из <code>runs/{meta.get("slug", "")}</code> скриптами
<code>demo/collect_demo_data.py</code> и <code>demo/build_page.py</code>.
Тексты приведены дословно, числа пересчитываются при каждой сборке.
</footer>
</div>

<script>
const D = {json.dumps(d, ensure_ascii=False)};
const EMO = ["anger","disgust","fear","guilt","joy","sadness","shame"];
const RU = {json.dumps(RU, ensure_ascii=False)};
const esc = s => String(s).replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
let active = 0;

function renderSteer(){{
  if(!D.scenarios || !D.scenarios.length) return;
  document.getElementById("picker").innerHTML = D.scenarios.map((s,i) =>
    `<button class="pick" data-i="${{i}}" aria-pressed="${{i===active}}">${{esc(s.label)}}</button>`).join("");
  const g = D.scenarios[active];
  document.getElementById("q").textContent = g.question;
  let html = `<div class="card base"><div class="card-head">
      <span class="emo baseline">без наведения</span>
      <span class="delta">исходная генерация</span></div>
      <p class="gen">${{esc((g.baseline.text||"").trim())}}</p></div>`;
  for(const e of EMO){{
    const s = g.steered[e]; if(!s) continue;
    const w = Math.max(1, Math.min(100, (s.scores[e]||0)*100));
    html += `<div class="card"><div class="card-head">
      <span class="emo ${{e}}">${{esc(RU[e])}}</span>
      <span class="delta ${{s.delta>=0.05?"up":""}}">Δ ${{s.delta>=0?"+":""}}${{s.delta.toFixed(3)}}</span></div>
      <div class="bar"><i class="${{e}}" style="width:${{w}}%"></i></div>
      <p class="gen">${{esc((s.text||"").trim())}}</p></div>`;
  }}
  document.getElementById("grid").innerHTML = html;
  document.querySelectorAll(".pick").forEach(b =>
    b.setAttribute("aria-pressed", String(+b.dataset.i === active)));
}}

function renderLadder(){{
  if(!D.ladder || !D.ladder.length) return;
  const L = D.ladder[0];
  document.getElementById("ladder").innerHTML = Object.entries(L.steps).map(([c,v]) =>
    `<div class="step"><div class="step-head">
       <span class="coeff">coeff ${{parseFloat(c).toFixed(0)}}</span>
       <span class="score">балл ${{v.score.toFixed(3)}}</span></div>
     <div class="bar"><i class="anger" style="width:${{Math.max(1,Math.min(100,v.score*100))}}%"></i></div>
     <p class="gen">${{esc((v.text||"").trim())}}</p></div>`).join("");
}}

function renderCliff(){{
  if(!D.cliff) return;
  const rows = Object.entries(D.cliff).sort((a,b)=>parseFloat(a[0])-parseFloat(b[0]));
  document.getElementById("cliff").innerHTML = rows.map(([c,v]) => {{
    const cls = v.degen <= 10 ? "good" : v.degen <= 40 ? "warn" : "bad";
    return `<tr><td class="num">${{parseFloat(c).toFixed(0)}}</td>
      <td class="num">${{v.score.toFixed(3)}}</td>
      <td class="num flag ${{cls}}">${{v.degen.toFixed(0)}}%</td></tr>`;
  }}).join("");
}}

function renderMatrix(){{
  const M = D.matrix; if(!M) return;
  const top = Math.max(...M.rows.flatMap(r => r.cells.map(Math.abs)), 1e-9);
  const head = `<tr><th></th>${{M.emo.map(e=>`<th>${{esc(RU[e]||e)}}</th>`).join("")}}<th class="ci">итог</th></tr>`;
  const body = M.rows.map(r => {{
    const tds = r.cells.map((v,i) => {{
      const dg = M.emo[i] === r.steer;
      const a = (Math.abs(v)/top*0.62).toFixed(3);
      const bg = v >= 0 ? `rgba(var(--m),${{a}})`
                        : `color-mix(in srgb, var(--rule) ${{Math.round(a*160)}}%, transparent)`;
      return `<td class="${{dg?"dg":""}}" style="background:${{bg}}">${{v>=0?"+":""}}${{v.toFixed(3)}}</td>`;
    }}).join("");
    const note = r.hit ? "цель" : `максимум в «${{esc(RU[r.argmax]||r.argmax)}}»`;
    return `<tr><th class="rh">${{esc(RU[r.steer]||r.steer)}}</th>${{tds}}<td class="ci">${{note}}</td></tr>`;
  }}).join("");
  document.getElementById("matrix").innerHTML = head + body;
  const hits = M.rows.filter(r => r.hit).length;
  document.getElementById("mxnote").textContent =
    `Диагональ оказалась максимумом своей строки у ${{hits}} эмоций из ${{M.rows.length}}.`;
}}

function renderCoh(){{
  if(!D.coherence) return;
  const rows = Object.entries(D.coherence).sort((a,b)=>b[1]-a[1]);
  document.getElementById("coh").innerHTML = rows.map(([k,v]) =>
    `<div class="coh-row"><span class="name emo ${{k}}">${{esc(RU[k]||k)}}</span>
     <span class="track"><i class="${{v>=95?"hi":""}}" style="width:${{v}}%"></i></span>
     <span class="n">${{v.toFixed(1)}}</span></div>`).join("");
}}

const picker = document.getElementById("picker");
if(picker) picker.addEventListener("click", e => {{
  const b = e.target.closest(".pick"); if(!b) return;
  active = +b.dataset.i; renderSteer();
}});

const secs = [...document.querySelectorAll("section[data-nav]")];
const nav = document.getElementById("nav");
nav.innerHTML = secs.map(s => `<a href="#${{s.id}}">${{esc(s.dataset.nav)}}</a>`).join("");
const links = [...nav.querySelectorAll("a")];
function setActive(){{
  const y = window.scrollY + (window.innerWidth >= 1460 ? 90 : 130);
  let idx = 0;
  secs.forEach((s,i) => {{ if(s.offsetTop <= y) idx = i; }});
  if(window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) idx = secs.length - 1;
  links.forEach((a,i) => a.classList.toggle("on", i === idx));
}}
let ticking = false;
window.addEventListener("scroll", () => {{
  if(ticking) return; ticking = true;
  requestAnimationFrame(() => {{ setActive(); ticking = false; }});
}}, {{passive:true}});
window.addEventListener("resize", setActive, {{passive:true}});

renderSteer(); renderLadder(); renderCliff(); renderMatrix(); renderCoh(); setActive();
</script>
'''


def main() -> None:
    ap = argparse.ArgumentParser(description="Собрать страницу демо из данных прогона.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = json.loads(args.data.read_text(encoding="utf-8"))
    for s in d.get("scenarios", []):
        s["label"] = short_label(s.get("question", ""))

    out = args.out or (HERE / f"index_{d.get('meta', {}).get('slug', 'run')}.html")
    out.write_text(build(d), encoding="utf-8")
    kb = out.stat().st_size // 1024
    shown = [k for k in ("scenarios", "ladder", "cliff", "matrix", "coherence") if d.get(k)]
    print(f"записано {out} ({kb} КБ)")
    print(f"  разделы: {', '.join(shown)}")


if __name__ == "__main__":
    main()
