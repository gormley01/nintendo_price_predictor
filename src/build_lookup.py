import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def safe_float(v, d=2):
    try:
        f = float(v)
        return round(f, d) if not np.isnan(f) else None
    except Exception:
        return None


def safe_int(v):
    try:
        f = float(v)
        return int(f) if not np.isnan(f) else None
    except Exception:
        return None


def parse_history_monthly(json_str):
    if not isinstance(json_str, str) or not json_str.strip():
        return {}
    try:
        raw = ast.literal_eval(json_str)
        monthly = {}
        for d, p in raw.items():
            mk = str(d)[:7]
            monthly.setdefault(mk, []).append(float(p))
        return {k: round(sum(v) / len(v), 2) for k, v in monthly.items() if int(k[:4]) >= 2005}
    except Exception:
        return {}


def to_annual(monthly):
    annual = {}
    for mk, price in monthly.items():
        annual.setdefault(int(mk[:4]), []).append(price)
    return {y: round(sum(v) / len(v), 2) for y, v in annual.items()}


def build_game_index(dataset, pred_df, mode):
    # build igdb_id: title lookup so rerelease ids can be resolved to names
    id_to_title = {}
    for _, r in dataset.iterrows():
        iid = safe_int(r.get("igdb_id"))
        if iid is not None:
            id_to_title[iid] = str(r.get("title", ""))

    pred_by_id = {}
    if not pred_df.empty:
        group_cols = ["prediction_date", "prediction", "lower_bound", "upper_bound", "confidence_pct"]
        if mode == "eval" and "actual_price" in pred_df.columns:
            group_cols.append("actual_price")
        for igdb_id, grp in pred_df.groupby("igdb_id"):
            if "months_ahead" in grp.columns:
                grp = grp.sort_values("months_ahead")
            rows = []
            for _, r in grp.iterrows():
                date_val = str(r.get("prediction_date", r.get("year", "")))[:7]
                entry = [
                    date_val,
                    safe_float(r.get("prediction")),
                    safe_float(r.get("lower_bound")),
                    safe_float(r.get("upper_bound")),
                    safe_float(r.get("confidence_pct"), 1),
                ]
                if mode == "eval":
                    entry.append(safe_float(r.get("actual_price")))
                rows.append(entry)
            pred_by_id[igdb_id] = rows

    games = []
    for _, row in dataset.iterrows():
        igdb_id  = row.get("igdb_id")
        hist_m   = parse_history_monthly(row.get("price_history_json", ""))
        hist_a   = to_annual(hist_m)
        preds    = pred_by_id.get(igdb_id, [])
        int_id   = safe_int(igdb_id)

        franchise_raw = row.get("franchise_names", "")
        franchise = str(franchise_raw) if pd.notna(franchise_raw) and str(franchise_raw).strip() else ""

        rr_ids_raw = row.get("rerelease_igdb_ids", "")
        rr_names = []
        if pd.notna(rr_ids_raw) and str(rr_ids_raw).strip():
            for rid_str in str(rr_ids_raw).split("|"):
                rid = safe_int(rid_str)
                if rid is not None and rid in id_to_title:
                    rr_names.append(id_to_title[rid])

        games.append({
            "id":        int_id,
            "title":     str(row.get("title", "")),
            "console":   str(row.get("console", "")),
            "genre":     str(row.get("genre", "")) if pd.notna(row.get("genre")) else "",
            "pub":       str(row.get("publisher", "")) if pd.notna(row.get("publisher")) else "",
            "year":      safe_int(row.get("release_year")),
            "esrb":      str(row.get("esrb", "")) if pd.notna(row.get("esrb")) else "",
            "franc":     franchise,
            "critic":    safe_float(row.get("igdb_critic_score"), 1),
            "user":      safe_float(row.get("igdb_user_score"), 1),
            "units":     safe_float(row.get("units_sold_na_m"), 2),
            "cond":      str(row.get("canonical_condition", "")) if pd.notna(row.get("canonical_condition")) else "",
            "records":   safe_int(row.get("price_record_count")),
            "rerel":     bool(row.get("rerelease_exists", False)),
            "rerel_yr":  safe_int(row.get("rerelease_year")),
            "rerel_tp":  str(row.get("rerelease_type", "")) if pd.notna(row.get("rerelease_type")) else "",
            "rerel_names": rr_names,
            "hm":        hist_m,
            "ha":        hist_a,
            "preds":     preds,
        })
    return games


def build_html(games, mode):
    is_eval    = mode == "eval"
    title_str  = "Eval Backtest 2024–2025" if is_eval else "Production Forecast 2025–2030"
    pred_label = "Eval Prediction (2024–2025)" if is_eval else "Forecast (2025–2030)"
    table_header = (
        "<th>Month</th><th>Lower</th><th>Prediction</th><th>Upper</th>"
        "<th>Actual</th><th>Error</th><th>Confidence</th>"
        if is_eval else
        "<th>Month</th><th>Lower</th><th>Prediction</th><th>Upper</th><th>Confidence</th>"
    )
    games_json = json.dumps(games, ensure_ascii=False, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Nintendo Price Lookup — {title_str}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:monospace;font-size:13px;height:100vh;overflow:hidden}}
header{{background:#222;color:#fff;padding:8px 12px;display:flex;align-items:center;gap:12px}}
#search{{padding:4px 8px;font-size:13px;font-family:monospace;border:1px solid #ccc;width:320px;color:#000;background:#fff}}
#sw{{position:relative}}
#sugg{{position:absolute;top:100%;left:0;width:400px;background:#fff;border:1px solid #999;max-height:260px;overflow-y:auto;z-index:200;display:none}}
.si{{padding:5px 8px;cursor:pointer;border-bottom:1px solid #eee;display:flex;justify-content:space-between;color:#000;background:#fff}}
.si:hover{{background:#eee}}
.sc{{color:#666;font-size:11px}}
.layout{{display:flex;height:calc(100vh - 36px)}}
#sidebar{{width:240px;min-width:240px;border-right:1px solid #ccc;overflow-y:auto;padding:10px;font-size:12px}}
#ph{{color:#999;padding-top:30px}}
#meta{{display:none}}
#meta h2{{font-size:13px;font-weight:bold;margin-bottom:4px}}
.mr{{display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #eee}}
.ml{{color:#666}}
#main{{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}}
#ctrl{{padding:6px 10px;border-bottom:1px solid #ccc;display:flex;gap:8px;align-items:center;background:#f5f5f5}}
#ctrl button{{padding:3px 10px;font-size:12px;font-family:monospace;cursor:pointer;border:1px solid #999;background:#fff}}
#ctrl button.active{{background:#222;color:#fff}}
#ca{{flex:1;min-height:0}}
#chart{{width:100%;height:100%}}
#ta{{height:200px;overflow-y:auto;border-top:1px solid #ccc;flex-shrink:0}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-family:monospace}}
th{{position:sticky;top:0;background:#eee;padding:4px 8px;text-align:left;border-bottom:1px solid #ccc}}
td{{padding:3px 8px;border-bottom:1px solid #eee}}
.nd{{color:#999;padding:16px}}
</style>
</head>
<body>
<header>
  <strong>Nintendo Price Lookup</strong> &nbsp;[{title_str}]
  <div id="sw">
    <input id="search" type="text" placeholder="Search game title…" autocomplete="off"/>
    <div id="sugg"></div>
  </div>
</header>
<div class="layout">
  <div id="sidebar">
    <div id="ph">Search for a game to view its data.</div>
    <div id="meta"></div>
  </div>
  <div id="main">
    <div id="ctrl">
      Granularity:
      <button id="bm" class="active" onclick="sv('monthly')">Monthly</button>
      <button id="ba" onclick="sv('annual')">Annual</button>
      &nbsp; Show:
      <button id="ball" class="active" onclick="sl('all')">History + Forecast</button>
      <button id="bh" onclick="sl('hist')">History Only</button>
      <button id="bp" onclick="sl('pred')">Forecast Only</button>
    </div>
    <div id="ca"><div id="chart"></div></div>
    <div id="ta"><div class="nd">Select a game to view predictions.</div></div>
  </div>
</div>
<script>
const GAMES={games_json};
const IS_EVAL={str(is_eval).lower()};
const PRED_LABEL="{pred_label}";

const IDX=GAMES.map((g,i)=>({{i,k:(g.title+' '+g.console).toLowerCase(),t:g.title,c:g.console}}));
let CG=null,CV='monthly',CL='all';

const se=document.getElementById('search');
const sg=document.getElementById('sugg');

se.addEventListener('input',()=>{{
  const q=se.value.trim().toLowerCase();
  if(q.length<2){{sg.style.display='none';return;}}
  const res=IDX.filter(x=>x.k.includes(q)).slice(0,30);
  if(!res.length){{sg.style.display='none';return;}}
  sg.innerHTML=res.map(r=>`<div class="si" onclick="pick(${{r.i}})"><span>${{esc(r.t)}}</span><span class="sc">${{esc(r.c)}}</span></div>`).join('');
  sg.style.display='block';
}});
document.addEventListener('click',e=>{{if(!e.target.closest('#sw'))sg.style.display='none';}});

function pick(i){{
  CG=GAMES[i];
  se.value=CG.title;
  sg.style.display='none';
  renderMeta();
  renderChart();
  renderTable();
}}

function sv(v){{
  CV=v;
  ['bm','ba'].forEach((id,i)=>document.getElementById(id).classList.toggle('active',['monthly','annual'][i]===v));
  renderChart();
}}

function sl(l){{
  CL=l;
  ['ball','bh','bp'].forEach((id,i)=>document.getElementById(id).classList.toggle('active',['all','hist','pred'][i]===l));
  renderChart();
}}

function renderMeta(){{
  const g=CG;
  document.getElementById('ph').style.display='none';
  const el=document.getElementById('meta');
  el.style.display='block';
  const f=(v)=>v!=null&&v!==''?v:'—';
  const fd=(v)=>v!=null?'$'+v.toFixed(2):'—';
  const mk=Object.keys(g.hm||{{}}).sort();
  const latest=mk.length?g.hm[mk[mk.length-1]]:null;
  // plan: add google trends search interest overlay on chart
  const condLabel=g.cond?g.cond.toUpperCase():'';
  el.innerHTML=`
    <h2>${{esc(g.title)}}</h2>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
      <span style="color:#666">${{esc(g.console)}}</span>
      ${{condLabel?`<span style="background:#222;color:#fff;font-size:10px;padding:1px 5px;border-radius:3px">${{condLabel}}</span>`:''}}
    </div>
    ${{row('Release Year',f(g.year))}}
    ${{row('Franchise',f(g.franc))}}
    ${{row('Genre',f(g.genre))}}
    ${{row('Publisher',f(g.pub))}}
    ${{row('ESRB',f(g.esrb))}}
    ${{row('Latest Price',fd(latest))}}
    ${{row('Critic Score',f(g.critic))}}
    ${{row('User Score',f(g.user))}}
    ${{row('NA Units Sold',g.units!=null?g.units.toFixed(2)+'M':'—')}}
    ${{row('Price Records',f(g.records))}}
    ${{row('Re-release',g.rerel?'Yes':'No')}}
    ${{row('Predictions',f(g.preds.length))}}
  `;
}}

function row(l,v){{return `<div class="mr"><span class="ml">${{l}}</span><span class="mv">${{v}}</span></div>`;}}

function renderChart(){{
  if(!CG)return;
  const g=CG;
  const traces=[];

  if(CL!=='pred'){{
    if(CV==='monthly'){{
      const ks=Object.keys(g.hm).sort();
      if(ks.length)traces.push({{
        x:ks.map(k=>k+'-01'),y:ks.map(k=>g.hm[k]),
        name:'Price History',type:'scatter',mode:'lines',
        line:{{color:'#1d4ed8',width:2}},yaxis:'y',
        hovertemplate:'%{{x|%b %Y}}: $%{{y:.2f}}<extra></extra>'
      }});
    }}else{{
      const yrs=Object.keys(g.ha).map(Number).sort();
      if(yrs.length)traces.push({{
        x:yrs.map(y=>y+'-07-01'),y:yrs.map(y=>g.ha[y]),
        name:'Price History',type:'scatter',mode:'lines+markers',
        line:{{color:'#1d4ed8',width:2}},marker:{{size:6}},yaxis:'y',
        hovertemplate:'%{{x|%Y}}: $%{{y:.2f}}<extra></extra>'
      }});
    }}

  }}

  if(CL!=='hist'&&g.preds.length){{
    const rows=CV==='annual'?aggAnnual(g.preds):g.preds;
    const dates=rows.map(r=>CV==='annual'?r.yr+'-07-01':r[0]+'-01');
    const lo=rows.map(r=>CV==='annual'?r.lo:r[2]);
    const hi=rows.map(r=>CV==='annual'?r.hi:r[3]);
    const mid=rows.map(r=>CV==='annual'?r.mid:r[1]);

    if(!IS_EVAL){{
      let lx=null,ly=null;
      if(CV==='monthly'){{const ks=Object.keys(g.hm).sort();if(ks.length){{lx=ks[ks.length-1]+'-01';ly=g.hm[ks[ks.length-1]];}}}}
      else{{const yrs=Object.keys(g.ha).map(Number).sort();if(yrs.length){{lx=yrs[yrs.length-1]+'-07-01';ly=g.ha[yrs[yrs.length-1]];}}}}
      if(lx!=null&&ly!=null){{dates.unshift(lx);mid.unshift(ly);lo.unshift(ly);hi.unshift(ly);}}
    }}

    traces.push({{
      x:[...dates,...[...dates].reverse()],
      y:[...hi,...[...lo].reverse()],
      fill:'toself',fillcolor:'rgba(220,38,38,0.1)',
      line:{{width:0}},showlegend:false,hoverinfo:'skip',yaxis:'y'
    }});
    traces.push({{
      x:dates,y:mid,name:PRED_LABEL,type:'scatter',mode:'lines',
      line:{{color:'#dc2626',width:2}},yaxis:'y',
      hovertemplate:'Forecast: $%{{y:.2f}}<extra></extra>'
    }});

  }}

  const shapes=[], annotations=[];
  if(g.rerel&&g.rerel_yr){{
    const rx=g.rerel_yr+'-07-01';
    const label=(g.rerel_names&&g.rerel_names.length?g.rerel_names[0]:(g.rerel_tp?g.rerel_tp.charAt(0).toUpperCase()+g.rerel_tp.slice(1):'Re-release'))
                +' ('+g.rerel_yr+')';
    shapes.push({{type:'line',x0:rx,x1:rx,y0:0,y1:1,yref:'paper',
      line:{{color:'#64748b',width:1.5,dash:'dot'}}}});
    annotations.push({{x:rx,y:0.98,yref:'paper',text:label,showarrow:false,
      textangle:-90,xanchor:'left',yanchor:'top',
      font:{{size:10,color:'#64748b'}}}});
  }}

  const layout={{
    margin:{{l:60,r:16,t:24,b:48}},
    paper_bgcolor:'white',plot_bgcolor:'#fafafa',
    xaxis:{{type:'date',gridcolor:'#f1f5f9',tickfont:{{size:11}}}},
    yaxis:{{gridcolor:'#f1f5f9',tickformat:'$,.0f',tickfont:{{size:11}},
      title:{{text:'Price (USD)',font:{{size:11}}}}}},
    legend:{{orientation:'h',y:1.08,font:{{size:11}}}},
    hovermode:'x unified',
    shapes,annotations
  }};
  Plotly.react('chart',traces,layout,{{responsive:true,displayModeBar:false}});
}}

function aggAnnual(preds){{
  const by={{}};
  preds.forEach(r=>{{
    const yr=parseInt(r[0]);
    if(!by[yr])by[yr]={{mid:[],lo:[],hi:[],act:[]}};
    if(r[1]!=null)by[yr].mid.push(r[1]);
    if(r[2]!=null)by[yr].lo.push(r[2]);
    if(r[3]!=null)by[yr].hi.push(r[3]);
    if(IS_EVAL&&r[5]!=null)by[yr].act.push(r[5]);
  }});
  return Object.keys(by).sort().map(yr=>{{
    const b=by[yr];
    return {{yr:parseInt(yr),mid:avg(b.mid),lo:avg(b.lo),hi:avg(b.hi),
             act:b.act.length?avg(b.act):null}};
  }});
}}
function avg(a){{return a.length?Math.round(a.reduce((x,y)=>x+y,0)/a.length*100)/100:null;}}

function renderTable(){{
  const el=document.getElementById('ta');
  if(!CG||!CG.preds.length){{el.innerHTML='<div class="nd">No predictions available.</div>';return;}}
  const cc=c=>c>=70?'ch':c>=40?'cm':'cl2';
  let h=`<table><thead><tr>{table_header}</tr></thead><tbody>`;
  CG.preds.forEach(r=>{{
    const [date,mid,lo,hi,conf,...rest]=r;
    const act=IS_EVAL?rest[0]:null;
    const err=IS_EVAL&&act!=null&&mid!=null?((mid-act)/act*100).toFixed(1)+'%':'—';
    const ec=IS_EVAL&&act!=null&&mid!=null?(Math.abs((mid-act)/act)<=0.2?'#16a34a':'#dc2626'):'';
    h+=`<tr>
      <td>${{fmtM(date)}}</td>
      <td>${{lo!=null?'$'+lo.toFixed(2):'—'}}</td>
      <td><strong>${{mid!=null?'$'+mid.toFixed(2):'—'}}</strong></td>
      <td>${{hi!=null?'$'+hi.toFixed(2):'—'}}</td>
      ${{IS_EVAL?`<td>${{act!=null?'$'+act.toFixed(2):'—'}}</td><td style="color:${{ec}}">${{err}}</td>`:''}}
      <td class="${{cc(conf)}}">${{conf!=null?conf.toFixed(1)+'%':'—'}}</td>
    </tr>`;
  }});
  el.innerHTML=h+'</tbody></table>';
}}

function fmtM(s){{
  if(!s)return'—';
  const [y,m]=s.split('-');
  return ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][parseInt(m)-1]+' '+y;
}}
function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
</script>
</body>
</html>"""


def main():
    dataset   = pd.read_csv(DATA_DIR / "merged_dataset.csv")
    eval_path = DATA_DIR / "eval_predictions_2023.csv"
    prod_path = DATA_DIR / "predictions.csv"

    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    prod_df = pd.read_csv(prod_path) if prod_path.exists() else pd.DataFrame()

    eval_games = build_game_index(dataset, eval_df, "eval")
    prod_games = build_game_index(dataset, prod_df, "prod")

    eval_html = build_html(eval_games, "eval")
    prod_html = build_html(prod_games, "prod")

    (DATA_DIR / "lookup_eval.html").write_text(eval_html, encoding="utf-8")
    print(f"Saved to {DATA_DIR / 'lookup_eval.html'}")

    (DATA_DIR / "lookup_prod.html").write_text(prod_html, encoding="utf-8")
    print(f"Saved to {DATA_DIR / 'lookup_prod.html'}")


if __name__ == "__main__":
    main()
