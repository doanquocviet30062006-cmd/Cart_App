const fieldLabels = {
  code_module: "Học phần (mã ẩn danh)", code_presentation: "Kỳ mở lớp",
  gender: "Giới tính (danh mục OULAD)", highest_education: "Trình độ học vấn đầu vào",
  age_band: "Nhóm tuổi", num_of_prev_attempts: "Số lần học phần trước đây",
  studied_credits: "Tổng tín chỉ đang học", clicks_day_0_28: "Lượt tương tác VLE (ngày 0–28)",
  active_days_day_0_28: "Số ngày hoạt động (ngày 0–28)"
};
const descriptions = {
  clicks_day_0_28: "Tổng sum_click đến hết ngày 28", active_days_day_0_28: "Số ngày lịch có hoạt động, tối đa 29"
};
let schema, metrics;
const $ = (s) => document.querySelector(s);
const fmt = (v, digits=3) => Number(v).toLocaleString("vi-VN", {maximumFractionDigits:digits});

async function boot() {
  const [s,m,h] = await Promise.all([fetch('/api/schema'),fetch('/api/metrics'),fetch('/api/health')]);
  schema = await s.json(); metrics = await m.json();
  const health = await h.json(); $('#health').textContent = `${health.status} · ${health.model_version}`;
  renderFields(schema); renderOverview(metrics); renderEvaluation(metrics); renderDataset(metrics);
}
function go(page) {
  document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id===`page-${page}`));
  document.querySelectorAll('.nav').forEach(x=>x.classList.toggle('active',x.dataset.page===page));
  window.scrollTo({top:0,behavior:'smooth'});
}
document.querySelectorAll('.nav').forEach(btn=>btn.addEventListener('click',()=>go(btn.dataset.page)));
document.querySelectorAll('[data-go]').forEach(btn=>btn.addEventListener('click',()=>go(btn.dataset.go)));
function renderFields(s) {
  const form=$('#fields'); form.replaceChildren();
  for (const key of s.feature_order) {
    const wrap=document.createElement('div'); wrap.className='field';
    const label=document.createElement('label'); label.htmlFor=`f-${key}`; label.textContent=fieldLabels[key]||key;
    let input;
    if (s.categories[key]) {
      input=document.createElement('select');
      for(const value of s.categories[key]) {const option=document.createElement('option'); option.value=value; option.textContent=value; input.append(option);}
    } else {
      input=document.createElement('input'); input.type='number'; input.step=s.numeric_ranges[key].integer?'1':'any';
      input.min=s.numeric_ranges[key].min; input.max=s.numeric_ranges[key].max;
    }
    input.id=`f-${key}`; input.name=key; input.required=true;
    const small=document.createElement('small');
    small.textContent=descriptions[key] || (s.categories[key] ? 'Giá trị theo các danh mục có trong dữ liệu huấn luyện' : `Phạm vi quan sát train: ${s.numeric_ranges[key].min} – ${s.numeric_ranges[key].max}`);
    wrap.append(label,input,small); form.append(wrap);
  }
  loadExample();
}
function loadExample(){if(!schema)return; for(const [key,value] of Object.entries(schema.example)){const el=$(`#f-${key}`);if(el)el.value=value;}}
$('#reset-example').addEventListener('click',loadExample);
$('#predict-form').addEventListener('submit',async (event)=>{
  event.preventDefault(); $('#form-error').textContent='';
  const payload={};
  for(const key of schema.feature_order){const el=$(`#f-${key}`);payload[key]=schema.categories[key]?el.value:Number(el.value);}
  try {
    const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const result=await res.json(); if(!res.ok)throw Error(result.error||'Không thể tạo dự báo.');
    renderPrediction(result); $('#empty-result').hidden=true; $('#prediction-result').hidden=false;
  } catch(err){$('#form-error').textContent=err.message;}
});
function renderPrediction(r){
  const box=$('#prediction-result'); box.replaceChildren();
  const banner=document.createElement('div'); banner.className=`risk-banner${r.prediction==='Withdrawn'?' risk':''}`;
  banner.innerHTML=`<span>Nhãn cây CART</span><strong>${r.prediction}</strong><div class="risk-score">Tỷ lệ lớp Withdrawn trong nút lá: ${fmt(r.leaf_withdrawn_fraction*100,1)}% · ${r.score_note}</div>`; box.append(banner);
  const title=document.createElement('h3'); title.textContent='Đường đi từ nút gốc đến nút lá'; box.append(title);
  const path=document.createElement('div'); path.className='path';
  r.decision_path.forEach((step,index)=>{
    const card=document.createElement('div'); card.className=`path-step${step.leaf?' leaf':''}`;
    const b=document.createElement('b');
    const small=document.createElement('small');
    if(step.leaf){b.textContent=`Nút lá ${step.node} · ${step.prediction}`; small.textContent=`${step.samples} mẫu huấn luyện trong lá · Withdrawn ${fmt(step.class_fraction.withdrawn*100,1)}% · trạng thái còn lại ${fmt(step.class_fraction.not_withdrawn*100,1)}%`;}
    else {b.textContent=`Nút ${step.node} · ${step.feature}`; small.textContent=`Kiểm tra ${step.condition} → ${step.condition_result?'Đúng':'Sai'}, đi nhánh ${step.branch} sang nút ${step.next_node}`;}
    card.append(b,small);path.append(card);
  });box.append(path);
}
function renderOverview(m){
  const n=m.data_counts.eligible_enrollments, rate=m.data_counts.positive_withdrawn/n;
  $('#overview-stats').innerHTML=[['ENROLLMENTS',n.toLocaleString('vi-VN')],['WITHDRAWN LABEL',`${fmt(rate*100,1)}%`],['PREDICTION CUTOFF','Day 28'],['TREE DEPTH',m.pruned_tree.depth]].map(([a,b])=>`<div class="stat"><b>${b}</b><span>${a}</span></div>`).join('');
}
function renderEvaluation(m){
  const r=m.pruned_tree_test, items=[['Accuracy',r.accuracy],['Precision · Withdrawn',r.precision_withdrawn],['Recall · Withdrawn',r.recall_withdrawn],['F1 · Withdrawn',r.f1_withdrawn]];
  $('#eval-summary').innerHTML=items.map(([a,b])=>`<div class="metric"><strong>${fmt(b*100,1)}%</strong><span>${a}</span></div>`).join('');
  const cm=r.confusion_matrix;
  $('#confusion').innerHTML=`<div class="head"></div><div class="head">Pred 0</div><div class="head">Pred 1</div><div class="head">Actual 0</div><div>${cm[0][0].toLocaleString()}</div><div class="miss">${cm[0][1].toLocaleString()}</div><div class="head">Actual 1</div><div class="miss">${cm[1][0].toLocaleString()}</div><div class="positive">${cm[1][1].toLocaleString()}</div>`;
  const base=m.baseline_test, un=m.unpruned_tree_test, tree=m.pruned_tree;
  const rows=[['Majority baseline','—',fmt(base.accuracy*100,1)+'%','F1 '+fmt(base.f1_withdrawn)],['CART · unpruned',`depth ${m.unpruned_tree.depth} · ${m.unpruned_tree.leaves} leaves`,fmt(un.accuracy*100,1)+'%','F1 '+fmt(un.f1_withdrawn)],['CART · pruned',`depth ${tree.depth} · ${tree.leaves} leaves`,fmt(r.accuracy*100,1)+'%','F1 '+fmt(r.f1_withdrawn)]];
  $('#model-compare').innerHTML=rows.map(x=>`<div class="compare-row"><span>${x[0]}</span><b>${x[1]}<br>${x[2]} · ${x[3]}</b></div>`).join('');
  $('#cv-summary').textContent=`${m.cv.method}, ${m.cv.folds} folds grouped by student; selection metric: ${m.cv.selection_metric}. Chọn cây đơn giản nhất trong khoảng một độ lệch chuẩn so với F1 CV tốt nhất (${fmt(m.cv.best_mean_f1)}; ngưỡng ${fmt(m.cv.one_standard_error_floor)}). Chosen ccp_alpha = ${fmt(tree.ccp_alpha,6)}. Cross-validation is confined to training students.`;
  $('#alpha-list').innerHTML=m.cv.candidates.map(c=>`<span class="alpha-chip${c.alpha===tree.ccp_alpha?' best':''}">α ${fmt(c.alpha,5)} · F1 ${fmt(c.mean_cv_f1_withdrawn,3)}</span>`).join('');
}
function renderDataset(m){
  $('#feature-chips').innerHTML=m.features.map(f=>`<span class="chip">${fieldLabels[f]||f}</span>`).join('');
  $('#limitations').innerHTML=m.limitations.map(x=>`<li>${x}</li>`).join('');
}
boot().catch(e=>{$('#health').textContent='Không tải được model';console.error(e);});
