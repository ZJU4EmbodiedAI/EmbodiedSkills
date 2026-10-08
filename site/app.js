import { cases, trajectories } from './evidence.js';

const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let motionPaused = reducedMotion.matches;
let pageVisible = !document.hidden;
let heroVisible = true;
let loopVisible = false;
let loopRunning = !reducedMotion.matches;
let loopStage = 0;
let loopTimer;
let canvasFrame;
let playbackTimer;
let replayPlaying = false;
let activeTrajectory = trajectories[0];
let frameIndex = 0;
let toastTimer;

function toast(message) {
  $('.toast').textContent = message;
  $('.toast').classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('.toast').classList.remove('visible'), 2500);
}
function refreshEntrance(element) {
  if (motionPaused) return;
  element.classList.remove('switch-enter');
  void element.offsetWidth;
  element.classList.add('switch-enter');
}
function selectTab(buttons, active) {
  buttons.forEach(button => {
    const selected = button === active;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
}
function keyboardTabs(buttons) {
  buttons.forEach((button, index) => button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % buttons.length;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + buttons.length) % buttons.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = buttons.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      buttons[next].click();
      buttons[next].focus();
    }
  }));
}

// Reveal once; continuous effects are paused when their sections leave the screen.
const revealObserver = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    entry.target.classList.add('visible');
    revealObserver.unobserve(entry.target);
  }
}, { threshold: 0.06, rootMargin: '0px 0px -25px 0px' });
$$('.reveal').forEach(element => revealObserver.observe(element));
$$('.principles, .learning-steps, .prediction-notes').forEach(group => {
  $$('.reveal', group).forEach((element, index) => element.style.transitionDelay = `${index * 90}ms`);
});

const header = $('.site-header');
const menuButton = $('.menu-toggle');
function closeMenu() {
  header.classList.remove('menu-open');
  menuButton.setAttribute('aria-expanded', 'false');
  menuButton.setAttribute('aria-label', 'Open menu');
}
menuButton.addEventListener('click', () => {
  const open = header.classList.toggle('menu-open');
  menuButton.setAttribute('aria-expanded', String(open));
  menuButton.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
});
$$('#main-nav a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });
document.addEventListener('click', event => { if (!header.contains(event.target)) closeMenu(); });
let scrollPending = false;
const navSections = ['method', 'evidence', 'results', 'research'].map(id => document.getElementById(id));
function onScroll() {
  const y = scrollY;
  header.classList.toggle('scrolled', y > 90);
  const max = document.documentElement.scrollHeight - innerHeight;
  $('.reading-progress').style.width = `${max ? y / max * 100 : 0}%`;
  let current = '';
  for (const section of navSections) if (section.getBoundingClientRect().top < innerHeight * .45) current = section.id;
  $$('#main-nav a').forEach(link => link.classList.toggle('current', link.hash === `#${current}`));
  scrollPending = false;
}
addEventListener('scroll', () => {
  if (!scrollPending) { scrollPending = true; requestAnimationFrame(onScroll); }
}, { passive: true });
onScroll();

// Lightweight canvas illustration: no simulated rollout is presented as a prediction.
const canvas = $('#hero-canvas');
const context = canvas.getContext('2d');
let canvasWidth = 0, canvasHeight = 0;
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvasWidth = rect.width; canvasHeight = rect.height;
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (motionPaused) drawCanvas(0);
}
function bezier(t, a, b, c, d) {
  const n = 1 - t;
  return n*n*n*a + 3*n*n*t*b + 3*n*t*t*c + t*t*t*d;
}
function drawCanvas(time) {
  const w = canvasWidth, h = canvasHeight;
  context.clearRect(0, 0, w, h);
  const start = { x: w * .72, y: h * .71 };
  const colors = ['#a0c7e4', '#a0c7e4', '#dcaa91', '#9dc6ad', '#a0c7e4'];
  for (let i = 0; i < 5; i++) {
    const endX = w * (.57 + i * .087);
    const endY = h * (.40 - Math.sin(i * .8) * .13);
    const bx = start.x + (i-2)*w*.025;
    const by = h*.46;
    const cx = endX + (i-2)*w*.015;
    const cy = h*.43;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.bezierCurveTo(bx, by, cx, cy, endX, endY);
    context.strokeStyle = colors[i]; context.globalAlpha = i === 3 ? .40 : .16;
    context.lineWidth = i === 3 ? 1.25 : .75;
    context.setLineDash(i === 3 ? [] : [3, 6]); context.stroke(); context.setLineDash([]);
    const progress = (time / 5200 + i * .18) % 1;
    const x = bezier(progress, start.x, bx, cx, endX), y = bezier(progress, start.y, by, cy, endY);
    context.globalAlpha = i === 3 ? .85 : .5;
    context.fillStyle = colors[i]; context.beginPath(); context.arc(x,y,i===3?3:1.8,0,Math.PI*2); context.fill();
    context.globalAlpha = .4;
    context.strokeRect(endX-4, endY-4, 8, 8);
  }
  context.globalAlpha = .11; context.strokeStyle = '#a0c7e4'; context.lineWidth = 1;
  context.beginPath(); context.ellipse(w*.79,h*.66,w*.19,h*.1,-.15,0,Math.PI*2); context.stroke();
  context.globalAlpha = 1;
}
function canvasTick(time) {
  canvasFrame = null;
  if (motionPaused || !pageVisible || !heroVisible) return;
  drawCanvas(time);
  canvasFrame = requestAnimationFrame(canvasTick);
}
function syncCanvas() {
  if (canvasFrame) cancelAnimationFrame(canvasFrame);
  canvasFrame = null;
  if (!motionPaused && pageVisible && heroVisible) canvasFrame = requestAnimationFrame(canvasTick);
  else drawCanvas(0);
}
new ResizeObserver(resizeCanvas).observe(canvas);
new IntersectionObserver(([entry]) => { heroVisible = entry.isIntersecting; syncCanvas(); }).observe($('#top'));
$('#top').addEventListener('pointermove', event => {
  if (motionPaused || event.pointerType === 'touch' || innerWidth < 760) return;
  const x = (event.clientX / innerWidth - .5) * 12;
  const y = (event.clientY / innerHeight - .5) * 9;
  $('.hero-photo').style.setProperty('--px', `${x}px`);
  $('.hero-photo').style.setProperty('--py', `${y}px`);
}, { passive: true });

const stages = [
  { name:'Observe', title:'Make the present<br>machine-readable.', copy:'Frozen visual and language encoders turn the current scene and instruction into a shared context, alongside observable state and elapsed progress.', skills:'Scene observation · Context encoding', output:'Visual tokens + goal embedding + state', label:'CURRENT OBSERVATION', caption:'Scene + instruction + state', color:'#bf9742', image:'assets/robot-before.webp' },
  { name:'Propose', title:'Keep the policy.<br>Explore its possibilities.', copy:'The frozen policy produces five candidate actions. Its direct proposal remains candidate zero; alternatives retain the backend’s native action representation.', skills:'Candidate generation · Action packaging', output:'Five executable proposals, including the direct action', label:'POLICY PROPOSALS', caption:'A₀ · A₁ · A₂ · A₃ · A₄', color:'#367db7', image:'assets/robot-before.webp' },
  { name:'Predict', title:'Anticipate what<br>each action changes.', copy:'Three independently trained world action models estimate action-conditioned latent effects and outcomes. They share frozen pretrained visual and language features.', skills:'Latent consequence prediction · Outcome evaluation', output:'Predicted effects + three outcome estimates', label:'LATENT CONSEQUENCES', caption:'Three independent views of each action', color:'#b86c50', image:'assets/robot-before.webp' },
  { name:'Select', title:'Compare the effect.<br>Choose the action.', copy:'The comparison head uses predicted consequences and agreement between models to estimate the benefit and risk of replacing the direct proposal. The highest-scoring original candidate is selected.', skills:'Consequence comparison · Candidate selection', output:'One candidate identity, preserved for execution', label:'COMPARATIVE DECISION', caption:'Expected outcome + consequence agreement', color:'#478967', image:'assets/robot-before.webp' },
  { name:'Execute', title:'Let the world<br>answer back.', copy:'The backend executes the selected action or bounded action chunk. Its new observation closes the loop and supplies the starting point for the next decision.', skills:'Native action execution · Observation refresh', output:'Executed action → fresh observation', label:'NEXT OBSERVATION', caption:'The consequence becomes the next input', color:'#367db7', image:'assets/robot-after.webp' }
];
const stageButtons = $$('.stage-node');
function showStage(index) {
  loopStage = index;
  const stage = stages[index];
  selectTab(stageButtons, stageButtons[index]);
  $('.loop-visual').dataset.stage = index;
  $('.loop-visual').style.setProperty('--stage-color', stage.color);
  $('#stage-description').setAttribute('aria-labelledby', `stage-${index}`);
  $('#stage-number').textContent = `0${index+1} / ${stage.name.toUpperCase()}`;
  $('#stage-title').innerHTML = stage.title;
  $('#stage-copy').textContent = stage.copy;
  $('#stage-skills').textContent = stage.skills;
  $('#stage-output').textContent = stage.output;
  $('#loop-core-label').textContent = stage.label;
  $('#loop-core-caption').textContent = stage.caption;
  $('#loop-image').src = stage.image;
  $('#loop-image').alt = `Recorded robot observation in an illustration of the ${stage.name.toLowerCase()} stage`;
  $('.stage-progress span').style.width = `${(index + 1) * 20}%`;
  refreshEntrance($('#stage-description'));
}
function syncLoop() {
  clearInterval(loopTimer);
  const running = loopRunning && !motionPaused;
  $('#loop-play').textContent = running ? 'Pause loop Ⅱ' : 'Play loop ▷';
  $('#loop-play').setAttribute('aria-pressed', String(running));
  if (running && loopVisible && pageVisible) loopTimer = setInterval(() => {
    if (!$('.stage-tabs').contains(document.activeElement)) showStage((loopStage + 1) % stages.length);
  }, 4500);
}
stageButtons.forEach(button => button.addEventListener('click', () => {
  showStage(Number(button.dataset.stage));
  loopRunning = false;
  syncLoop();
}));
keyboardTabs(stageButtons);
$('#loop-play').addEventListener('click', () => {
  if (motionPaused) { motionPaused = false; syncMotion(); }
  loopRunning = !loopRunning; syncLoop();
});
new IntersectionObserver(([entry]) => { loopVisible = entry.isIntersecting; syncLoop(); }, { threshold: .15 }).observe($('.loop-workspace'));
$$('[data-heatmap]').forEach(map => {
  const seed = Number(map.dataset.heatmap);
  for (let i=0; i<32; i++) {
    const cell = document.createElement('i');
    const value = .22 + ((Math.sin(i * 2.78 + seed * .84) + 1) / 2) * .75;
    cell.style.setProperty('--v', value.toFixed(3));
    cell.style.setProperty('--i', String(i));
    map.append(cell);
  }
});

let activeCase = cases[0];
function inspectCandidate(index) {
  $$('.candidate-button').forEach((button, i) => {
    button.classList.toggle('active', i === index);
    button.setAttribute('aria-pressed', String(i === index));
  });
  const tag = index === activeCase.selected ? 'MODEL SELECTION' : index === 0 ? 'DIRECT POLICY' : 'ALTERNATIVE';
  $('#branch-title').textContent = `CANDIDATE A${'₀₁₂₃₄'[index]} · ${tag}`;
  $('#branch-after').src = activeCase.images[index];
  $('#branch-after').alt = `${activeCase.title}: actual observation immediately after candidate A${index}`;
  const success = activeCase.outcomes[index] === 1;
  $('#branch-outcome').replaceChildren();
  const dot = document.createElement('span');
  dot.className = success ? 'success-dot' : 'failure-dot';
  $('#branch-outcome').append(dot, document.createTextNode(success ? ' Branch completes after frozen-policy continuation.' : ' Branch does not complete after frozen-policy continuation.'));
}
function showCase(id) {
  activeCase = cases.find(item => item.id === id);
  $$('[data-case]').forEach(button => {
    const active = button.dataset.case === id;
    button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
  const sentence = activeCase.instruction;
  $('#case-instruction').textContent = `“${sentence[0].toUpperCase()}${sentence.slice(1)}.”`;
  $('#branch-before').src = activeCase.before;
  $('#branch-before').alt = `Shared starting observation: ${activeCase.instruction}`;
  const container = $('.candidate-selector');
  container.replaceChildren();
  const maxScore = Math.max(...activeCase.scores.map(Math.abs), .001);
  activeCase.scores.forEach((score, index) => {
    const button = document.createElement('button');
    button.className = 'candidate-button';
    const role = index === activeCase.selected ? 'Selected' : index === 0 ? 'Direct' : 'Alternative';
    button.setAttribute('aria-label', `Inspect candidate A${index}, ${role.toLowerCase()}, score ${score.toFixed(3)}`);
    button.innerHTML = `<span class="candidate-header">A${'₀₁₂₃₄'[index]}<small>${role}</small></span><span class="candidate-chart"><i style="--w:${Math.max(2, Math.abs(score)/maxScore*100)}%"></i></span><span class="candidate-score"><span>${score>=0?'+':''}${score.toFixed(3)}</span><span>${activeCase.outcomes[index] ? 'Success' : 'Unfinished'}</span></span>`;
    if (score < 0) button.classList.add('negative-score');
    button.addEventListener('click', () => inspectCandidate(index));
    container.append(button);
  });
  inspectCandidate(activeCase.selected);
}
$$('[data-case]').forEach(button => button.addEventListener('click', () => showCase(button.dataset.case)));
showCase('shapes');

function setPlayback(playing) {
  replayPlaying = playing;
  clearInterval(playbackTimer);
  const button = $('#replay-play');
  button.textContent = playing ? 'Ⅱ' : '▶';
  button.setAttribute('aria-pressed', String(playing));
  button.setAttribute('aria-label', playing ? 'Pause recorded trajectory' : 'Play recorded trajectory');
  if (playing && pageVisible) playbackTimer = setInterval(() => {
    if (frameIndex >= activeTrajectory.frames.length - 1) { setPlayback(false); return; }
    showFrame(frameIndex+1);
    if (frameIndex === activeTrajectory.frames.length - 1) setPlayback(false);
  }, 680);
}
function showFrame(index) {
  frameIndex = index;
  const frame = activeTrajectory.frames[index];
  $('#replay-frame').src = frame.image;
  $('#replay-frame').alt = `${activeTrajectory.title}, recorded state at step ${frame.step}`;
  $('#frame-counter').textContent = `STEP ${String(frame.step).padStart(3,'0')} / ${String(activeTrajectory.steps).padStart(3,'0')}`;
  $('#replay-scrub').value = index;
  const ended = index === activeTrajectory.frames.length - 1;
  $('#replay-state').textContent = ended ? 'TASK SUCCESS' : index === 0 ? 'INITIAL OBSERVATION' : `AFTER CANDIDATE A${'₀₁₂₃₄'[frame.selected]}`;
  $('#replay-state').classList.toggle('complete', ended);
  $$('.decision-strip button').forEach((button, i) => {
    button.classList.toggle('active', i + 1 === index);
    button.classList.toggle('past', i + 1 < index);
    button.setAttribute('aria-pressed', String(i + 1 === index));
  });
}
function showTrajectory(id) {
  setPlayback(false);
  activeTrajectory = trajectories.find(item => item.id === id);
  $$('[data-trajectory]').forEach(button => {
    const active = button.dataset.trajectory === id;
    button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
  $('#trajectory-steps').textContent = activeTrajectory.steps;
  $('#trajectory-prompt').textContent = activeTrajectory.instruction;
  $('#replay-scrub').max = activeTrajectory.frames.length - 1;
  const strip = $('.decision-strip');
  strip.replaceChildren();
  activeTrajectory.frames.slice(1).forEach((frame,index) => {
    const button = document.createElement('button');
    button.textContent = `A${'₀₁₂₃₄'[frame.selected]}`;
    button.title = `Step ${frame.step}: candidate A${frame.selected}`;
    button.setAttribute('aria-label', button.title);
    button.addEventListener('click', () => { setPlayback(false); showFrame(index+1); });
    strip.append(button);
  });
  showFrame(0);
}
$$('[data-trajectory]').forEach(button => button.addEventListener('click', () => showTrajectory(button.dataset.trajectory)));
$('#replay-play').addEventListener('click', () => {
  if (!replayPlaying && frameIndex === activeTrajectory.frames.length - 1) showFrame(0);
  if (!replayPlaying) activeTrajectory.frames.forEach(frame => { const image = new Image(); image.src = frame.image; });
  setPlayback(!replayPlaying);
});
$('#replay-reset').addEventListener('click', () => { setPlayback(false); showFrame(0); });
$('#replay-scrub').addEventListener('input', event => { setPlayback(false); showFrame(Number(event.target.value)); });
new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) setPlayback(false); }).observe($('.replay-viewer'));
showTrajectory('basketball');

const benchmarks = {
  mt50: { value:78.64, direct:76.28, gain:2.36, metric:'COMPLETE-TASK SUCCESS', policy:'Direct π₀.₅', description:'50 manipulation tasks with a frozen π₀.₅ policy. The loop compares action chunks at each decision.', count:'2,500 evaluation episodes', note:'Same frozen policy backbone; success is measured on the complete task.' },
  cliport: { value:36.83, direct:33.94, gain:2.89, metric:'COMPLETE-TASK SUCCESS', policy:'Direct CLIPort', description:'Language-conditioned pick-and-place tasks. The selected first action is followed by the original frozen policy.', count:'1,800 evaluation instances', note:'Branch outcomes measure complete-task success after frozen-policy continuation.' },
  calvin: { value:77.90, direct:77.10, gain:.80, metric:'FIVE-TASK CHAIN SUCCESS · SR₅', policy:'Direct FLOWER', description:'Long-horizon language-conditioned manipulation with a frozen FLOWER policy. SR₅ measures completion of all five tasks.', count:'1,000 evaluation chains', note:'The metric here is five-task chain success (SR₅), reported in the paper’s CALVIN comparison.' }
};
let countFrame;
function showBenchmark(id, animate=true) {
  const data = benchmarks[id];
  selectTab($$('[data-benchmark]'), $(`[data-benchmark="${id}"]`));
  $('#benchmark-panel').setAttribute('aria-labelledby', `bench-${id}`);
  $('#result-metric').textContent = data.metric;
  $('#result-gain').textContent = `+${data.gain.toFixed(2)}`;
  $('#result-description').textContent = data.description;
  $('#result-count').textContent = data.count;
  $('#result-note').textContent = data.note;
  $('#direct-policy-name').textContent = data.policy;
  $('#direct-number').textContent = `${data.direct.toFixed(2)}%`;
  $('#ours-number').textContent = `${data.value.toFixed(2)}%`;
  $('#direct-bar').style.setProperty('--bar', `${data.direct}%`);
  $('#ours-bar').style.setProperty('--bar', `${data.value}%`);
  cancelAnimationFrame(countFrame);
  const initial = Number($('#result-value').textContent);
  if (!animate || motionPaused) $('#result-value').textContent = data.value.toFixed(2);
  else {
    const start = performance.now();
    const count = time => {
      const t = Math.min((time - start) / 600, 1);
      $('#result-value').textContent = (initial + (data.value-initial)*(1-Math.pow(1-t,3))).toFixed(2);
      if (t < 1) countFrame = requestAnimationFrame(count);
    };
    countFrame = requestAnimationFrame(count);
  }
}
$$('[data-benchmark]').forEach(button => button.addEventListener('click', () => showBenchmark(button.dataset.benchmark)));
keyboardTabs($$('[data-benchmark]'));

const dialog = $('#figure-dialog');
let figureTrigger;
$$('[data-figure]').forEach(button => button.addEventListener('click', () => {
  figureTrigger = button;
  $('#dialog-image').src = button.dataset.figure;
  $('#dialog-image').alt = button.dataset.caption;
  $('#figure-caption').textContent = button.dataset.caption;
  $('.dialog-image-wrap').classList.remove('zoomed');
  dialog.showModal(); document.body.style.overflow = 'hidden';
  $('#close-figure').focus();
}));
$('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX<rect.left || event.clientX>rect.right || event.clientY<rect.top || event.clientY>rect.bottom) dialog.close();
});
dialog.addEventListener('close', () => { document.body.style.overflow = ''; figureTrigger?.focus({ preventScroll:true }); });
$('.dialog-image-wrap').addEventListener('click', () => $('.dialog-image-wrap').classList.toggle('zoomed'));
$('#copy-citation').addEventListener('click', async () => {
  const content = $('#citation-text').textContent;
  try {
    if (navigator.clipboard && isSecureContext) await navigator.clipboard.writeText(content);
    else {
      const field = document.createElement('textarea');
      field.value = content; field.style.position = 'fixed'; field.style.opacity = '0';
      document.body.append(field); field.select();
      const copied = document.execCommand('copy'); field.remove();
      if (!copied) throw new Error('Clipboard unavailable');
    }
    toast('BibTeX copied to clipboard.');
  } catch { toast('Select the citation below to copy it.'); }
});

function syncMotion() {
  document.documentElement.classList.toggle('motion-paused', motionPaused);
  $('.motion-toggle').setAttribute('aria-pressed', String(motionPaused));
  const label = motionPaused ? 'Resume ambient animations' : 'Pause ambient animations';
  $('.motion-toggle').setAttribute('aria-label', label);
  $('.motion-toggle').title = label;
  $('.pause-icon').textContent = motionPaused ? '▷' : 'Ⅱ';
  syncCanvas(); syncLoop();
}
$('.motion-toggle').addEventListener('click', () => {
  motionPaused = !motionPaused;
  if (motionPaused) setPlayback(false);
  syncMotion();
});
reducedMotion.addEventListener('change', event => { motionPaused = event.matches; syncMotion(); });
document.addEventListener('visibilitychange', () => {
  pageVisible = !document.hidden;
  if (!pageVisible) setPlayback(false);
  syncCanvas(); syncLoop();
});
// Pause off-screen CSS animation timelines as well as JavaScript rendering.
const animationObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => entry.target.classList.toggle('offscreen', !entry.isIntersecting));
}, { rootMargin:'100px' });
$$('#prediction, #learning, #method').forEach(section => animationObserver.observe(section));
syncMotion();
