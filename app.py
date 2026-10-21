from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import MODEL_FILES, MODEL_DIR, SOURCE_MODELS, load_class_names
from src.model_loader import load_model
from src.predictor import predict
from src.metrics import read_metrics, METRIC_COLUMNS
from src.camera import capture_image

st.set_page_config(page_title="VANTA · Smart Waste Intelligence Lab", page_icon="♻", layout="wide", initial_sidebar_state="expanded")
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
/* Paleta: solo 3 colores + 1 gris -> tinta, papel, lima */
:root{--ink:#0E0E0E;--paper:#F4F1E8;--lime:#C6F432;--grey:#6B6A60;--white:#FFFFFF;--line:#0E0E0E}

/* ============ BASE: texto siempre oscuro sobre papel ============ */
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--paper)!important;color:var(--ink)!important;font-family:'DM Sans',sans-serif}
[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:1400px;padding-top:2rem;padding-bottom:4rem}
[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3,[data-testid="stMain"] h4{font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;letter-spacing:-.035em;color:var(--ink)!important}
[data-testid="stMain"] h1{font-size:clamp(2.2rem,4.5vw,3.6rem)!important;line-height:1!important;text-transform:uppercase}
[data-testid="stMain"] p,[data-testid="stMain"] li,[data-testid="stMain"] label,[data-testid="stMain"] [data-testid="stWidgetLabel"] *,[data-testid="stMain"] [data-testid="stRadio"] *{color:var(--ink)!important}
[data-testid="stMain"] [data-testid="stCaptionContainer"] *{color:var(--grey)!important}

/* ============ CABECERA ============ */
.mast{display:flex;justify-content:space-between;align-items:center;border-bottom:3px solid var(--ink);padding:0 0 .9rem;margin-bottom:2.2rem}
.brand{font:700 1.05rem 'Space Grotesk';letter-spacing:.2em;color:var(--ink)}
.brand span{background:var(--lime);color:var(--ink);padding:0 .35rem;border:2px solid var(--ink)}
.mono{font:500 .72rem 'DM Mono';letter-spacing:.12em;text-transform:uppercase;color:var(--grey)}
.eyebrow{display:inline-block;font:500 .7rem 'DM Mono';letter-spacing:.14em;text-transform:uppercase;background:var(--ink);color:var(--lime)!important;padding:.2rem .55rem;margin-bottom:.7rem}

/* ============ MENÚ LATERAL: índice numerado, sin círculos ni redondeados ============ */
[data-testid="stSidebar"]{background:var(--ink)!important;border-right:3px solid var(--ink)}
[data-testid="stSidebar"] *{color:var(--paper)!important}
[data-testid="stSidebar"] h2{font:700 2.4rem 'Space Grotesk'!important;letter-spacing:-.04em;color:var(--lime)!important;margin-bottom:1.2rem}
[data-testid="stSidebar"] hr{border-color:#3A3A36!important}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"]{gap:0!important;counter-reset:nav}
[data-testid="stSidebar"] [data-testid="stRadio"] label{counter-increment:nav;width:100%;margin:0;padding:.95rem .5rem;border-radius:0;border-top:1px solid #3A3A36;display:flex;align-items:center;transition:padding .15s,background .15s}
[data-testid="stSidebar"] [data-testid="stRadio"] label>div:first-child{display:none!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label::before{content:counter(nav,decimal-leading-zero);font:500 .72rem 'DM Mono';color:var(--lime);margin-right:1rem;letter-spacing:.1em}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{font:500 .95rem 'Space Grotesk';text-transform:uppercase;letter-spacing:.06em}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{padding-left:1rem}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:var(--lime)}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)::before{color:var(--ink)!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) *{color:var(--ink)!important;font-weight:700}

/* ============ HERO: bloque lima, texto tinta ============ */
.stApp .hero{background:var(--lime);color:var(--ink);padding:clamp(1.6rem,5vw,4.2rem);position:relative;overflow:hidden;margin:.3rem 0 2.2rem;border:3px solid var(--ink);box-shadow:10px 10px 0 var(--ink)}
.stApp .hero h1{color:var(--ink)!important;font-size:clamp(2.4rem,6.5vw,5.6rem)!important;line-height:.92!important;margin:.3rem 0 1rem!important;max-width:820px;text-transform:uppercase}
.stApp .hero p{color:var(--ink)!important;font-size:1.08rem;font-weight:500;max-width:540px}
.stApp .hero .eyebrow{background:var(--ink);color:var(--lime)!important}
.hero-mark{position:absolute;right:4%;top:50%;transform:translateY(-50%);width:190px;height:190px;background:var(--ink);color:var(--lime);display:grid;place-items:center;font:700 6rem 'Space Grotesk';box-shadow:8px 8px 0 var(--paper),8px 8px 0 3px var(--ink)}
@media (max-width:900px){.hero-mark{display:none}}

/* ============ TARJETAS: borde grueso + sombra dura ============ */
.rule{background:var(--white);border:2px solid var(--ink);padding:1rem 1.15rem;margin:.7rem 6px .9rem 0;box-shadow:6px 6px 0 var(--ink)}
.rule h3{font:500 1rem 'DM Sans'!important;letter-spacing:0!important;text-transform:none;line-height:1.5!important;margin:.3rem 0 .6rem}
.metric{background:var(--white);border:2px solid var(--ink);padding:1rem 1.1rem;box-shadow:6px 6px 0 var(--lime),6px 6px 0 2px var(--ink);margin:0 6px 1rem 0}
.metric strong{font:700 2.4rem 'Space Grotesk';display:block;color:var(--ink);letter-spacing:-.04em}
.metric span{color:var(--grey);font:.66rem 'DM Mono';text-transform:uppercase;letter-spacing:.1em}
.panel{padding:1rem 1.3rem;background:var(--lime);border:2px solid var(--ink);box-shadow:6px 6px 0 var(--ink);margin:.7rem 6px 1.4rem 0;color:var(--ink)}
.tag{display:inline-block;background:var(--white);color:var(--ink);border:2px solid var(--ink);padding:.2rem .6rem;margin:.22rem .15rem;font:500 .72rem 'DM Mono';text-transform:uppercase}
.status{background:var(--ink);color:var(--lime)!important;font:500 .68rem 'DM Mono';text-transform:uppercase;letter-spacing:.1em;padding:.15rem .5rem}

/* ============ BOTONES ============ */
.stButton>button{border-radius:0;background:var(--ink);border:2px solid var(--ink);padding:.7rem 1.2rem;box-shadow:5px 5px 0 var(--lime),5px 5px 0 2px var(--ink);transition:transform .1s,box-shadow .1s}
.stButton>button p{color:var(--lime)!important;font:700 .85rem 'Space Grotesk';text-transform:uppercase;letter-spacing:.1em}
.stButton>button:hover{background:var(--ink);border-color:var(--ink);transform:translate(3px,3px);box-shadow:2px 2px 0 var(--lime),2px 2px 0 2px var(--ink)}
.stButton>button:active{background:var(--lime)}
.stButton>button:active p{color:var(--ink)!important}

/* ============ INPUTS ============ */
[data-baseweb="select"]>div{background:var(--white)!important;border:2px solid var(--ink)!important;border-radius:0!important}
[data-baseweb="select"] *{color:var(--ink)!important}
[data-baseweb="popover"] ul,[data-baseweb="popover"] li{background:var(--white)!important;color:var(--ink)!important;border-radius:0!important}
[data-baseweb="popover"] li:hover{background:var(--lime)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--white)!important;border:2px dashed var(--ink)!important;border-radius:0!important}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink)!important}
[data-testid="stFileUploaderDropzone"] button{background:var(--ink)!important;border-radius:0!important;border:0!important}
[data-testid="stFileUploaderDropzone"] button *{color:var(--lime)!important}
[data-testid="stCameraInput"]{border:2px solid var(--ink);border-radius:0}
[data-testid="stCameraInput"] button{background:var(--ink)!important;border-radius:0!important}
[data-testid="stCameraInput"] button *{color:var(--lime)!important}
[data-testid="stImage"] img{border-radius:0;border:2px solid var(--ink)}
[data-testid="stAlert"]{border-radius:0;border:2px solid var(--ink)}
[data-testid="stAlert"] *{color:var(--ink)!important}

/* ============ PROGRESO, TABS, DIVISORES ============ */
[data-testid="stProgress"] [role="progressbar"]{border:2px solid var(--ink);background:var(--white)!important;border-radius:0!important}
[data-testid="stProgress"] [role="progressbar"]>div{background:var(--lime)!important;border-radius:0!important}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:3px solid var(--ink)}
.stTabs [data-baseweb="tab"]{font-family:'DM Mono';text-transform:uppercase;font-size:.74rem;letter-spacing:.08em;border-radius:0;padding:.5rem 1rem;background:transparent}
.stTabs [data-baseweb="tab"] p{color:var(--ink)!important}
.stTabs [aria-selected="true"]{background:var(--ink)!important}
.stTabs [aria-selected="true"] p{color:var(--lime)!important;font-weight:700}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
hr{border:0!important;border-top:2px solid var(--ink)!important;opacity:1!important}
footer{visibility:hidden}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def header():
    st.markdown('<div class="mast"><div class="brand">VANTA <span>●</span> INTELLIGENCE LAB</div><div class="mono">WASTE CLASSIFICATION / 10 CLASSES</div></div>', unsafe_allow_html=True)

def section(kicker, title, description=None):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    if description: st.write(description)

def fmt_percent(value): return f"{float(value):.1%}"

@st.cache_data
def metrics_data():
    try: return read_metrics(), None
    except Exception as exc: return None, str(exc)

def model_status():
    rows=[]
    for name, filename in MODEL_FILES.items():
        local=MODEL_DIR/filename
        if local.is_file():
            rows.append((name, True, local))
        elif SOURCE_MODELS is not None and (SOURCE_MODELS/filename).is_file():
            rows.append((name, True, SOURCE_MODELS/filename))
        else:
            rows.append((name, False, local))
    return rows

def render_metrics():
    data,error=metrics_data()
    if error: st.warning(f"No se pudieron leer las métricas: {error}")
    elif data is None: st.info("Los resultados de evaluación todavía no se han cargado. Agrega comparison_results.csv junto a los checkpoints.")
    else:
        names={"CNN desde cero":"CNN from scratch","Transfer Learning":"MobileNetV2","Vision Transformer":"Vision Transformer","CNN from scratch":"CNN from scratch","MobileNetV2":"MobileNetV2"}
        data=data.copy(); data["model_display"]=data["model"].map(lambda x:names.get(x,x))
        for metric,label in METRIC_COLUMNS.items():
            a,b,c,d=st.columns(4)
            with [a,b,c,d][list(METRIC_COLUMNS).index(metric)]:
                best=data.loc[data[metric].idxmax()]
                st.markdown(f'<div class="metric"><strong>{fmt_percent(best[metric])}</strong><span>{label} · {best["model_display"]}</span></div>',unsafe_allow_html=True)
        long=data.melt(id_vars="model_display",value_vars=list(METRIC_COLUMNS),var_name="metric",value_name="score")
        long["metric"]=long["metric"].map(METRIC_COLUMNS)
        fig=px.bar(long,x="metric",y="score",color="model_display",barmode="group",color_discrete_sequence=["#0E0E0E","#C6F432","#9C9A8C"],range_y=[0,1])
        fig.update_traces(marker_line_color="#0E0E0E",marker_line_width=2)
        fig.update_layout(paper_bgcolor="#F4F1E8",plot_bgcolor="#F4F1E8",font_family="DM Sans",font_color="#0E0E0E",legend_title_text="Architecture",margin=dict(l=8,r=8,t=18,b=8),yaxis_title="Score",xaxis_title="")
        fig.update_yaxes(gridcolor="#D8D4C6",linecolor="#0E0E0E"); fig.update_xaxes(showgrid=False,linecolor="#0E0E0E")
        st.plotly_chart(fig,use_container_width=True)

def show_results(results):
    if len(results)==3:
        agree=len({r["prediction"] for r in results})==1
        st.markdown(f'<div class="panel"><b>{"Three models agree" if agree else "Models disagree"}</b> · {"All models returned the same class." if agree else "Review the individual outputs; there is no automatic winner."}</div>',unsafe_allow_html=True)
    cols=st.columns(len(results))
    for col,result in zip(cols,results):
        with col:
            st.markdown(f'<div class="eyebrow">{result["model"]}</div>',unsafe_allow_html=True)
            st.subheader(result["prediction"].title())
            st.progress(result["confidence"],text=f'Confidence {result["confidence"]:.1%}')
            st.caption(f'{result["inference_ms"]:.1f} ms · {result["device"]}')
            st.markdown("**Top categories**")
            for item in result["top"]: st.write(f'{item["class"].title()} — {item["probability"]:.1%}')
    st.caption("Confidence is the model's probability distribution, not a guarantee that the predicted category is correct.")

def main():
    header()
    try: classes=load_class_names()
    except Exception as exc:
        st.error(f"No se pudieron cargar las clases: {exc}"); st.stop()
    with st.sidebar:
        st.markdown("## VANTA")
        page=st.radio("Navigate",["Overview","Live Vision","Model Arena","Research Lab","Dataset Explorer","About"],label_visibility="collapsed")
        st.markdown("---")
        st.markdown('<span class="mono">LOCAL INFERENCE · NO TRAINING</span>',unsafe_allow_html=True)
    if page=="Overview":
        st.markdown('<div class="hero"><div class="eyebrow">SMART WASTE INTELLIGENCE LAB</div><h1>Three Architectures.<br>One Vision.</h1><p>A hands-on comparison of convolutional, transfer-learning, and transformer models for waste classification.</p><div class="hero-mark">V</div></div>',unsafe_allow_html=True)
        cols=st.columns([1.2,1,1])
        with cols[0]: st.markdown('<div class="eyebrow">THE EXPERIMENT</div>',unsafe_allow_html=True); st.subheader("One image. Three ways to see it."); st.write("Run trained checkpoints on an image or webcam capture, inspect their confidence distributions, and compare their behavior without hiding disagreement.")
        with cols[1]: st.markdown('<div class="eyebrow">RECOGNIZED CLASSES</div>',unsafe_allow_html=True); st.markdown("".join(f'<span class="tag">{c}</span>' for c in classes),unsafe_allow_html=True)
        with cols[2]:
            st.markdown('<div class="eyebrow">MODEL AVAILABILITY</div>',unsafe_allow_html=True)
            for name,available,path in model_status(): st.markdown(f'<div class="rule"><span class="status">{"● READY" if available else "○ CHECKPOINT MISSING"}</span><br><b>{name}</b><br><span class="mono">{path.name}</span></div>',unsafe_allow_html=True)
        st.divider(); section("Evaluation snapshot","Reported results","Values loaded from comparison_results.csv. They describe the supplied evaluation and are not recalculated by this app."); render_metrics()
    elif page=="Live Vision":
        section("01 / Capture","Live Vision","Capture a frame with the browser camera or upload a local image, then run one selected checkpoint.")
        left,right=st.columns([1,1])
        with left:
            source=st.radio("Image source",["Webcam","Upload"],horizontal=True)
            if source=="Webcam": image=capture_image()
            else:
                upload=st.file_uploader("Choose an image",type=["jpg","jpeg","png","webp"])
                image=Image.open(upload).convert("RGB") if upload else None
            selected=st.selectbox("Architecture",list(MODEL_FILES))
            if image is not None and st.button("Run inference",key="single",use_container_width=True):
                try: st.session_state.single_result=predict(image,selected,classes)
                except Exception as exc: st.error(f"Inference failed: {exc}")
        with right:
            if image is not None: st.image(image,caption="Input image",use_container_width=True)
            if "single_result" in st.session_state: show_results([st.session_state.single_result])
            elif source=="Webcam": st.caption("If camera access is unavailable, check browser permissions and open this local app in a supported browser.")
    elif page=="Model Arena":
        section("02 / Compare","Model Arena","Run all three real checkpoints on the same image and inspect where their predictions align or diverge.")
        tab1,tab2=st.tabs(["WEBCAM","UPLOAD"])
        with tab1: cam=capture_image()
        with tab2:
            file=st.file_uploader("Choose an image for all three models",type=["jpg","jpeg","png","webp"],key="arena_upload")
            uploaded=Image.open(file).convert("RGB") if file else None
        image=uploaded or cam
        if image is not None:
            st.image(image,caption="Shared input",width=450)
            if st.button("Compare all models",key="arena",use_container_width=True):
                results=[]
                try:
                    bar=st.progress(0)
                    for i,name in enumerate(MODEL_FILES): results.append(predict(image,name,classes)); bar.progress((i+1)/3)
                    st.session_state.arena_results=results
                except Exception as exc: st.error(f"Comparison stopped: {exc}")
        if "arena_results" in st.session_state: show_results(st.session_state.arena_results)
    elif page=="Research Lab":
        section("03 / Method","Research Lab","The architectures differ in how they represent visual structure and how much prior knowledge they bring.")
        tab1,tab2=st.tabs(["ARCHITECTURES","EVALUATION"])
        with tab1:
            entries=[("01 / CNN FROM SCRATCH","Learns spatial features from the training data using stacked convolutions and pooling. Compact and fully trained for this task.","128 × 128 · mean/std 0.5"),("02 / MOBILENETV2","A convolutional network initialized through transfer learning, using inverted residual blocks and depthwise convolutions.","224 × 224 · ImageNet normalization"),("03 / VISION TRANSFORMER","Splits an image into patches and models relationships among them with self attention; often needs more compute.","224 × 224 · ImageNet normalization")]
            for title,body,detail in entries:
                st.markdown(f'<div class="rule"><div class="eyebrow">{title}</div><h3>{body}</h3><span class="mono">{detail}</span></div>',unsafe_allow_html=True)
        with tab2: render_metrics()
        st.caption("The CSV reports CNN, Transfer Learning, and Vision Transformer evaluation metrics. Transfer Learning is labeled MobileNetV2 in the application based on the supplied three-model setup.")
    elif page=="Dataset Explorer":
        section("04 / Data","Dataset Explorer","Class names follow the exact order in class_names.json. Local example images are shown only when present.")
        samples=[]
        search_roots=[ROOT/"dataset",ROOT/"data"]
        if SOURCE_MODELS is not None:
            search_roots.append(SOURCE_MODELS.parent/"dataset")
        for category in classes:
            found=[]
            for base in search_roots:
                folder=base/category
                if folder.is_dir(): found += [p for p in folder.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
            samples.append((category,found[:3]))
        for start in range(0,len(classes),2):
            cols=st.columns(2)
            for col,(category,paths) in zip(cols,samples[start:start+2]):
                with col:
                    st.markdown(f'<div class="rule"><span class="eyebrow">CLASS {start+1 if category==classes[start] else start+2:02d}</span><h3>{category.title()}</h3></div>',unsafe_allow_html=True)
                    if paths:
                        imgs=st.columns(min(3,len(paths)))
                        for c,p in zip(imgs,paths): c.image(str(p),use_container_width=True)
                    else: st.caption("No local sample images found for this class.")
    else:
        section("05 / Project","About","An academic prototype for studying image classification in a practical waste sorting context.")
        st.markdown("""
### Objective
VANTA places three trained classifiers behind one consistent image workflow so their outputs can be examined side by side.

### Method
The app loads local PyTorch checkpoints only when first needed, applies architecture-specific input size and normalization, and keeps loaded models in memory. It performs inference without training.

### Limitations
Outputs depend on the training dataset, camera conditions, and similarity between an input and the training distribution. A high confidence score does not establish correctness. Evaluation figures are displayed as reported in the supplied CSV; an input is not considered correctly classified unless its real label is independently known.

### Camera and privacy
Webcam capture uses Streamlit's built-in `st.camera_input`; no additional camera package is required. The image is processed by the local app and is not uploaded by VANTA to a hosted inference service.
""")
    st.markdown('<div class="rule mono">VANTA / SMART WASTE INTELLIGENCE LAB · LOCAL RESEARCH TOOL</div>',unsafe_allow_html=True)

if __name__=="__main__": main()