import io
import re
import sys
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="03A FontCLIP Logo Prescreener", layout="wide")

APP_VERSION = "1.2"
RUNTIME_ROOT = Path.home() / ".cache" / "fontclip_logo_prescreener"

# Pin the same FontCLIP source/checkpoint used in the successful pilot run.
FONTCLIP_COMMIT = "3d4c6af01f668800d8e4f9f4f753d29c74dad252"
FONTCLIP_ARCHIVE_URL = f"https://github.com/yukistavailable/FontCLIP/archive/{FONTCLIP_COMMIT}.zip"
FONTCLIP_CHECKPOINT_GDRIVE_ID = "1Tym7rAIuaGr6Gv-gZRSJmPstQjOWPgl1"
EXPECTED_CHECKPOINT_SHA256 = "c441277fbed4366d32d8fb65725189b97d3fe88bae5fe0648b969feea01bbb00"

FACETS = {
    "Sincerity": ["Down-to-earth", "Honest", "Wholesome", "Cheerful"],
    "Excitement": ["Daring", "Spirited", "Imaginative", "Up-to-date"],
    "Competence": ["Reliable", "Intelligent", "Successful"],
    "Sophistication": ["Upper-class", "Charming"],
    "Ruggedness": ["Outdoorsy", "Tough"],
}
DIMENSIONS = list(FACETS.keys())
FACET_NAMES = [f for fs in FACETS.values() for f in fs]
POS_PROMPTS = [f"{f.lower()} font" for f in FACET_NAMES]
NEG_PROMPTS = [f"not {f.lower()} font" for f in FACET_NAMES]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

CANONICAL_BRANDS = {
    "sulwhasoo": "Sulwhasoo", "mixsoon": "mixsoon", "thesaem": "THE SAEM",
    "etude": "ETUDE", "cnp": "CNP Laboratory", "tirtir": "TIRTIR",
    "larocheposay": "LA ROCHE-POSAY", "thefaceshop": "THE FACE SHOP",
    "naturerepublic": "NATURE REPUBLIC", "peripera": "peripera", "sum": "su:m37°",
    "fwee": "fwee", "loundlab": "ROUND LAB", "roundlab": "ROUND LAB",
    "tonymoly": "TONYMOLY", "thewhoo": "THE WHOO", "sooryehan": "Sooryehan",
    "cosrx": "COSRX", "dominas": "DOMINAS", "ourwhy": "OURWHY",
    "somebymi": "SOME BY MI", "joseon": "Beauty of Joseon", "fvrts": "FVRTS",
    "stembell": "STEMBELL", "retune": "retune", "medicube": "medicube",
    "neopharm": "NEOPHARM", "drbelmeur": "Dr.Belmeur", "fromrier": "fromrier",
    "amuse": "AMUSE", "numbuzin": "numbuzin", "alternativestereo": "alternative stereo",
    "freshian": "freshian", "celimax": "celimax", "torriden": "Torriden",
    "kahi": "KAHI", "skin1004": "SKIN1004", "codeglokolor": "code glökolor",
    "skinfood": "SKINFOOD", "glint": "Glint", "vdl": "VDL", "ohui": "OHUI",
    "centellian": "Centellian24+", "abib": "Abib", "hera": "HERA",
    "uglylovely": "UGLY LOVELY", "tiela": "TIELA", "missha": "MISSHA",
    "violetdream": "VIOLET DREAM", "fation": "FATION", "menokin": "MENOKIN",
    "anua": "Anua", "farmrx": "farmrx", "drjart": "Dr.Jart+", "laboh": "LABO-H",
    "iisaknox": "ISA KNOX", "isaknox": "ISA KNOX", "beyond": "BEYOND",
    "unove": "UNOVE", "laneige": "LANEIGE", "dalba": "d'Alba",
    "innisfree": "innisfree", "biohealboh": "BIOHEAL BOH", "aestura": "AESTURA",
    "drg": "Dr.G", "mediheal": "MEDIHEAL", "beplain": "beplain", "clio": "CLIO",
    "manyo": "ma:nyo", "tpsy": "TPSY", "age20s": "AGE20'S",
}

KNOWN_NOTES = {
    "mediheal_logotype.png": "이전 검토에서 파일 내용이 MISSHA로 보였음. 원본 확인 필요.",
    "larocheposay_logotype.png": "K-코스메틱 연구범위 해당 여부 확인 필요.",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_brand_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"(?i)_?logotype$", "", stem)
    raw = re.sub(r"[_-]+", " ", stem).strip()
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    return CANONICAL_BRANDS.get(key, raw)


def safe_extract_zip(uploaded_bytes: bytes):
    out, skipped = [], []
    with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/"):
                continue
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            if info.file_size > 20 * 1024 * 1024:
                skipped.append({"Filename": name, "Reason": "File > 20 MB"})
                continue
            data = zf.read(info)
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except Exception as e:
                skipped.append({"Filename": name, "Reason": f"Image decode failed: {e}"})
                continue
            out.append({"filename": name, "bytes": data, "image": img})
    return out, skipped


def inspect_logo(item: dict) -> dict:
    img = item["image"]
    gray = img.convert("L")
    arr = np.asarray(gray)
    mask = arr < 245
    if mask.any():
        ys, xs = np.where(mask)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_max = max(bbox_w, bbox_h)
    else:
        bbox_w = bbox_h = bbox_max = 0

    size_ok = img.size == (1024, 1024)
    format_ok = str(getattr(img, "format", "")).upper() == "PNG"
    mode_ok = img.mode in {"L", "RGB", "RGBA"}
    logo_size_ok = 760 <= bbox_max <= 840
    standard_ok = size_ok and format_ok and mode_ok and logo_size_ok
    base = Path(item["filename"]).name.lower()

    return {
        "Filename": item["filename"],
        "Brand": normalize_brand_name(item["filename"]),
        "Width": img.size[0],
        "Height": img.size[1],
        "Mode": img.mode,
        "Format": str(getattr(img, "format", "")),
        "Foreground_BBox_W": bbox_w,
        "Foreground_BBox_H": bbox_h,
        "Foreground_Max": bbox_max,
        "White_Background_Ratio": float((arr >= 250).mean()),
        "Standard_OK": bool(standard_ok),
        "Eligibility": "Unreviewed",
        "Researcher_Note": KNOWN_NOTES.get(base, ""),
    }


def merge_previous_review(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    out = current.copy()
    editable = ["Brand", "Eligibility", "Researcher_Note"]
    key = "Filename" if "Filename" in previous.columns else "Brand"
    if key not in out.columns or key not in previous.columns:
        return out
    prev = previous.drop_duplicates(key, keep="last").set_index(key)
    for i, row in out.iterrows():
        k = row[key]
        if k not in prev.index:
            continue
        for c in editable:
            if c in previous.columns:
                v = prev.loc[k, c]
                if pd.notna(v) and str(v).strip() != "":
                    out.at[i, c] = v
    return out


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    z = x / norm
    sim = np.clip(z @ z.T, -1.0, 1.0)
    return 1.0 - sim


def maximin_shortlist(x: np.ndarray, n: int) -> list[int]:
    n_total = len(x)
    if n_total == 0 or n <= 0:
        return []
    if n >= n_total:
        return list(range(n_total))
    d = cosine_distance_matrix(x)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    selected = [int(i)]
    if int(j) != int(i) and n > 1:
        selected.append(int(j))
    remaining = set(range(n_total)) - set(selected)
    while len(selected) < n and remaining:
        nxt = max(remaining, key=lambda k: float(d[k, selected].min()))
        selected.append(int(nxt))
        remaining.remove(nxt)
    return selected[:n]


def pca_2d_centered(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.zeros((len(x), 2))
    z = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    out = z @ vt[:2].T
    if out.shape[1] == 1:
        out = np.c_[out, np.zeros(len(out))]
    return out[:, :2]


def plot_radar(values, title: str):
    vals = list(values) + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMENSIONS, fontsize=8)
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.set_title(title, pad=18, fontsize=11)
    return fig


def zip_csv(files: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in files.items():
            zf.writestr(name, df.to_csv(index=False).encode("utf-8-sig"))
    return bio.getvalue()


def download_file(url: str, path: Path, timeout: int = 120):
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def prepare_fontclip_source() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    source_root = RUNTIME_ROOT / "source" / f"FontCLIP-{FONTCLIP_COMMIT}"
    if source_root.exists() and (source_root / "models").exists():
        return source_root
    archive_path = RUNTIME_ROOT / f"fontclip-{FONTCLIP_COMMIT}.zip"
    if not archive_path.exists():
        download_file(FONTCLIP_ARCHIVE_URL, archive_path)
    parent = RUNTIME_ROOT / "source"
    parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(parent)
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("FontCLIP-") and (p / "models").exists()]
    if not candidates:
        raise RuntimeError("FontCLIP source archive structure could not be recognized.")
    return candidates[0]


def prepare_checkpoint() -> Path:
    ckpt = RUNTIME_ROOT / "model_checkpoints" / "model.pt"
    if ckpt.exists() and ckpt.stat().st_size > 10 * 1024 * 1024:
        if sha256_file(ckpt) == EXPECTED_CHECKPOINT_SHA256:
            return ckpt
        ckpt.unlink(missing_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError("gdown is required to download the official FontCLIP checkpoint.") from e
    url = f"https://drive.google.com/uc?id={FONTCLIP_CHECKPOINT_GDRIVE_ID}"
    out = gdown.download(url, str(ckpt), quiet=False)
    if not out or not ckpt.exists() or ckpt.stat().st_size <= 10 * 1024 * 1024:
        raise RuntimeError("Official FontCLIP checkpoint download failed.")
    actual = sha256_file(ckpt)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch. Expected {EXPECTED_CHECKPOINT_SHA256}, got {actual}.")
    return ckpt


@st.cache_resource(show_spinner=False)
def load_fontclip_runtime():
    repo_dir = prepare_fontclip_source()
    checkpoint_path = prepare_checkpoint()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from models.init_model import device, load_model, preprocess
    from models.lora import LoRAConfig
    from utils.tokenizer import tokenize

    lora_config_text = LoRAConfig(
        r=256, alpha=1024.0, bias=False, learnable_alpha=False,
        apply_q=True, apply_k=True, apply_v=True, apply_out=True,
    )
    model = load_model(
        str(checkpoint_path), model_name="ViT-B/32",
        use_oft_vision=False, use_oft_text=False,
        oft_config_vision=None, oft_config_text=None,
        use_lora_text=True, use_lora_vision=False,
        lora_config_vision=None, lora_config_text=lora_config_text,
        use_coop_text=False, use_coop_vision=False,
        precontext_length_vision=10, precontext_length_text=77,
        precontext_dropout_rate=0, pt_applied_layers=None,
    )
    model.eval()
    meta = {
        "FontCLIP_commit": FONTCLIP_COMMIT,
        "Checkpoint_SHA256": sha256_file(checkpoint_path),
        "Model": "FontCLIP / ViT-B/32 / LoRA-text checkpoint",
        "Checkpoint_source": f"Google Drive ID {FONTCLIP_CHECKPOINT_GDRIVE_ID}",
        "Prompt_method": "paired positive/negative prompts for descriptive Aaker profile",
    }
    return model, preprocess, tokenize, device, meta


def analyze_images(items, model, preprocess, tokenize, device, batch_size=8):
    import torch
    pos_tokens = tokenize(POS_PROMPTS).to(device)
    neg_tokens = tokenize(NEG_PROMPTS).to(device)
    with torch.no_grad():
        pos_text = model.encode_text(pos_tokens).float()
        neg_text = model.encode_text(neg_tokens).float()
        pos_text = pos_text / pos_text.norm(dim=-1, keepdim=True)
        neg_text = neg_text / neg_text.norm(dim=-1, keepdim=True)

    image_embs, pos_sims, neg_sims = [], [], []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        tensors = [preprocess(x["image"].convert("RGB")) for x in batch]
        image_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            pos = image_features @ pos_text.T
            neg = image_features @ neg_text.T
        image_embs.append(image_features.cpu().numpy())
        pos_sims.append(pos.cpu().numpy())
        neg_sims.append(neg.cpu().numpy())
    return np.vstack(image_embs), np.vstack(pos_sims), np.vstack(neg_sims)


# ----------------------------------------------------------------------------
# UI — prescreening only
# ----------------------------------------------------------------------------
st.title(f"03A FontCLIP Logo Prescreener · v{APP_VERSION}")
st.caption("영문 로고타입 사전선별 전용 · 이미지 표준화 확인 → FontCLIP → 15 facets → 다양성 shortlist")
st.info(
    "이 앱은 본연구 전체를 수행하지 않습니다. 표준화된 로고타입 후보군을 FontCLIP으로 사전분석하여 "
    "서체 표현이 서로 다른 사례를 찾기 위한 prescreening 도구입니다. 공식 텍스트·패키지·웹사이트 분석은 후속 단계에서 별도로 수행합니다."
)

with st.expander("사전분석 기준"):
    st.write("**1차 기준:** FontCLIP 정규화 이미지 임베딩 간 cosine distance")
    st.write("**보조 지표:** Aaker 15 facets에 대해 positive / negative prompt를 모두 계산하고, contrast(positive − negative)를 5차원으로 요약")
    rows = []
    for d, fs in FACETS.items():
        for f in fs:
            rows.append((d, f, f"{f.lower()} font", f"not {f.lower()} font"))
    st.dataframe(pd.DataFrame(rows, columns=["Dimension", "Facet", "Positive prompt", "Negative prompt"]), use_container_width=True, hide_index=True)
    st.write(f"**Pinned FontCLIP commit:** `{FONTCLIP_COMMIT}`")

uploaded = st.file_uploader("표준화된 로고타입 ZIP 업로드", type=["zip"], help="예: logo_dataset.zip")
previous_review_file = st.file_uploader("이전 A/B/C 검토 CSV 불러오기 (선택)", type=["csv"], key="previous_review")

if uploaded:
    try:
        items, skipped = safe_extract_zip(uploaded.getvalue())
    except Exception as e:
        st.error(f"ZIP 읽기 실패: {e}")
        st.stop()
    if not items:
        st.error("분석 가능한 이미지가 없습니다.")
        st.stop()

    inspection = pd.DataFrame([inspect_logo(x) for x in items])
    if previous_review_file is not None:
        try:
            inspection = merge_previous_review(inspection, pd.read_csv(previous_review_file))
            st.success("이전 A/B/C 검토값을 불러왔습니다.")
        except Exception as e:
            st.warning(f"이전 검토 CSV 병합 실패: {e}")

    st.success(f"로고 이미지 {len(items)}개 인식 완료")
    if skipped:
        st.warning(f"읽지 못한 이미지 {len(skipped)}개")
        st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)

    st.subheader("1. 이미지 표준화 및 A/B/C 검토")
    cols = st.columns(5)
    cols[0].metric("Images", len(inspection))
    cols[1].metric("1024×1024", int(((inspection.Width == 1024) & (inspection.Height == 1024)).sum()))
    cols[2].metric("PNG", int((inspection.Format.str.upper() == "PNG").sum()))
    cols[3].metric("Standard OK", int(inspection.Standard_OK.sum()))
    cols[4].metric("Unreviewed", int((inspection.Eligibility == "Unreviewed").sum()))

    review = st.data_editor(
        inspection,
        disabled=[c for c in inspection.columns if c not in ["Brand", "Eligibility", "Researcher_Note"]],
        column_config={
            "Eligibility": st.column_config.SelectboxColumn(
                "Eligibility", options=["Unreviewed", "A", "B", "C"], required=True,
                help="A=바로 사용 가능, B=전처리/확인 후 가능, C=제외",
            ),
            "Researcher_Note": st.column_config.TextColumn("Researcher_Note"),
        },
        use_container_width=True, height=440, hide_index=True, key="logo_review_editor",
    )
    for i, b in enumerate(review["Brand"].astype(str).tolist()):
        items[i]["brand"] = b

    with st.expander("로고 이미지 미리보기"):
        idx = st.selectbox("이미지 선택", range(len(items)), format_func=lambda i: f"{review.iloc[i]['Brand']} · {Path(items[i]['filename']).name}")
        st.image(items[idx]["image"], width=520)
        st.write(review.iloc[idx].to_dict())

    st.subheader("2. FontCLIP 사전분석")
    batch_size = st.select_slider("Batch size", options=[1, 2, 4, 8, 16], value=8)
    if st.button("FontCLIP 사전분석 실행", type="primary"):
        try:
            with st.spinner("FontCLIP 모델 준비 중..."):
                model, preprocess, tokenize, device, model_meta = load_fontclip_runtime()
            with st.spinner(f"{len(items)}개 로고타입 분석 중..."):
                image_emb, pos_sims, neg_sims = analyze_images(items, model, preprocess, tokenize, device, int(batch_size))
        except Exception as e:
            st.exception(e)
            st.error("FontCLIP 실행 실패. Streamlit Cloud Python 3.12 및 dependency를 확인하세요.")
            st.stop()

        contrast = pos_sims - neg_sims
        dim_dict = {}
        for d, fs in FACETS.items():
            idxs = [FACET_NAMES.index(f) for f in fs]
            dim_dict[d] = contrast[:, idxs].mean(axis=1)
        dim_df = pd.DataFrame(dim_dict)

        emb_dmat = cosine_distance_matrix(image_emb)
        emb_mean = emb_dmat.mean(axis=1)
        emb_rank = pd.Series(emb_mean).rank(method="min", ascending=False).astype(int).to_numpy()
        emb_coords = pca_2d_centered(image_emb)

        results = review.reset_index(drop=True).copy()
        for i, f in enumerate(FACET_NAMES):
            results[f"Positive_{f}"] = pos_sims[:, i]
            results[f"Negative_{f}"] = neg_sims[:, i]
            results[f"Contrast_{f}"] = contrast[:, i]
        for d in DIMENSIONS:
            results[d] = dim_df[d]
        results["Primary_Dimension_Reference"] = results[DIMENSIONS].idxmax(axis=1)
        results["Embedding_Mean_Distance"] = emb_mean
        results["Embedding_Diversity_Rank"] = emb_rank
        results["Embedding_PCA1"] = emb_coords[:, 0]
        results["Embedding_PCA2"] = emb_coords[:, 1]

        emb_cols = [f"Embedding_{i:03d}" for i in range(image_emb.shape[1])]
        embedding_df = pd.DataFrame(image_emb, columns=emb_cols)
        embedding_df.insert(0, "Brand", review["Brand"].astype(str).tolist())
        embedding_df.insert(0, "Filename", review["Filename"].astype(str).tolist())

        st.session_state["R"] = {
            "results": results,
            "review": review.copy(),
            "embedding_df": embedding_df,
            "distance": pd.DataFrame(emb_dmat, index=results.Brand, columns=results.Brand),
            "meta": model_meta,
            "pos_sims": pos_sims,
            "neg_sims": neg_sims,
            "contrast": contrast,
        }
        st.success("FontCLIP 사전분석 완료")

if "R" in st.session_state:
    R = st.session_state["R"]
    results = R["results"]
    st.divider()
    st.header("사전분석 결과")

    st.subheader("3. FontCLIP 임베딩 다양성")
    st.dataframe(
        results[["Brand", "Eligibility", "Embedding_Diversity_Rank", "Embedding_Mean_Distance"] + DIMENSIONS]
        .sort_values("Embedding_Diversity_Rank"),
        use_container_width=True, hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(results.Embedding_PCA1, results.Embedding_PCA2)
    for _, r in results.iterrows():
        ax.annotate(str(r.Brand), (r.Embedding_PCA1, r.Embedding_PCA2), fontsize=6, alpha=0.75)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("FontCLIP image embedding distribution")
    st.pyplot(fig, use_container_width=True)
    st.caption("PCA는 시각화용이며 shortlist는 FontCLIP 이미지 임베딩의 cosine distance로 계산합니다.")

    st.subheader("4. Aaker 15 facets / 5차원 참고 프로파일")
    brand = st.selectbox("브랜드 상세 보기", results.Brand.tolist())
    row = results[results.Brand == brand].iloc[0]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(plot_radar([float(row[d]) for d in DIMENSIONS], f"{brand} · reference 5D"), use_container_width=True)
    with c2:
        ref = pd.DataFrame({"Dimension": DIMENSIONS, "Contrast score": [float(row[d]) for d in DIMENSIONS]})
        st.dataframe(ref, use_container_width=True, hide_index=True)

    facet_rows = []
    for d, fs in FACETS.items():
        for f in fs:
            facet_rows.append({
                "Dimension": d,
                "Facet": f,
                "Positive": float(row[f"Positive_{f}"]),
                "Negative": float(row[f"Negative_{f}"]),
                "Contrast": float(row[f"Contrast_{f}"]),
            })
    facet_brand_df = pd.DataFrame(facet_rows).sort_values(["Dimension", "Facet"])
    st.dataframe(facet_brand_df, use_container_width=True, hide_index=True)
    st.caption("Aaker 15 facet은 positive / negative prompt를 모두 계산하고, contrast(positive − negative)를 참고 프로파일로 제시합니다. 이 값은 표본 선정의 1차 기준이 아니라 로고타입의 의미적 경향을 확인하는 참고값입니다.")

    st.subheader("5. 다양성 shortlist")
    pool_option = st.radio("후보 범위", ["모든 분석 로고", "A 등급만", "A+B 등급"], horizontal=True)
    if pool_option == "A 등급만":
        pool = results[results.Eligibility == "A"].copy()
    elif pool_option == "A+B 등급":
        pool = results[results.Eligibility.isin(["A", "B"])].copy()
    else:
        pool = results.copy()

    shortlist = pd.DataFrame(columns=results.columns)
    if len(pool) < 2:
        st.warning("선택한 범위의 후보가 2개 미만입니다.")
    else:
        default_n = min(10, len(pool))
        n_short = st.number_input("Shortlist 수", min_value=2, max_value=len(pool), value=default_n, step=1)
        pool_emb = R["embedding_df"].set_index("Filename")
        emb_cols = [c for c in pool_emb.columns if c.startswith("Embedding_")]
        x = np.vstack([pool_emb.loc[f, emb_cols].to_numpy(dtype=float) for f in pool.Filename])
        idxs = maximin_shortlist(x, int(n_short))
        shortlist = pool.iloc[idxs].copy()
        shortlist["Shortlist_Order"] = range(1, len(shortlist) + 1)
        st.dataframe(
            shortlist[["Shortlist_Order", "Brand", "Eligibility", "Embedding_Mean_Distance", "Primary_Dimension_Reference"] + DIMENSIONS],
            use_container_width=True, hide_index=True,
        )
        st.caption("이 shortlist는 FontCLIP 서체 임베딩의 다양성을 기준으로 한 사전선별 결과입니다. 최종 연구대상 확정은 별도 단계에서 수행합니다.")

    marked = results.copy()
    marked["Prescreen_Shortlist"] = marked.Brand.isin(shortlist.Brand).astype(int) if len(shortlist) else 0

    prompt_df = pd.DataFrame(
        [(d, f, f"{f.lower()} font", f"not {f.lower()} font") for d, fs in FACETS.items() for f in fs],
        columns=["Dimension", "Facet", "Positive_Prompt", "Negative_Prompt"],
    )
    meta_df = pd.DataFrame([{
        "App_Version": APP_VERSION,
        **R["meta"],
        "N_Logos": len(results),
        "N_Shortlist": len(shortlist),
        "Shortlist_Pool": pool_option,
        "Primary_Prescreen_Method": "FontCLIP normalized image embedding cosine distance + deterministic maximin",
        "Aaker_Profile_Role": "Descriptive reference only",
    }])

    dmat = R["distance"].copy()
    dmat.index.name = "Brand"
    distance_matrix = dmat.reset_index()
    distance_long = dmat.rename_axis(index="Brand_A", columns="Brand_B").stack().rename("Cosine_Distance").reset_index()

    pos_cols = [c for c in marked.columns if c.startswith("Positive_")]
    neg_cols = [c for c in marked.columns if c.startswith("Negative_")]
    contrast_cols = [c for c in marked.columns if c.startswith("Contrast_")]
    facet_export_wide = marked[["Filename", "Brand", "Eligibility", "Researcher_Note"] + pos_cols + neg_cols + contrast_cols].copy()

    facet_long_rows = []
    for _, r in marked.iterrows():
        for d, fs in FACETS.items():
            for f in fs:
                facet_long_rows.append({
                    "Filename": r["Filename"],
                    "Brand": r["Brand"],
                    "Eligibility": r["Eligibility"],
                    "Researcher_Note": r["Researcher_Note"],
                    "Dimension": d,
                    "Facet": f,
                    "Positive": r[f"Positive_{f}"],
                    "Negative": r[f"Negative_{f}"],
                    "Contrast": r[f"Contrast_{f}"],
                })
    facet_export_long = pd.DataFrame(facet_long_rows)
    profile_export = marked[["Filename", "Brand", "Eligibility", "Researcher_Note", "Primary_Dimension_Reference"] + DIMENSIONS].copy()

    files = {
        "03A_01_logo_standardization_review.csv": R["review"],
        "03A_02_fontclip_prescreen_results.csv": marked,
        "03A_03_fontclip_image_embeddings.csv": R["embedding_df"],
        "03A_04_fontclip_pairwise_distance_matrix.csv": distance_matrix,
        "03A_05_fontclip_pairwise_distance_long.csv": distance_long,
        "03A_06_fontclip_diversity_shortlist.csv": shortlist,
        "03A_07_aaker_15facet_scores_wide.csv": facet_export_wide,
        "03A_08_aaker_15facet_scores_long.csv": facet_export_long,
        "03A_09_aaker_5D_reference.csv": profile_export,
        "03A_10_prompt_definition.csv": prompt_df,
        "03A_11_run_metadata.csv": meta_df,
    }

    st.subheader("6. CSV 다운로드")
    st.download_button(
        "모든 CSV ZIP 다운로드", zip_csv(files),
        file_name="03A_fontclip_prescreen_csv_v1_2.zip", mime="application/zip", type="primary",
    )
    items2 = list(files.items())
    for i in range(0, len(items2), 3):
        cols = st.columns(3)
        for col, (name, df) in zip(cols, items2[i:i+3]):
            with col:
                st.download_button(name.replace(".csv", ""), df.to_csv(index=False).encode("utf-8-sig"), name, "text/csv", use_container_width=True)

st.divider()
st.caption(
    "범위: 이 앱은 FontCLIP 기반 로고타입 사전선별까지만 수행합니다. 공식 커뮤니케이션 텍스트 분석, 패키지·웹사이트 분석, 일치도·일관성 및 인간평가는 별도 연구 단계입니다."
)
