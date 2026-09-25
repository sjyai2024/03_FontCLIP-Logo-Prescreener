import io
import os
import re
import sys
import json
import math
import time
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

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="03A FontCLIP Logo Prescreener", layout="wide")

APP_VERSION = "1.1"
RUNTIME_ROOT = Path.home() / ".cache" / "fontclip_logo_prescreener"
FONTCLIP_REPO_API = "https://api.github.com/repos/yukistavailable/FontCLIP/commits/main"
FONTCLIP_ARCHIVE_URL = "https://github.com/yukistavailable/FontCLIP/archive/{sha}.zip"
FONTCLIP_CHECKPOINT_GDRIVE_ID = "1Tym7rAIuaGr6Gv-gZRSJmPstQjOWPgl1"

# Aaker (1997) 15 facets -> 5 dimensions.
# Primary analysis uses raw FontCLIP cosine similarity.
FACETS = {
    "Sincerity": ["Down-to-earth", "Honest", "Wholesome", "Cheerful"],
    "Excitement": ["Daring", "Spirited", "Imaginative", "Up-to-date"],
    "Competence": ["Reliable", "Intelligent", "Successful"],
    "Sophistication": ["Upper-class", "Charming"],
    "Ruggedness": ["Outdoorsy", "Tough"],
}
DIMENSIONS = list(FACETS.keys())
FACET_NAMES = [f for fs in FACETS.values() for f in fs]

# FontCLIP's public demo converts semantic queries to "... font" prompts.
# Keep a single fixed prompt template for reproducibility.
PROMPT_TEMPLATE = "{} font"
PROMPTS = [PROMPT_TEMPLATE.format(f.lower()) for f in FACET_NAMES]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
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
    stem = re.sub(r"[_-]+", " ", stem).strip()
    return stem


def safe_extract_zip(uploaded_bytes: bytes) -> list[dict]:
    """Read logo images from an uploaded ZIP without writing untrusted paths."""
    out = []
    with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/"):
                continue
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            # Protect against ZIP bombs / giant accidental files.
            if info.file_size > 20 * 1024 * 1024:
                continue
            data = zf.read(info)
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except Exception:
                continue
            out.append({"filename": name, "bytes": data, "image": img})
    return out


def inspect_logo(item: dict) -> dict:
    img = item["image"]
    gray = img.convert("L")
    arr = np.asarray(gray)

    # Foreground estimate. Anti-aliased black/gray logo pixels are typically <245.
    mask = arr < 245
    if mask.any():
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        bbox_max = max(bbox_w, bbox_h)
    else:
        bbox_w = bbox_h = bbox_max = 0

    white_ratio = float((arr >= 250).mean())
    size_ok = img.size == (1024, 1024)
    format_ok = str(getattr(img, "format", "")).upper() == "PNG"
    canvas_mode_ok = img.mode in {"L", "RGB", "RGBA"}
    # User's preprocessing rule is nominally 800 px; allow anti-aliasing/cropping tolerance.
    logo_size_ok = 760 <= bbox_max <= 840
    standard_ok = size_ok and format_ok and canvas_mode_ok and logo_size_ok

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
        "White_Background_Ratio": white_ratio,
        "Standard_OK": bool(standard_ok),
        "Eligibility": "Unreviewed",  # Researcher review: A / B / C
        "Researcher_Note": "",
    }


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    z = x / norm
    sim = np.clip(z @ z.T, -1.0, 1.0)
    return 1.0 - sim


def maximin_shortlist(x: np.ndarray, n: int) -> list[int]:
    """Deterministic maximum-variation shortlist using 5D profile cosine distance."""
    n_total = len(x)
    if n_total == 0 or n <= 0:
        return []
    if n >= n_total:
        return list(range(n_total))

    d = cosine_distance_matrix(x)
    # Start with the profile that is, on average, furthest from all others.
    selected = [int(np.argmax(d.mean(axis=1)))]
    remaining = set(range(n_total)) - set(selected)

    while len(selected) < n and remaining:
        # Maximize minimum distance to already selected profiles.
        nxt = max(remaining, key=lambda i: float(d[i, selected].min()))
        selected.append(int(nxt))
        remaining.remove(nxt)
    return selected


def pca_2d(x: np.ndarray) -> np.ndarray:
    """Dependency-free PCA for visualization only."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.zeros((len(x), 2))
    std = x.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    z = (x - x.mean(axis=0)) / std
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    comps = vt[:2].T
    out = z @ comps
    if out.shape[1] == 1:
        out = np.c_[out, np.zeros(len(out))]
    return out[:, :2]


def plot_radar(values, title: str):
    labels = DIMENSIONS
    vals = list(values) + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, pad=18, fontsize=11)
    return fig


def zip_csv(files: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in files.items():
            zf.writestr(name, df.to_csv(index=False).encode("utf-8-sig"))
    return bio.getvalue()

# -----------------------------------------------------------------------------
# FontCLIP runtime preparation
# -----------------------------------------------------------------------------
def get_latest_commit_sha() -> str:
    r = requests.get(FONTCLIP_REPO_API, timeout=20)
    r.raise_for_status()
    return r.json()["sha"]


def download_file(url: str, path: Path, timeout: int = 120):
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def prepare_fontclip_source() -> tuple[Path, str]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    sha_file = RUNTIME_ROOT / "fontclip_commit.txt"
    source_root = RUNTIME_ROOT / "source"

    # If a prepared source exists, reuse it and its recorded SHA.
    if sha_file.exists() and source_root.exists():
        sha = sha_file.read_text().strip()
        candidates = [p for p in source_root.iterdir() if p.is_dir() and (p / "models").exists()]
        if candidates:
            return candidates[0], sha

    sha = get_latest_commit_sha()
    archive_path = RUNTIME_ROOT / f"fontclip-{sha}.zip"
    if not archive_path.exists():
        download_file(FONTCLIP_ARCHIVE_URL.format(sha=sha), archive_path)

    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(source_root)

    candidates = [p for p in source_root.iterdir() if p.is_dir() and (p / "models").exists()]
    if not candidates:
        raise RuntimeError("FontCLIP source archive structure could not be recognized.")
    repo_dir = candidates[0]
    sha_file.write_text(sha)
    return repo_dir, sha


def prepare_checkpoint() -> Path:
    """Download the official checkpoint referenced by FontCLIP setup_data.py."""
    ckpt = RUNTIME_ROOT / "model_checkpoints" / "model.pt"
    if ckpt.exists() and ckpt.stat().st_size > 10 * 1024 * 1024:
        return ckpt

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError("gdown is required to download the official FontCLIP checkpoint.") from e

    url = f"https://drive.google.com/uc?id={FONTCLIP_CHECKPOINT_GDRIVE_ID}"
    out = gdown.download(url, str(ckpt), quiet=False)
    if not out or not ckpt.exists() or ckpt.stat().st_size <= 10 * 1024 * 1024:
        raise RuntimeError("Official FontCLIP checkpoint download failed.")
    return ckpt


@st.cache_resource(show_spinner=False)
def load_fontclip_runtime():
    """Load the official FontCLIP model and checkpoint, following font_retrieval.py."""
    repo_dir, commit_sha = prepare_fontclip_source()
    checkpoint_path = prepare_checkpoint()

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import torch
    from models.init_model import device, load_model, preprocess
    from models.lora import LoRAConfig
    from utils.tokenizer import tokenize

    lora_config_text = LoRAConfig(
        r=256,
        alpha=1024.0,
        bias=False,
        learnable_alpha=False,
        apply_q=True,
        apply_k=True,
        apply_v=True,
        apply_out=True,
    )

    model = load_model(
        str(checkpoint_path),
        model_name="ViT-B/32",
        use_oft_vision=False,
        use_oft_text=False,
        oft_config_vision=None,
        oft_config_text=None,
        use_lora_text=True,
        use_lora_vision=False,
        lora_config_vision=None,
        lora_config_text=lora_config_text,
        use_coop_text=False,
        use_coop_vision=False,
        precontext_length_vision=10,
        precontext_length_text=77,
        precontext_dropout_rate=0,
        pt_applied_layers=None,
    )
    model.eval()

    metadata = {
        "FontCLIP_commit": commit_sha,
        "Checkpoint_SHA256": sha256_file(checkpoint_path),
        "Model": "FontCLIP / ViT-B/32 / LoRA-text checkpoint",
        "Prompt_template": PROMPT_TEMPLATE,
        "Checkpoint_source": f"Google Drive ID {FONTCLIP_CHECKPOINT_GDRIVE_ID}",
    }
    return model, preprocess, tokenize, device, metadata


def analyze_images(items: list[dict], model, preprocess, tokenize, device, batch_size: int = 8):
    import torch

    tokenized = tokenize(PROMPTS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_sims = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        tensors = [preprocess(x["image"].convert("RGB")) for x in batch]
        image_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = image_features @ text_features.T
        all_sims.append(sims.cpu().numpy())

    return np.vstack(all_sims) if all_sims else np.empty((0, len(PROMPTS)))

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title(f"03A FontCLIP Logo Prescreener · v{APP_VERSION}")
st.caption("K-cosmetic English logotypes · Aaker 15 facets → 5 dimensions · preliminary maximum-variation sampling")

st.info(
    "이 도구는 최종 브랜드 개성을 판정하기 위한 것이 아니라, 표준화된 영문 로고타입 후보군의 "
    "FontCLIP 의미 프로파일 분포를 사전 확인하여 서로 다른 사례를 포함하는 표본을 구성하기 위한 보조도구입니다. "
    "본 분석값은 raw cosine similarity를 사용합니다."
)

with st.expander("고정 분석 기준"):
    st.write("**Aaker 5 dimensions / 15 facets**")
    st.dataframe(
        pd.DataFrame([(d, f, PROMPT_TEMPLATE.format(f.lower())) for d, fs in FACETS.items() for f in fs],
                     columns=["Dimension", "Facet", "FontCLIP prompt"]),
        use_container_width=True,
        hide_index=True,
    )
    st.write("**FontCLIP source:** Tatsukawa et al. (2024), official repository `yukistavailable/FontCLIP`")
    st.write("**Primary value:** image–text cosine similarity. Softmax percentage is not used as the primary metric.")

uploaded = st.file_uploader("표준화된 로고타입 ZIP 업로드", type=["zip"], help="예: logo_dataset.zip")

if uploaded:
    try:
        items = safe_extract_zip(uploaded.getvalue())
    except Exception as e:
        st.error(f"ZIP 읽기 실패: {e}")
        st.stop()

    if not items:
        st.error("분석 가능한 PNG/JPG/WebP 이미지가 없습니다.")
        st.stop()

    inspection = pd.DataFrame([inspect_logo(x) for x in items])
    st.success(f"로고 이미지 {len(items)}개 인식 완료")

    st.subheader("1. 이미지 표준화 조건 확인")
    cols = st.columns(4)
    cols[0].metric("Images", len(inspection))
    cols[1].metric("1024×1024", int(((inspection.Width == 1024) & (inspection.Height == 1024)).sum()))
    cols[2].metric("PNG", int((inspection.Format.str.upper() == "PNG").sum()))
    cols[3].metric("Standard OK", int(inspection.Standard_OK.sum()))

    st.caption("Foreground_Max는 흰 배경에서 비백색 문자 영역의 최대 가로/세로 길이를 추정한 값입니다. nominal 800 px에 ±40 px 허용범위를 사용합니다.")

    review = st.data_editor(
        inspection,
        disabled=[c for c in inspection.columns if c not in ["Brand", "Eligibility", "Researcher_Note"]],
        column_config={
            "Eligibility": st.column_config.SelectboxColumn(
                "Eligibility", options=["Unreviewed", "A", "B", "C"], required=True,
                help="A=바로 사용 가능, B=전처리/확인 후 가능, C=제외"
            ),
            "Researcher_Note": st.column_config.TextColumn("Researcher_Note"),
        },
        use_container_width=True,
        height=420,
        hide_index=True,
        key="logo_review_editor",
    )

    # Keep researcher-edited brand labels aligned to image order.
    for i, b in enumerate(review["Brand"].astype(str).tolist()):
        items[i]["brand"] = b

    with st.expander("로고 이미지 미리보기"):
        idx = st.selectbox("이미지 선택", range(len(items)), format_func=lambda i: f"{review.iloc[i]['Brand']} · {Path(items[i]['filename']).name}")
        st.image(items[idx]["image"], width=520)
        st.write(review.iloc[idx].to_dict())

    st.subheader("2. FontCLIP 실행")
    st.caption("처음 실행 시 공식 FontCLIP 소스와 공개 체크포인트를 다운로드하여 준비하므로 시간이 걸릴 수 있습니다.")
    batch_size = st.select_slider("Batch size", options=[1, 2, 4, 8, 16], value=8)

    if st.button("FontCLIP 사전분석 실행", type="primary"):
        try:
            with st.spinner("공식 FontCLIP 모델 준비 중..."):
                model, preprocess, tokenize, device, model_meta = load_fontclip_runtime()
            with st.spinner(f"{len(items)}개 로고타입 분석 중..."):
                sims = analyze_images(items, model, preprocess, tokenize, device, int(batch_size))
        except Exception as e:
            st.exception(e)
            st.error(
                "FontCLIP 준비 또는 실행에 실패했습니다. Streamlit Cloud에서는 Python 3.11/3.12를 권장하며, "
                "네트워크에서 GitHub/Google Drive 다운로드가 허용되어야 합니다."
            )
            st.stop()

        facet_cols = ["Facet_" + f for f in FACET_NAMES]
        facet_df = pd.DataFrame(sims, columns=facet_cols)

        dim_dict = {}
        for d, fs in FACETS.items():
            idxs = [FACET_NAMES.index(f) for f in fs]
            dim_dict[d] = sims[:, idxs].mean(axis=1)
        dim_df = pd.DataFrame(dim_dict)

        results = pd.concat([
            review.reset_index(drop=True),
            facet_df,
            dim_df,
        ], axis=1)
        results["Primary_Dimension"] = results[DIMENSIONS].idxmax(axis=1)

        # Profile diversity is based on scale-invariant cosine distance among 5D vectors.
        dmat = cosine_distance_matrix(results[DIMENSIONS].to_numpy())
        results["Mean_Profile_Distance"] = dmat.mean(axis=1)
        results["Diversity_Rank"] = results["Mean_Profile_Distance"].rank(method="min", ascending=False).astype(int)

        coords = pca_2d(results[DIMENSIONS].to_numpy())
        results["PCA1"] = coords[:, 0]
        results["PCA2"] = coords[:, 1]

        st.session_state["fontclip_results"] = {
            "results": results,
            "distance": pd.DataFrame(dmat, index=results.Brand, columns=results.Brand),
            "review": review.copy(),
            "meta": model_meta,
            "items": items,
        }
        st.success("FontCLIP 사전분석 완료")

if "fontclip_results" in st.session_state:
    R = st.session_state["fontclip_results"]
    results = R["results"]
    st.divider()
    st.header("사전분석 결과")

    st.subheader("3. 5차원 프로파일")
    show_cols = ["Brand", "Primary_Dimension", "Diversity_Rank", "Mean_Profile_Distance"] + DIMENSIONS
    st.dataframe(results[show_cols].sort_values("Diversity_Rank"), use_container_width=True, hide_index=True)

    brand = st.selectbox("브랜드 상세 보기", results.Brand.tolist(), key="detail_brand")
    row = results[results.Brand == brand].iloc[0]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(plot_radar([float(row[d]) for d in DIMENSIONS], f"{brand} · FontCLIP 5D"), use_container_width=True)
    with c2:
        ftable = pd.DataFrame({
            "Facet": FACET_NAMES,
            "Cosine": [float(row["Facet_" + f]) for f in FACET_NAMES],
        }).sort_values("Cosine", ascending=False)
        st.dataframe(ftable, use_container_width=True, hide_index=True)

    st.subheader("4. 후보군 프로파일 분포")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(results.PCA1, results.PCA2)
    for _, r in results.iterrows():
        ax.annotate(str(r.Brand), (r.PCA1, r.PCA2), fontsize=6, alpha=0.75)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("FontCLIP 5D profile distribution (visualization only)")
    st.pyplot(fig, use_container_width=True)
    st.caption("PCA는 후보군의 5차원 프로파일 분포를 시각적으로 확인하기 위한 보조표현이며 최종 선정 기준값 자체는 아닙니다.")

    st.subheader("5. 최대변이 표본 shortlist")
    eligible_option = st.radio(
        "Shortlist 후보 범위",
        ["모든 분석 로고", "A 등급만", "A+B 등급"],
        horizontal=True,
    )
    if eligible_option == "A 등급만":
        pool = results[results.Eligibility == "A"].copy()
    elif eligible_option == "A+B 등급":
        pool = results[results.Eligibility.isin(["A", "B"])].copy()
    else:
        pool = results.copy()

    shortlist = pd.DataFrame(columns=results.columns)
    if len(pool) == 0:
        st.warning("선택한 후보 범위에 해당하는 로고가 없습니다. Eligibility를 먼저 검토하세요.")
    else:
        n_short = st.number_input("Shortlist 브랜드 수", min_value=2, max_value=max(2, len(pool)), value=min(10, len(pool)), step=1)
        selected_idx = maximin_shortlist(pool[DIMENSIONS].to_numpy(), int(n_short))
        shortlist = pool.iloc[selected_idx].copy()
        shortlist["Shortlist_Order"] = range(1, len(shortlist) + 1)
        shortlist = shortlist[["Shortlist_Order"] + [c for c in shortlist.columns if c != "Shortlist_Order"]]
        st.dataframe(shortlist[["Shortlist_Order", "Brand", "Eligibility", "Primary_Dimension", "Mean_Profile_Distance"] + DIMENSIONS], use_container_width=True, hide_index=True)
        st.caption(
            "이 shortlist는 5차원 프로파일 간 cosine distance의 maximin 방식으로 서로 다른 사례를 우선 배치한 보조결과입니다. "
            "최종 표본은 공식 브랜드 텍스트·패키지·웹사이트 확보 가능성을 추가 확인한 뒤 연구자가 확정해야 합니다."
        )

    # -------------------------------------------------------------------------
    # CSV exports: make every analysis output downloadable as an individual CSV
    # as well as one ZIP archive. Downloads are available even when no shortlist
    # can be produced for the selected eligibility pool.
    # -------------------------------------------------------------------------
    marked = results.copy()
    if len(shortlist):
        marked["Diversity_Shortlist"] = marked.Brand.isin(shortlist.Brand).astype(int)
    else:
        marked["Diversity_Shortlist"] = 0

    facet_cols = ["Facet_" + f for f in FACET_NAMES]
    id_cols = ["Filename", "Brand", "Eligibility", "Researcher_Note"]
    facet_export = marked[[c for c in id_cols if c in marked.columns] + facet_cols].copy()

    profile_cols = [
        "Filename", "Brand", "Eligibility", "Researcher_Note",
        "Primary_Dimension", "Mean_Profile_Distance", "Diversity_Rank",
        "PCA1", "PCA2", "Diversity_Shortlist",
    ] + DIMENSIONS
    profile_export = marked[[c for c in profile_cols if c in marked.columns]].copy()

    prompt_df = pd.DataFrame(
        [(d, f, PROMPT_TEMPLATE.format(f.lower())) for d, fs in FACETS.items() for f in fs],
        columns=["Dimension", "Facet", "FontCLIP_Prompt"],
    )

    meta_df = pd.DataFrame([{
        "App_Version": APP_VERSION,
        **R["meta"],
        "N_Logos": len(results),
        "N_Shortlist": len(shortlist),
        "Shortlist_Pool": eligible_option,
        "Facet_Aggregation": "Arithmetic mean of Aaker facets within each dimension",
        "Primary_Metric": "Raw image-text cosine similarity",
        "Diversity_Method": "5D profile cosine distance + deterministic maximin",
    }])

    # Matrix form is useful for inspection; long form is easier for statistics.
    distance_matrix = R["distance"].copy()
    distance_matrix.index.name = "Brand"
    distance_export = distance_matrix.reset_index()
    distance_long = (
        distance_matrix
        .rename_axis(index="Brand_A", columns="Brand_B")
        .stack()
        .rename("Cosine_Distance")
        .reset_index()
    )

    files = {
        "03A_01_logo_standardization_review.csv": R["review"],
        "03A_02_fontclip_all_analysis_results.csv": marked,
        "03A_03_fontclip_15facet_scores.csv": facet_export,
        "03A_04_fontclip_5D_profiles.csv": profile_export,
        "03A_05_fontclip_pairwise_distance_matrix.csv": distance_export,
        "03A_06_fontclip_pairwise_distance_long.csv": distance_long,
        "03A_07_fontclip_diversity_shortlist.csv": shortlist,
        "03A_08_fontclip_prompt_definition.csv": prompt_df,
        "03A_09_fontclip_run_metadata.csv": meta_df,
    }

    st.subheader("6. 모든 분석결과 CSV 다운로드")
    st.caption(
        "전체 점수, 15 facet, 5차원 프로파일, pairwise distance, shortlist, prompt 정의, 실행 메타데이터를 "
        "각각 CSV로 내려받을 수 있습니다."
    )

    st.download_button(
        "📦 모든 CSV를 ZIP으로 한 번에 다운로드",
        data=zip_csv(files),
        file_name="03A_fontclip_logo_prescreen_all_csv_v1_1.zip",
        mime="application/zip",
        type="primary",
    )

    file_items = list(files.items())
    for i in range(0, len(file_items), 3):
        cols = st.columns(3)
        for col, (name, df) in zip(cols, file_items[i:i+3]):
            with col:
                st.download_button(
                    name.replace(".csv", ""),
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=name,
                    mime="text/csv",
                    key=f"download_{name}",
                    use_container_width=True,
                )

    with st.expander("실행 재현정보"):
        st.json(R["meta"])

st.divider()
st.caption(
    "해석 주의: FontCLIP은 브랜드 개성 척도가 아니라 서체 이미지와 언어적 속성의 의미 관계를 계산하는 타이포그래피 특화 시각-언어 모델입니다. "
    "본 앱의 Aaker 15 facets 적용은 후보 로고타입의 상대적 프로파일 분포를 사전 진단하기 위한 연구 조작화입니다."
)
