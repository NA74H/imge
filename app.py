import streamlit as st
from PIL import Image
import numpy as np
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Matrix Processing Studio",
    page_icon="🎨",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stFileUploader label {
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #a0aec0 !important;
}

/* Main background */
.main .block-container {
    background: #f0f4f8;
    padding-top: 2rem;
}

/* Header */
.studio-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    text-align: center;
    color: #1a1a2e;
    margin-bottom: 0.25rem;
    letter-spacing: -1px;
}
.studio-sub {
    text-align: center;
    color: #718096;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Image cards */
.img-card {
    background: white;
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5568;
    margin-bottom: 0.75rem;
}

/* Note box */
.note-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.75rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.7;
    margin-top: 1.5rem;
    border-left: 4px solid #667eea;
}

/* Sidebar button */
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    width: 100%;
    transition: opacity 0.2s;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.88;
}

/* Welcome */
.welcome {
    text-align: center;
    padding: 5rem 2rem;
    color: #718096;
}
.welcome-icon { font-size: 5rem; }
.welcome h3 { color: #4a5568; font-size: 1.4rem; margin: 1rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────────────────────
def to_float_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0

def from_float_rgb(A: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(A * 255, 0, 255)), mode="RGB")

def conv2d_rgb(A: np.ndarray, K: np.ndarray) -> np.ndarray:
    kh, kw = K.shape
    pad = kh // 2
    Ah, Aw, _ = A.shape
    B = np.zeros_like(A)
    for ch in range(3):
        for i in range(pad, Ah - pad):
            for j in range(pad, Aw - pad):
                region = A[i - pad:i + pad + 1, j - pad:j + pad + 1, ch]
                B[i, j, ch] = np.sum(region * K)
    return np.clip(B, 0, 1)

def svd_compress_rgb(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    B = np.zeros_like(A)
    # Capture singular values from the red channel (index 0) up front
    _, s_vals, _ = np.linalg.svd(A[:, :, 0], full_matrices=False)
    for ch in range(3):
        U, S, Vt = np.linalg.svd(A[:, :, ch], full_matrices=False)
        B[:, :, ch] = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return np.clip(B, 0, 1), s_vals

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Operations Panel")
    st.markdown("---")

    uploaded = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg"])

    operation = st.selectbox("🔧 Operation", options={
        "view": "View (no change)",
        "add": "Add Constant",
        "scale": "Scale (Brightness)",
        "transpose": "Transpose",
        "conv": "Convolution (3×3)",
        "svd": "SVD Compression",
    })

    kernel = st.selectbox("🎛️ Convolution Kernel", ["Sharpen", "Blur", "Edge Detection", "Emboss"])

    c_val  = st.number_input("➕ Add Constant (c)", value=0.0, step=0.1, format="%.2f")
    alpha  = st.number_input("🔆 Scale Factor (α)", value=1.0, step=0.1, format="%.2f")
    k_rank = st.number_input("📊 SVD Rank (k)", value=20, min_value=1, step=1)

    apply = st.button("✨ Apply Operation")


# ── Main area ───────────────────────────────────────────────────────────────────
st.markdown('<div class="studio-header">🎨 Image Matrix Processing Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="studio-sub">Explore linear algebra on images — convolutions, SVD compression, and more</div>', unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-icon">🖼️</div>
        <h3>Upload an image to get started</h3>
        <p>Use the sidebar to upload a photo and choose a matrix operation.<br>
        Supports PNG, JPG, and JPEG files.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Load image
    img = Image.open(uploaded).convert("RGB")
    A   = to_float_rgb(img)

    # Process
    note = ""
    if True:   # always show result after upload
        op = operation

        if op == "view":
            B    = A.copy()
            note = "Displayed raw colour image — no transformation applied."

        elif op == "add":
            B    = np.clip(A + c_val, 0, 1)
            note = f"A' = A + {c_val:.2f}  (pixel-wise constant addition)"

        elif op == "scale":
            B    = np.clip(alpha * A, 0, 1)
            note = f"A' = α · A  with  α = {alpha:.2f}  (brightness scaling)"

        elif op == "transpose":
            B    = np.transpose(A, (1, 0, 2))
            note = "A' = Aᵀ  — each colour channel transposed independently."

        elif op == "conv":
            kernels = {
                "Sharpen":       np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
                "Blur":          np.ones((3,3), dtype=float) / 9.0,
                "Edge Detection":np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float),
                "Emboss":        np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=float),
            }
            K    = kernels[kernel]
            with st.spinner(f"Applying {kernel} convolution… (may take a moment on large images)"):
                B = conv2d_rgb(A, K)
            note = f"Applied {kernel} convolution kernel (3×3)."

        elif op == "svd":
            k = min(int(k_rank), min(A.shape[:2]))
            with st.spinner(f"Running SVD with rank k={k}…"):
                B, s = svd_compress_rgb(A, k)
            top = ", ".join(f"{v:.3f}" for v in s[:8])
            note = f"SVD compression · rank k = {k}\nTop singular values: [{top}, …]"

        else:
            B    = A.copy()
            note = "Unknown operation."

        # Display side-by-side
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="img-card"><div class="img-label">📷 Original Image</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="img-card"><div class="img-label">✨ Processed Image</div>', unsafe_allow_html=True)
            processed_img = from_float_rgb(B)
            st.image(processed_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Download button
            st.download_button(
                label="⬇️ Download Result",
                data=pil_to_bytes(processed_img),
                file_name="processed.png",
                mime="image/png",
            )

        # Operation note
        st.markdown(f'<div class="note-box">📝 <strong>Operation Details</strong><br><br>{note}</div>', unsafe_allow_html=True)