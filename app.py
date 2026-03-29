from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import io, base64, os

app = Flask(__name__)
app.secret_key = "secret123"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def to_float_rgb(img):
    return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0

def from_float_rgb(A):
    A8 = np.uint8(np.clip(A * 255, 0, 255))
    return Image.fromarray(A8, mode="RGB")

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def conv2d_rgb(A, K):
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

def svd_compress_rgb(A, k):
    B = np.zeros_like(A)
    for ch in range(3):
        U, S, Vt = np.linalg.svd(A[:, :, ch], full_matrices=False)
        Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        B[:, :, ch] = Ak
    return np.clip(B, 0, 1), S

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Matrix Processing Studio</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #2d3748;
        }

        .container {
            display: flex;
            min-height: 100vh;
            max-width: 1600px;
            margin: 0 auto;
            background: #fff;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        .sidebar {
            background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
            padding: 30px;
            width: 350px;
            color: #fff;
            overflow-y: auto;
            box-shadow: 4px 0 20px rgba(0,0,0,0.1);
        }

        .sidebar h2 {
            font-size: 28px;
            margin-bottom: 25px;
            color: #fff;
            text-align: center;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.2);
            letter-spacing: 1px;
        }

        .form-group {
            margin-bottom: 25px;
            animation: fadeInUp 0.5s ease-out;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 14px;
            color: #cbd5e0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input[type="file"] {
            width: 100%;
            padding: 12px;
            background: rgba(255,255,255,0.1);
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
        }

        input[type="file"]:hover {
            background: rgba(255,255,255,0.15);
            border-color: rgba(255,255,255,0.5);
        }

        input[type="file"]::file-selector-button {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            margin-right: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        input[type="file"]::file-selector-button:hover {
            background: #5a67d8;
            transform: translateY(-2px);
        }

        select, input[type="number"] {
            width: 100%;
            padding: 12px;
            background: rgba(255,255,255,0.95);
            border: 2px solid transparent;
            border-radius: 8px;
            font-size: 14px;
            color: #2d3748;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        select:focus, input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
            background: #fff;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }

        .btn {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 24px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102,126,234,0.4);
            margin-top: 10px;
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 25px rgba(102,126,234,0.6);
        }

        .btn:active {
            transform: translateY(-1px);
        }

        .content {
            flex: 1;
            padding: 50px;
            background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
            overflow-y: auto;
        }

        h1 {
            font-size: 36px;
            color: #2d3748;
            text-align: center;
            margin-bottom: 40px;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            letter-spacing: -0.5px;
        }

        .emoji {
            font-size: 42px;
            display: inline-block;
            animation: bounce 2s infinite;
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        .images-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            flex-wrap: wrap;
            margin: 30px 0;
        }

        .image-box {
            text-align: center;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }

        .image-box h3 {
            margin-bottom: 15px;
            color: #4a5568;
            font-size: 18px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .image-wrapper {
            background: white;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .image-wrapper::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .image-wrapper:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.25);
        }

        .image-wrapper:hover::before {
            opacity: 1;
        }

        img {
            max-width: 500px;
            width: 100%;
            height: auto;
            border-radius: 10px;
            display: block;
            position: relative;
        }

        .note {
            margin-top: 30px;
            padding: 20px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(102,126,234,0.3);
            font-size: 15px;
            line-height: 1.6;
            text-align: center;
            font-weight: 500;
            animation: slideUp 0.6s ease-out;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .welcome-message {
            text-align: center;
            padding: 80px 20px;
        }

        .welcome-message p {
            font-size: 20px;
            color: #718096;
            line-height: 1.8;
            max-width: 600px;
            margin: 0 auto;
        }

        .icon {
            font-size: 120px;
            margin-bottom: 30px;
            opacity: 0.8;
            animation: float 3s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
        }

        /* Scrollbar Styling */
        .sidebar::-webkit-scrollbar {
            width: 8px;
        }

        .sidebar::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.1);
        }

        .sidebar::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.3);
            border-radius: 4px;
        }

        .sidebar::-webkit-scrollbar-thumb:hover {
            background: rgba(255,255,255,0.5);
        }

        /* Responsive Design */
        @media (max-width: 1024px) {
            .container {
                flex-direction: column;
            }

            .sidebar {
                width: 100%;
            }

            .images-container {
                flex-direction: column;
            }

            img {
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="sidebar">
        <h2>⚙️ Operations Panel</h2>
        <form action="/process" method="POST" enctype="multipart/form-data">
            <div class="form-group">
                <label>📁 Upload Image:</label>
                <input type="file" name="image" accept=".jpg,.png,.jpeg" required>
            </div>

            <div class="form-group">
                <label>🔧 Operation:</label>
                <select name="operation">
                    <option value="view">View Matrix</option>
                    <option value="add">Add Constant</option>
                    <option value="scale">Scale (Brightness)</option>
                    <option value="transpose">Transpose</option>
                    <option value="conv">Convolution (3×3)</option>
                    <option value="svd">SVD Compression</option>
                </select>
            </div>

            <div class="form-group">
                <label>🎛️ Convolution Kernel:</label>
                <select name="kernel">
                    <option value="sharpen">Sharpen</option>
                    <option value="blur">Blur</option>
                    <option value="edge">Edge Detection</option>
                    <option value="emboss">Emboss</option>
                </select>
            </div>

            <div class="form-group">
                <label>➕ Add Constant (c):</label>
                <input type="number" step="0.1" name="c_val" value="0.0">
            </div>

            <div class="form-group">
                <label>🔆 Scale Factor (α):</label>
                <input type="number" step="0.1" name="alpha" value="1.0">
            </div>

            <div class="form-group">
                <label>📊 SVD Rank (k):</label>
                <input type="number" name="k_rank" value="20">
            </div>

            <button class="btn" type="submit">✨ Apply Operation</button>
        </form>
    </div>
    
    <div class="content">
        <h1><span class="emoji">🎨</span> Image Matrix Processing Studio</h1>
        {% if original_img %}
            <div class="images-container">
                <div class="image-box">
                    <h3>Original Image</h3>
                    <div class="image-wrapper">
                        <img src="data:image/png;base64,{{original_img}}" alt="Original">
                    </div>
                </div>
                <div class="image-box">
                    <h3>Processed Image</h3>
                    <div class="image-wrapper">
                        <img src="data:image/png;base64,{{processed_img}}" alt="Processed">
                    </div>
                </div>
            </div>
            <div class="note">
                <strong>📝 Operation Details:</strong><br>
                {{note}}
            </div>
        {% else %}
            <div class="welcome-message">
                <div class="icon">🖼️</div>
                <p>
                    Welcome to the Image Matrix Processing Studio!<br><br>
                    Upload an image to explore matrix operations including convolution filters, 
                    SVD compression, and various transformations.
                </p>
            </div>
        {% endif %}
    </div>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML, original_img=None, processed_img=None, note=None)

@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if not file or not file.filename or not allowed_file(file.filename):
        flash("Invalid file format", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = Image.open(path)
    A = to_float_rgb(img)

    op = request.form.get("operation", "view")
    kernel_preset = request.form.get("kernel", "sharpen")
    c_val = float(request.form.get("c_val", 0.0))
    alpha = float(request.form.get("alpha", 1.0))
    k_rank = int(request.form.get("k_rank", 20))

    if op == "view":
        B = A.copy()
        note = "Displayed raw color image (no change)."
    elif op == "add":
        B = np.clip(A + c_val, 0, 1)
        note = f"Applied A' = A + {c_val:.2f}"
    elif op == "scale":
        B = np.clip(alpha * A, 0, 1)
        note = f"Applied A' = αA with α={alpha:.2f}"
    elif op == "transpose":
        B = np.transpose(A, (1, 0, 2))
        note = "Applied A' = Aᵀ (transposed each color channel)."
    elif op == "conv":
        if kernel_preset == "sharpen":
            K = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        elif kernel_preset == "blur":
            K = np.ones((3,3))/9.0
        elif kernel_preset == "edge":
            K = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        elif kernel_preset == "emboss":
            K = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        else:
            K = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

        B = conv2d_rgb(A, K)
        note = f"Applied {kernel_preset.title()} convolution kernel."
    elif op == "svd":
        k = min(k_rank, min(A.shape[:2]))
        B, s = svd_compress_rgb(A, k)
        note = f"SVD compression with k={k}. Top singular values: {np.round(s[:10],3)}"
    else:
        B = A.copy()
        note = "Unknown operation."

    original_img = image_to_base64(img.convert("RGB"))
    processed_img = image_to_base64(from_float_rgb(B))
    return render_template_string(HTML, original_img=original_img, processed_img=processed_img, note=note)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    print("🚀 Running Flask Image Matrix Processing Studio")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)