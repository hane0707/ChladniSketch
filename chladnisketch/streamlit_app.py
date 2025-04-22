import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# タイトル表示用のアプリ名
APP_NAME = "ChladniSketch"


# 1. AudioInput
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.samples = np.array([])

    def recv(self, frame):
        # 変更: フレームごとに音声データをバッファに蓄積
        audio = frame.to_ndarray().flatten()
        self.samples = np.concatenate((self.samples, audio))
        return frame


def get_audio(file=None, live=False, processor=None):
    if file:
        # ファイルアップロードモード
        y, sr = librosa.load(file, sr=None)
        return y, sr
    elif live and processor:
        # マイク入力モード
        # 変更: webrtc_streamer で AudioProcessor を利用
        webrtc_ctx = webrtc_streamer(
            key="mic", mode="SENDONLY", audio_processor_factory=AudioProcessor
        )
        if webrtc_ctx.audio_processor:
            processor = webrtc_ctx.audio_processor
            # 処理終了後に取得 (簡易例)
            y = processor.samples
            sr = 48000  # 変更: WebRTC のデフォルトサンプリングレート
            return y, sr
        return None, None
    else:
        return None, None


# 2. Analyzer
def analyze_frequency(y, sr):
    # STFT
    stft_result = librosa.stft(y)
    magnitude = np.abs(stft_result)
    dominant_idx = np.argmax(magnitude, axis=0)
    freqs = librosa.fft_frequencies(sr=sr)
    try:
        idx = np.bincount(dominant_idx).argmax()
        dominant_freq = freqs[idx]
    except ValueError:
        dominant_freq = 440.0  # デフォルト

    # パラメータマッピング
    m = int(dominant_freq / 150) % 6 + 1
    n = int(dominant_freq / 100) % 6 + 1
    if m == n:
        n = (n % 6) + 1

    return dominant_freq, m, n


# 3. Visualizer
def draw_cladogram(m, n, L=1.0):
    x = np.linspace(0, L, 300)
    y = np.linspace(0, L, 300)
    X, Y = np.meshgrid(x, y)
    term1 = np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L)
    term2 = np.sin(n * np.pi * X / L) * np.sin(m * np.pi * Y / L)
    Z = term1 + term2 if (m + n) % 2 == 0 else term1 - term2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(X, Y, Z, levels=[0], linewidths=2)
    ax.axis("equal")
    ax.axis("off")
    return fig


# Streamlit UI


def main():
    st.set_page_config(page_title=APP_NAME)
    st.title(APP_NAME)
    mode = st.sidebar.radio("入力モードを選択", ("ファイルアップロード", "マイク入力"))
    y, sr = None, None

    if mode == "ファイルアップロード":
        uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3"])
        if uploaded:
            # 変更: bytesIO を librosa に直接渡す
            y, sr = get_audio(file=uploaded)
    else:
        st.info("マイク入力モード：録音開始ボタンを押してください")
        if st.button("録音開始"):  # 変更: ボタンで録音開始
            y, sr = get_audio(live=True)

    if st.button("処理実行"):
        if y is not None:
            freq, m, n = analyze_frequency(y, sr)
            st.write(f"推定ドミナント周波数: {freq:.2f} Hz")
            st.write(f"マッピングパラメータ: m={m}, n={n}")
            fig = draw_cladogram(m, n)
            st.pyplot(fig)
        else:
            st.error(
                "音声データが取得できませんでした。モードと入力を確認してください。"
            )


if __name__ == "__main__":
    main()
