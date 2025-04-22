import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# アプリ名
APP_NAME = "ChladniSketch"


class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.samples = np.array([])

    def recv(self, frame):
        self.samples = np.concatenate((self.samples, frame.to_ndarray().flatten()))
        return frame


def get_audio(file=None, live=False):
    if file:
        return librosa.load(file, sr=None)
    elif live:
        ctx = webrtc_streamer(
            key="mic", mode="SENDONLY", audio_processor_factory=AudioProcessor
        )
        if ctx.audio_processor:
            return ctx.audio_processor.samples, 48000
    return None, None


def analyze_frequency(y, sr, n_fft, hop_length, m_div, n_div):
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    idx = np.bincount(np.argmax(stft, axis=0)).argmax() if stft.size else None
    freq = librosa.fft_frequencies(sr=sr)[idx] if idx is not None else 440.0
    m = int(freq / m_div) % 6 + 1
    n = int(freq / n_div) % 6 + 1
    if m == n:
        n = n % 6 + 1
    return freq, m, n


def draw_cladogram(m, n, L=1.0):
    X, Y = np.meshgrid(np.linspace(0, L, 300), np.linspace(0, L, 300))
    Z = np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L) + np.sin(
        n * np.pi * X / L
    ) * np.sin(m * np.pi * Y / L)
    Z = (
        Z
        if (m + n) % 2 == 0
        else (
            np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L)
            - np.sin(n * np.pi * X / L) * np.sin(m * np.pi * Y / L)
        )
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(X, Y, Z, levels=[0], linewidths=2)
    ax.axis("off")
    return fig


def main():
    st.set_page_config(page_title=APP_NAME)
    st.title(APP_NAME)

    # 使い方説明
    st.markdown(
        """
        **使い方:**
        1. 入力モードを選び、音声ファイルをアップロードまたは「録音開始」を押す
        2. スライダーで「音の細かさ」「時間感度」「模様の変化度」「線のゆらぎ」を調整
        3. 「処理実行」をクリックし、あなたの音が生むクラドニ図形を楽しむ
        """
    )

    st.sidebar.header("入力モード")
    mode = st.sidebar.radio("", ("ファイルアップロード", "マイク入力"))

    # パラメータ設定（シンプルなラベル）
    st.sidebar.header("パターン設定")
    n_fft = st.sidebar.slider("音の細かさ (大まか→詳細)", 256, 4096, 2048, step=256)
    hop_length = st.sidebar.slider("時間感度 (ざっくり→細かく)", 64, 1024, 512, step=64)
    m_div = st.sidebar.slider("模様の変化度 (小→大)", 50, 300, 150)
    n_div = st.sidebar.slider("線のゆらぎ (小→大)", 50, 300, 100)

    y, sr = None, None

    if mode == "ファイルアップロード":
        uploaded = st.file_uploader("音声ファイル", type=["wav", "mp3"])
        if uploaded:
            y, sr = get_audio(file=uploaded)
    else:
        st.info("マイク入力モード: 録音開始ボタンを押してください")
        if st.button("録音開始"):
            y, sr = get_audio(live=True)

    if st.button("処理実行"):
        if y is not None:
            freq, m, n = analyze_frequency(y, sr, n_fft, hop_length, m_div, n_div)
            st.write(f"**ドミナント周波数:** {freq:.0f} Hz")
            st.write(f"**m, n:** {m}, {n}")
            st.pyplot(draw_cladogram(m, n))
        else:
            st.error("音声データがありません。モードと入力を確認してください。")


if __name__ == "__main__":
    main()
