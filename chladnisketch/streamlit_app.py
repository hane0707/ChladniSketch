import streamlit as st
import numpy as np
import io
import librosa
import librosa.effects
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# アプリ名
APP_NAME = "ChladniSketch"


def analyze_frequency(y, sr, n_fft, hop_length, m_div, n_div):
    y_trim, _ = librosa.effects.trim(y, top_db=20)
    y_proc = y_trim if y_trim.size > 0 else y
    S = np.abs(librosa.stft(y_proc, n_fft=n_fft, hop_length=hop_length))
    spectral_sum = S.sum(axis=1)
    idx = int(np.argmax(spectral_sum))
    freq = librosa.fft_frequencies(sr=sr)[idx]
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
    1. 音声をアップロードまたはマイク入力を選択
    2. 下のスライダーで模様の変化を調整
    3. [処理実行] ボタンで図形を生成・保存
    """
    )

    # 左ペイン: 入力設定とパターン設定を順に配置
    st.sidebar.header("入力設定")
    mode = st.sidebar.radio("入力モード", ("ファイルアップロード", "マイク入力"))

    st.sidebar.markdown("---")
    st.sidebar.header("パターン設定")
    n_fft = st.sidebar.slider("音の細かさ (高音の反映度)", 256, 4096, 2048, 256)
    hop_length = st.sidebar.slider("時間感度 (変化の検出)", 64, 1024, 512, 64)
    m_div = st.sidebar.slider("模様の変化度 (縦横の模様差)", 50, 300, 150)
    n_div = st.sidebar.slider("線のゆらぎ (複雑さ)", 50, 300, 100)

    # 入力処理
    y, sr = None, None
    if mode == "ファイルアップロード":
        uploaded = st.file_uploader("音声ファイル", type=["wav", "mp3"])
        if uploaded:
            y, sr = librosa.load(uploaded, sr=None)
    else:
        st.info(
            "マイク入力モード: 音声を録音して解析します（録音した音声は保存されません）"
        )
        audio_bytes = audio_recorder()
        if audio_bytes:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 実行ボタン
    if st.button("処理実行"):
        if y is not None:
            freq, m, n = analyze_frequency(y, sr, n_fft, hop_length, m_div, n_div)
            fig = draw_cladogram(m, n)
            st.session_state["last_result"] = {"freq": freq, "m": m, "n": n, "fig": fig}
        else:
            st.error("音声データがありません。")

    # 結果表示・保存
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        st.write(f"**ドミナント周波数:** {res['freq']:.0f} Hz")
        st.write(f"**m, n:** {res['m']}, {res['n']}")
        st.pyplot(res["fig"], clear_figure=False)
        buf = io.BytesIO()
        res["fig"].savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            "画像を保存", data=buf, file_name="cladogram.png", mime="image/png"
        )


if __name__ == "__main__":
    main()
