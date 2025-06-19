import streamlit as st
import gdown
import os
import numpy as np
import cv2
from PIL import Image
import pathlib
from fastai.vision.all import load_learner, PILImage
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from collections import Counter

import pathlib
import os

# --- Patch agar PosixPath di Linux bisa baca WindowsPath dari file.pkl ---
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath


# --- Fungsi Preprocessing ---
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# --- Unduh model dari Google Drive jika belum ada ---
model_path = "modelbaru.pkl"
drive_file_id = "1fi9zbYODAe007Bf020UoNaQVTFqLolMu"
url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(url, model_path, quiet=False)
# --- Load Model ---
model = load_learner(model_path)
labels = model.dls.vocab

# --- Load Haar Cascade Classifier ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Setup Halaman ---
st.set_page_config(page_title="VISMO : VISUAL EMOTION MONITOR", layout="wide")

# --- Tema warna custom ---
st.markdown("""
    <style>
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #001f3f; /* Navy blue */
    }

    /* Warna teks di sidebar */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* ===== BACKGROUND HALAMAN UTAMA ===== */
    div[data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #B0E0E6, #FFFACD);
    }

    /* Kontainer isi utama transparan agar gradasi terlihat */
    div.block-container {
        background-color: rgba(255,255,255,0.0) !important;
    }

    /* ====== UPLOAD GAMBAR BUTTON ====== */
    .stFileUploader > div > div {
        background-color: #f0f0f0 !important;  /* abu-abu muda */
        border-radius: 8px;
        padding: 0.5em 1em;
        border: 1px solid #ccc;
        color: #333 !important;
        font-weight: 500;
    }

    .stFileUploader > div > div:hover {
        background-color: #e0e0e0 !important;
        cursor: pointer;
    }

    /* ====== TOMBOL LAIN (st.button, st.radio, dst) ====== */
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }

    /* Radio button label */
    .stRadio > div {
        color: #0F172A !important;
        font-weight: 500;
    }

    /* Heading dan teks agar lebih kontras di gradasi */
    .stTitle, .stHeader, .stSubheader, .stMarkdown, .stCaption {
        color: #0F172A !important;
    }

    /* Bar chart label color fix */
    .stPlotlyChart, .stAltairChart {
        color: black !important;
    }
            
    video {
        max-width: 360px !important;
        height: auto !important;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    </style>
""", unsafe_allow_html=True)


# --- Session state halaman ---
if "page" not in st.session_state:
    st.session_state.page = "home"

# --- Sidebar navigasi ---
st.sidebar.title("📋 Menu")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"
if st.sidebar.button("📷 Face Detection"):
    st.session_state.page = "detect"
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 • VISMO : VISUAL EMOTION MONITOR")

# =======================
# === HALAMAN: HOME ===
# =======================
if st.session_state.page == "home":
    st.title("Selamat Datang di Aplikasi VISMO : VISUAL EMOTION MONITOR 👋")
    st.markdown("""
    ### 👩‍⚕️ Tentang Aplikasi
    Aplikasi ini dirancang untuk membantu konselor dalam mengenali emosi pasien secara otomatis melalui ekspresi wajah.

    **Fitur utama:**
    - Deteksi emosi dari gambar statis
    - Deteksi emosi melalui kamera secara real-time (multi-wajah)
    - Rekomendasi penanganan berdasarkan ekspresi

    Gunakan sidebar untuk berpindah ke halaman **Face Detection**.
    """)

# ================================
# === HALAMAN: FACE DETECTION ===
# ================================
elif st.session_state.page == "detect":
    st.title("📷 Deteksi Emosi dari Wajah")
    mode = st.radio("Pilih metode deteksi:", ["🖼️ Upload Foto", "📷 Kamera Real-time"])

    # === Upload Foto ===
    if mode == "🖼️ Upload Foto":
        uploaded = st.file_uploader("Unggah foto", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Hasil Apload Foto", width=480)
            pred, _, probs = model.predict(PILImage.create(img))
            st.subheader(f"📊 Monitoring Emosi")
            st.bar_chart({label: float(prob) for label, prob in zip(labels, probs)})

            # Tampilkan persentase dalam tabel atau teks
            st.markdown("### 🎯 Persentase Emosi:")
            for label, prob in zip(labels, probs):
                st.markdown(f"- **{label.capitalize()}**: {prob*100:.2f}%")

            st.markdown("### 🚀 Rekomendasi:")
            if str(pred) == 'sad':
                st.warning("😢Emosi: Sedih (Sad)")
                st.markdown("""
                **1. Coping skills for depression**
                such as simple self-care, talking to a trusted person, or doing light
                activities are recommended in this resource from GoodRx
                (https://www.goodrx.com/conditions/depression/coping-skills-for-depression?srsltid=AfmBOopQo0qChBbXCq6iQTXZuFKk_d4YDvpErR4hlzVMvaw2YbHL6JjF&).  

                **2. “Three Good Things” Journal:**
                Writing down three good things every day improves emotional
                well-being for up to 6 months (https://en.wikipedia.org/wiki/Gratitude_journal?).

                **3. Savoring (stretching positive moments):**     
                such as enjoying accomplishments or good moments,
                has been shown to expand emotional well-being and creativity, according to Fredrickson's theory.(https://en.wikipedia.org/wiki/Savoring?).

                **4. Visual gratitude diary:**    
                Combining images and words to record happy moments strengthens the
                positive effects neurologically (https://positivepsychology.com/neuroscience-of-gratitude/?).

                **5. Daily micro-interactions:** 
                Small talk and warm greetings from strangers increase life satisfaction
                and positive mood (https://community.macmillanlearning.com/t5/talk-psych-blog/the-happy-science-of-micro-friendships/ba-p/13993?).                                         
                """)

            elif str(pred) == 'angry':
                st.warning("😡Emosi: Marah (Angry)")
                st.markdown("""
                **1. Anger journaling:**
                : Writing down triggers & responses to anger improves self-awareness and
                emotional regulation (https://en.wikipedia.org/wiki/Anger_management?utm).

                **2. Positive reframing:**
                A method of changing perspective to improve emotional management and
                positive thinking patterns (https://www.researchgate.net/publication/340315129_Not_all_disengagement_coping_strategies_are_created_equal_Positive_distraction_but_not_avoidance_can_be_an_adaptive_coping_strategy_for_chronic_life_stressors?utm).

                **3. CBT prompts untuk kemarahan:**
                Objectively evaluating the causes of anger helps to reduce the
                intensity (https://medium.com/%40mnwieschalla/journaling-prompts-for-dealing-with-anger-a-cbt-based-solution-66bedc1dc6b9).

                **4. Light physical exercise:**
                Athletic activity or walking helps release energy and relieve anger (https://en.wikipedia.org/wiki/Emotional_self-regulation?).

                **5. Therapeutic emotion journaling:**
                Expressive writing helps process strong emotions like anger
                safely (https://positivepsychology.com/benefits-of-journaling/?).                                                                                                                                                                                           
                """)
            elif str(pred) == 'happy':
                st.success("😊Emosi: Senang (Happy)")
                st.markdown("""
                **1. Micro-acts of joy**  
                Small actions like writing down gratitude, taking a short walk, or a daily act of
                kindness for 5–10 minutes. A UCSF study shows these activities can increase happiness and
                decrease stress in just a week (https://www.sfchronicle.com/health/article/joy-mood-life-health-20372907.php?) 

                **2. “Three Good Things” Journal**  
                Writing down three good things every day improves emotional
                well-being for up to 6 months (https://en.wikipedia.org/wiki/Gratitude_journal?).

                **3. Savoring (stretching positive moments)**  
                Such as enjoying accomplishments or good moments,
                has been shown to expand emotional well-being and creativity, according to Fredrickson's theory.(https://en.wikipedia.org/wiki/Savoring?).

                **4. Visual gratitude diary**  
                Combining images and words to record happy moments strengthens the
                positive effects neurologically (https://positivepsychology.com/neuroscience-of-gratitude/?).

                **5. Daily micro-interactions**  
                Small talk and warm greetings from strangers increase life satisfaction
                and positive mood (https://community.macmillanlearning.com/t5/talk-psych-blog/the-happy-science-of-micro-friendships/ba-p/13993?). """)

            elif str(pred) == 'fear':
                st.warning("😨Emosi: Takut (Fear)")
                st.markdown("""
                **1. Mindfulness meditation:**
                An 8-week MBSR program changes the brain's response to fear &
                improves emotional regulation (https://positivepsychology.com/mindfulness-based-stress- reduction-mbsr/)                            
                **2. Deep breathing / SKY technique:**
                Deep breathing techniques help calm the nervous system during panic (https://time.com/6244576/deep-breathing-better-well-being/?).                            
                **3. Positive distraction dengan hobi:**
                Diverting your mind to enjoyable activities is effective in
                reducing anxiety (https://pmc.ncbi.nlm.nih.gov/articles/PMC3122271/?).                         
                **4. Positive visualization expression:**
                Autobiographical visualization exercises have a significant effect on improving mood (https://pmc.ncbi.nlm.nih.gov/articles/PMC6532958/?).                         
                **5. Yoga & guided breathing:** 
                Yoga and breathing practices reduce anxiety and stress (including in
                college students (https://pmc.ncbi.nlm.nih.gov/articles/PMC6491852/).                                                                                                                                   
                """)

            elif str(pred) == 'suprised':
                st.info("😲Emosi: Terkejut (Surprised)")
                st.markdown("""
                **1. Savoring & awe moments:**
                When you experience a “wow” moment, take the time to capture and
                share it—according to the broaden-and-build theory (https://en.wikipedia.org/wiki/Broaden-and-build?).                            
                **2. Regular awe walks:**
                Regular walks in search of natural wonders increase gratitude & social
                connection (https://www.ucsf.edu/news/2020/09/418551/awe-walks-boost-emotional-well-being?).                            
                **3. Sharing awe experiences:**
                Discussing moments of awe with others amplifies their positive effect
                (https://ggsc.berkeley.edu/images/uploads/GGSC-JTF_White_Paper-Awe_FINAL.pdf?).                            
                **4. Reflection meditation on awe:**
                Deep reflection on the experience of surprise increases
                self-compassion (https://link.springer.com/article/10.1007/s12144-025-07706-1?).                       
                **5. Awe journaling:**  
                Writing down reactions to overwhelming experiences has been shown to deepen the positive effects of the emotion of surprise (https://www.alodokter.com/manfaat-journaling-bagi-kesehatan-mental-yang-sayang-untuk-dilewatkan).                                                                                                                              
                """)

            elif str(pred) == 'neutral':
                st.info("😐Emosi: Netral (Neutral)")
                st.markdown("""
                **1. Presence pausing / mindfulness break:**
                Taking a moment to be aware of what is happening now
                without judgment increases self-awareness and simple joy
                (https://www.realsimple.com/dopamine-boosting-wellness-rituals-11753500?).                            
                **2. Boundary gratitude / gratitude weekly check-in:**
                Regular reflection on personal values and
                boundaries helps build emotional well-being (https://www.realsimple.com/dopamine-boosting-wellness-rituals-11753500?).                            
                **3. Social micro-interactions:**
                A warm greeting or brief interaction improves mood even in neutral
                conditions (https://www.mother.ly/life/micro-interactions-increase-life-satisfactionso-go-ahead-and-talk-to-strangers/?).                            
                **4. Gratitude weekly check-in:**
                Keeping a weekly gratitude journal improves gratitude and mood
                (https://www.verywellhealth.com/benefits-of-yoga-11685529?).                            
                **5. Connect with nature & social micro-interactions:** 
                Short chats/nature views can improve your
                mood even in a neutral state (https://www.nature.com/articles/s44159-025-00455-9).                                                                                                                                           
                """)



    # === Kamera Real-time Multi-Face ===
    elif mode == "📷 Kamera Real-time":
        class EmotionDetector(VideoTransformerBase):
            def __init__(self):
                self.emotion_seq = []
                self.prob_seq = []

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:  
                    face_img = img[y:y+h, x:x+w]
                    try:
                        img_processed = preprocess_image(face_img)
                        pred, _, probs = model.predict(PILImage.create(img_processed))
                        self.emotion_seq.append(str(pred))
                        self.prob_seq.append(probs.numpy())
                        label = f"{pred}"
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    except Exception as e:
                        continue

                return img

        ctx = webrtc_streamer(
            key="realtime",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=EmotionDetector,
            media_stream_constraints={"video": True, "audio": False},
        )

        if st.button("⏸️ Selesai Sesi") and ctx.video_transformer:
            all_preds = ctx.video_transformer.emotion_seq
            all_probs = ctx.video_transformer.prob_seq
            counter = Counter(all_preds)
            most_common = counter.most_common(3)

            if all_probs:
                avg_probs = np.mean(all_probs, axis=0)
                st.subheader("📈 Rata-rata Probabilitas Emosi:")
                st.bar_chart({label: float(prob) for label, prob in zip(labels, avg_probs)})

            st.markdown("### 🎯 Rekomendasi Aktivitas:")
            if "sad" in [e for e, _ in most_common]:
                st.warning("😢Emosi: Sedih (Sad)")
                st.markdown("""
                **1. Coping skills for depression**
                such as simple self-care, talking to a trusted person, or doing light
                activities are recommended in this resource from GoodRx
                (https://www.goodrx.com/conditions/depression/coping-skills-for-depression?srsltid=AfmBOopQo0qChBbXCq6iQTXZuFKk_d4YDvpErR4hlzVMvaw2YbHL6JjF&).  

                **2. “Three Good Things” Journal:**
                Writing down three good things every day improves emotional
                well-being for up to 6 months (https://en.wikipedia.org/wiki/Gratitude_journal?).

                **3. Savoring (stretching positive moments):**     
                such as enjoying accomplishments or good moments,
                has been shown to expand emotional well-being and creativity, according to Fredrickson's theory.(https://en.wikipedia.org/wiki/Savoring?).

                **4. Visual gratitude diary:**    
                Combining images and words to record happy moments strengthens the
                positive effects neurologically (https://positivepsychology.com/neuroscience-of-gratitude/?).

                **5. Daily micro-interactions:** 
                Small talk and warm greetings from strangers increase life satisfaction
                and positive mood (https://community.macmillanlearning.com/t5/talk-psych-blog/the-happy-science-of-micro-friendships/ba-p/13993?).                                         
                """)

            if "angry" in [e for e, _ in most_common]:
                st.warning("😡Emosi: Marah (Angry)")
                st.markdown("""
                **1. Anger journaling:**
                : Writing down triggers & responses to anger improves self-awareness and
                emotional regulation (https://en.wikipedia.org/wiki/Anger_management?utm).

                **2. Positive reframing:**
                A method of changing perspective to improve emotional management and
                positive thinking patterns (https://www.researchgate.net/publication/340315129_Not_all_disengagement_coping_strategies_are_created_equal_Positive_distraction_but_not_avoidance_can_be_an_adaptive_coping_strategy_for_chronic_life_stressors?utm).

                **3. CBT prompts untuk kemarahan:**
                Objectively evaluating the causes of anger helps to reduce the
                intensity (https://medium.com/%40mnwieschalla/journaling-prompts-for-dealing-with-anger-a-cbt-based-solution-66bedc1dc6b9).

                **4. Light physical exercise:**
                Athletic activity or walking helps release energy and relieve anger (https://en.wikipedia.org/wiki/Emotional_self-regulation?).

                **5. Therapeutic emotion journaling:**
                Expressive writing helps process strong emotions like anger
                safely (https://positivepsychology.com/benefits-of-journaling/?).                                                                                                                                                                                           
                """)

            if "happy" in [e for e, _ in most_common]:
                st.success("😊Emosi: Senang (Happy)")
                st.markdown("""
                **1. Micro-acts of joy**  
                Small actions like writing down gratitude, taking a short walk, or a daily act of
                kindness for 5–10 minutes. A UCSF study shows these activities can increase happiness and
                decrease stress in just a week (https://www.sfchronicle.com/health/article/joy-mood-life-health-20372907.php?) 

                **2. “Three Good Things” Journal**  
                Writing down three good things every day improves emotional
                well-being for up to 6 months (https://en.wikipedia.org/wiki/Gratitude_journal?).

                **3. Savoring (stretching positive moments)**  
                Such as enjoying accomplishments or good moments,
                has been shown to expand emotional well-being and creativity, according to Fredrickson's theory.(https://en.wikipedia.org/wiki/Savoring?).

                **4. Visual gratitude diary**  
                Combining images and words to record happy moments strengthens the
                positive effects neurologically (https://positivepsychology.com/neuroscience-of-gratitude/?).

                **5. Daily micro-interactions**  
                Small talk and warm greetings from strangers increase life satisfaction
                and positive mood (https://community.macmillanlearning.com/t5/talk-psych-blog/the-happy-science-of-micro-friendships/ba-p/13993?). """)

            if "neutral" in [e for e, _ in most_common]:
                st.info("😐Emosi: Netral (Neutral)")
                st.markdown("""
                **1. Presence pausing / mindfulness break:**
                Taking a moment to be aware of what is happening now
                without judgment increases self-awareness and simple joy
                (https://www.realsimple.com/dopamine-boosting-wellness-rituals-11753500?).                            
                **2. Boundary gratitude / gratitude weekly check-in:**
                Regular reflection on personal values and
                boundaries helps build emotional well-being (https://www.realsimple.com/dopamine-boosting-wellness-rituals-11753500?).                            
                **3. Social micro-interactions:**
                A warm greeting or brief interaction improves mood even in neutral
                conditions (https://www.mother.ly/life/micro-interactions-increase-life-satisfactionso-go-ahead-and-talk-to-strangers/?).                            
                **4. Gratitude weekly check-in:**
                Keeping a weekly gratitude journal improves gratitude and mood
                (https://www.verywellhealth.com/benefits-of-yoga-11685529?).                            
                **5. Connect with nature & social micro-interactions:** 
                Short chats/nature views can improve your
                mood even in a neutral state (https://www.nature.com/articles/s44159-025-00455-9).                                                                                                                                           
                """)

            if "suprised" in [e for e, _ in most_common]:
                st.info("😲Emosi: Terkejut (Surprised)")
                st.markdown("""
                **1. Savoring & awe moments:**
                When you experience a “wow” moment, take the time to capture and
                share it—according to the broaden-and-build theory (https://en.wikipedia.org/wiki/Broaden-and-build?).                            
                **2. Regular awe walks:**
                Regular walks in search of natural wonders increase gratitude & social
                connection (https://www.ucsf.edu/news/2020/09/418551/awe-walks-boost-emotional-well-being?).                            
                **3. Sharing awe experiences:**
                Discussing moments of awe with others amplifies their positive effect
                (https://ggsc.berkeley.edu/images/uploads/GGSC-JTF_White_Paper-Awe_FINAL.pdf?).                            
                **4. Reflection meditation on awe:**
                Deep reflection on the experience of surprise increases
                self-compassion (https://link.springer.com/article/10.1007/s12144-025-07706-1?).                       
                **5. Awe journaling:**  
                Writing down reactions to overwhelming experiences has been shown to deepen the positive effects of the emotion of surprise (https://www.alodokter.com/manfaat-journaling-bagi-kesehatan-mental-yang-sayang-untuk-dilewatkan).                                                                                                                              
                """)

            if "fear" in [e for e, _ in most_common]:
                st.warning("😨Emosi: Takut (Fear)")
                st.markdown("""
                **1. Mindfulness meditation:**
                An 8-week MBSR program changes the brain's response to fear &
                improves emotional regulation (https://positivepsychology.com/mindfulness-based-stress- reduction-mbsr/)                            
                **2. Deep breathing / SKY technique:**
                Deep breathing techniques help calm the nervous system during panic (https://time.com/6244576/deep-breathing-better-well-being/?).                            
                **3. Positive distraction dengan hobi:**
                Diverting your mind to enjoyable activities is effective in
                reducing anxiety (https://pmc.ncbi.nlm.nih.gov/articles/PMC3122271/?).                         
                **4. Positive visualization expression:**
                Autobiographical visualization exercises have a significant effect on improving mood (https://pmc.ncbi.nlm.nih.gov/articles/PMC6532958/?).                         
                **5. Yoga & guided breathing:** 
                Yoga and breathing practices reduce anxiety and stress (including in
                college students (https://pmc.ncbi.nlm.nih.gov/articles/PMC6491852/).                                                                                                                                   
                """)

