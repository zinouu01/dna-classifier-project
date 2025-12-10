import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. PAGE SETUP
st.set_page_config(page_title="Bio-Classifier", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧬 DNA Species Identifier</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'> Human vs Dog vs Cat </h5>", unsafe_allow_html=True)
st.write("---")

# 2. LOAD & TRAIN
@st.cache_data
def train_model():
    try:
        human = pd.read_table('human_data.txt')
        dog = pd.read_table('dog_data.txt')
        cat = pd.read_table('cat_data.txt')
    except FileNotFoundError:
        return None, None, 0

    human['label'] = 'Human'
    dog['label'] = 'Dog'
    cat['label'] = 'Cat'

    min_size = min(len(human), len(dog), len(cat))
    sample_size = min(min_size, 800)
    
    data = pd.concat([
        human.sample(sample_size, random_state=42),
        dog.sample(sample_size, random_state=42),
        cat.sample(sample_size, random_state=42)
    ])

    def getKmers(sequence, size=4):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

    data['words'] = data.apply(lambda x: getKmers(x['sequence']), axis=1)
    data_texts = list(data['words'])
    for item in range(len(data_texts)):
        data_texts[item] = ' '.join(data_texts[item])

    cv = CountVectorizer(ngram_range=(1,1))
    X = cv.fit_transform(data_texts)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, cv, accuracy

# 3. INTERFACE
with st.spinner('Initializing AI Model...'):
    model, vectorizer, acc = train_model()

if model is None:
    st.error("❌ Error: Files not found! Please check your directory.")
else:
    def getKmers(sequence, size=4):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

    # --- INPUT SECTION ---
    st.subheader("1. Enter DNA Sequence")
    default_seq = "ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCCATACTCCTTACACTATTCCTCATCACCCAACTAAAAATATTAAACACAACTACCACCTACCTCCCTCACCATTAG"
    sequence_input = st.text_area("Paste sequence here:", height=100, value=default_seq)

    # --- PREDICTION BUTTON ---
    if st.button("🔍 Identify Species", use_container_width=True):
        
        # 1. Clean input (remove spaces/newlines) and convert to Uppercase
        clean_seq = sequence_input.replace("\n", "").replace("\r", "").replace(" ", "").upper()
        
        # 2. VALIDATION CHECK 🛑
        valid_dna = set("ACGT")
        
        if not clean_seq:
            st.warning("⚠️ The input is empty. Please paste a DNA sequence.")
        
        # Check if the set of characters in input is a subset of {A,C,G,T}
        elif not set(clean_seq).issubset(valid_dna):
            st.error("❌ **Invalid Data!**\n\nDNA sequences can only contain the letters **A**, **C**, **G**, and **T**.\nPlease check your input for numbers or other characters.")
        
        else:
            # 3. Prediction (Only runs if valid)
            kmers = getKmers(clean_seq)
            kmer_text = ' '.join(kmers)
            
            input_vector = vectorizer.transform([kmer_text])
            prediction = model.predict(input_vector)[0]
            probs = model.predict_proba(input_vector)

            st.write("---")
            st.subheader(f"Result: It is a **{prediction}**")
            
            # 4. Visualization
            col1, col2 = st.columns([1, 2])

            with col1:
                image_map = {
                    "Dog": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
                    "Cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
                    "Human": "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?w=400"
                }
                st.image(image_map[prediction], caption=f"{prediction} Detected", use_container_width=True)

            with col2:
                st.write("**Probability Distribution**")
                prob_df = pd.DataFrame(probs, columns=model.classes_)
                st.bar_chart(prob_df.T)
                
                confidence = prob_df[prediction][0] * 100
                st.info(f"Model Confidence: **{confidence:.2f}%**")