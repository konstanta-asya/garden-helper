import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Завантаження даних про класи (ті, що збереглися в моделі)
checkpoint = torch.load("garden_expert_model_v2.pth", map_location="cpu")
all_classes = checkpoint['classes']

# 2. Словник порад українською для всіх культур
ADVICE_UA = {
    "healthy": "Рослина здорова! Продовжуйте стандартний догляд.",
    "Bacterial_spot": "Бактеріальна плямистість. Обробіть препаратами на основі міді.",
    "Early_blight": "Альтернаріоз. Видаліть нижні листки та обробіть фунгіцидом.",
    "Late_blight": "Фітофтороз. Необхідна термінова обробка системними фунгіцидами.",
    "Leaf_scorch": "Опік листя. Перевірте режим поливу та захистіть від палючого сонця.",
    "Black_rot": "Чорна гниль. Обов'язкова обрізка хворих гілок та обробка саду восени.",
    "Powdery_mildew": "Борошниста роса. Використовуйте препарати на основі сірки або сучасні фунгіциди."
    # Можна додати специфічні поради для лохини, малини тощо
}


# 3. Завантаження моделі
@st.cache_resource
def load_garden_model():
    model = models.mobilenet_v3_small()
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(all_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


model = load_garden_model()

# 4. Веб-інтерфейс
st.set_page_config(page_title="Садовий Помічник", page_icon="🌿")
st.title("🌿 Експерт вашого саду")
st.write("Завантажте фото листка (томат, яблуня, полуниця, вишня, картопля, малина, лохина)")

# Замість file_uploader використовуйте це:
file = st.camera_input("Зробіть фото листка прямо зараз")

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, use_container_width=True)

    # Підготовка
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    # Передбачення
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        conf, idx = torch.max(prob, 0)

    # Виведення
    class_name = all_classes[idx]
    st.subheader(f"Діагноз: {class_name.replace('___', ' ').replace('_', ' ')}")

    # Пошук поради у словнику
    advice = "Порада: Проконсультуйтеся з агрономом або перевірте тип добрив."
    for key, val in ADVICE_UA.items():
        if key in class_name:
            advice = val
            break

    st.info(f"💡 {advice}")
    st.write(f"Впевненість моделі: {conf.item() * 100:.1f}%")
