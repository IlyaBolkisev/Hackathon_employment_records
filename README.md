# Hackathon_employment_records


Команда: Лебедь, рак и щука <br>
Состав: Илья Болкисев, Матвей Колтунов, Слава Шишаев
<br>

Описание проекта: <br>
Веб-сервис с API, способный распозновать ключевые поля в трудовых книжках, визуально парсить их в интерфейс, в котором пользователь (оператор) может редактировать ошибки модели. <br>
На основе исправленных ошибок собирается json файл, который используется для дообучения моделей. <br>
<br>

Навигация: <br>
modules - модули функционала: <br>
  обернутые модели (localization.py, recognition.py); <br>
  CV алгоритмы детекции печатей (stamp_detection.py) и ячеек таблицы (parse_table.py); <br>
  формирование json файла (wrapper.py). <br>
app.py - веб-приложение, написанное на Flask. <br> 
templates - шаблоны веб-страниц, в которые парсятся данные книжек. <br>
notebooks - ноутбуки с процессом обучения моделей локализации книжек и распознавания текста. <br>
<br>

Датасеты: <br>
Развороты книжек в формате yolo (https://drive.google.com/drive/folders/13W-6vfIt0dEH0q_yxsZ7wxAh9qL2M7PD?usp=sharing) <br>
Рукописный русский текст (https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset/data) <br>
<br>

Веса обученных моделей: <br>
Yolov8-obb (https://drive.google.com/file/d/1yDUP7-pzwcY-YLgRealOoiUVe2aEq_Uc/view?usp=sharing) <br>
Text Recognition Transformer (https://drive.google.com/file/d/1hu3k1mHYaKF9CCRydBcFlFPaA6BPLChL/view?usp=sharing) <br>
Text Recognition CRNN onxx (https://drive.google.com/file/d/1rS9DB_0ZSRoYJvDgUbpZD2hrQX6CaW2L/view?usp=sharing) <br>
Text Recognition CRNN H5 (https://drive.google.com/file/d/1pnRhqVB_kKya3QWMMHkOWwNsO9RgTvdU/view?usp=sharing) <br>
