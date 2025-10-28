<h1 align="center">
  <img src="https://github.com/insane-machines/fsh/fsh.egg-info/logo.jpg"></img>
</h1>
<h1 align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=00FFAA&center=true&vCenter=true&width=500&lines=Hi,+we're+Insane+Machines!;Welcome+to+Forward+Stepwise+Heuristics!" alt="Typing SVG" />
</h1>

<p align="center">
  <a href="https://t.me/insane_machines" target="_blank">
    <img src="https://img.shields.io/badge/Insane%20Machines-Telegram-blue?style=for-the-badge&logo=telegram" alt="Telegram link" />
  </a>
</p>

---

### 🧠 About Forward Stepwise Heuristics

**Forward Stepwise Heuristics (FSH)** — это лёгкая, но мощная библиотека для обучения и экспериментов с **линейными моделями**.  
Она создана для исследователей и инженеров, которые хотят контролировать каждый шаг обучения.

---

### ⚙️ Features

✅ **Линейные модели** с кастомными весами и смещением  
✅ **Callbacks** — гибкое управление процессом обучения  
✅ **Metrics** — оценка качества модели в реальном времени  
✅ **Preprocessing** — встроенные функции нормализации и масштабирования  
✅ **История обучения** — трекинг loss и val_loss по эпохам  
✅ **Простая интеграция** с NumPy и Pandas  

---

### 🚀 Quick Start

```python
from fsh import Linear
from fsh.callbacks import History
from fsh.metrics import mse
from fsh.preprocessing import normalize
import numpy as np

# Пример данных
X = np.random.randn(100, 3)
y = X @ np.array([[2], [-1], [0.5]]) + 1.0

# Препроцессинг
X = normalize(X)

# Модель
model = Linear(n_features=3, lr=0.01)

# Обучение
model.fit(X, y, epochs=100, callbacks=[History()], metrics=[mse])

# Прогноз
y_pred = model.predict(X)
