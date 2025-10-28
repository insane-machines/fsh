<h1 align="center">
  <img src="https://github.com/insane-machines/fsh/blob/main/fsh/fsh.egg-info/logo.jpg"></img>
</h1>
<h1 align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=00FFAA&center=true&vCenter=true&width=500&lines=Hi,+we're+Insane+Machines!;Welcome+to++our+machinelearning+library;named+Forward+Stepwise+Heuristics!" alt="Typing SVG" />
</h1>

<p align="center">
  <a href="https://t.me/insane_machines" target="_blank">
    <img src="https://img.shields.io/badge/Insane%20Machines-Telegram-blue?style=for-the-badge&logo=telegram" alt="Telegram link" />
  </a>
</p>
Features

Linear model with: 
• **Callbacks**
• **Metrics** 
• **Preprocessing functions**
• **Simple integration**  

---

Quick Start

```python
import fsh

X = [[1, 2], [3, 4]]
y = [[5], [6]]

#array converting is integrated! but you can do it yourself!
X = fsh.fsh.preprocessing.to_array(X)
y = fsh.fsh.preprocessing.to_array(y)

callback = fsh.fsh.callbacks.History()
model = fsh.fsh.Linear(n_features=3, lr=0.01)

# Обучение
model.fit(X, y, epochs=100, callbacks=[callback], metrics=fsh.fsh.metrics.mse)

# Прогноз
y_pred = model.predict(X)
