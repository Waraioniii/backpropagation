{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e4dbcc1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1eb5afc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# функция для вычесления гиперболочискего тангенса\n",
    "def f(x):\n",
    "    return 2 / (1 + np.exp(-x)) - 1 \n",
    "\n",
    "# Функция для вычесления производной\n",
    "def df(x):\n",
    "    return 0.5 * (1 + x) * (1 - x)\n",
    "\n",
    "# веса для нейронной сети (1ый и 2ой слой)\n",
    "W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])\n",
    "W2 = np.array ([0.2, 0.3])\n",
    "\n",
    "# функция пропускающая вектор наблюдения через нейроную сеть\n",
    "def go_forward(inp):\n",
    "    sum = np.dot(W1, inp)\n",
    "    out = np.array([f(x) for x in sum])\n",
    "    \n",
    "    sum = np.dot(W2, out)\n",
    "    y = f(sum)\n",
    "    return (y, out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "e1f05ce2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Выходные значения НС: 0.04485667576833108 => -1\n",
      "Выходные значения НС: 0.9289479096744906 => 1\n",
      "Выходные значения НС: -0.8717657213207441 => -1\n",
      "Выходные значения НС: 0.8720036258297941 => 1\n",
      "Выходные значения НС: -0.8720036258297941 => -1\n",
      "Выходные значения НС: 0.8717657213207441 => 1\n",
      "Выходные значения НС: -0.9289479096744906 => -1\n",
      "Выходные значения НС: -0.044856675768330856 => -1\n"
     ]
    }
   ],
   "source": [
    "def train(epoch):\n",
    "    global W2, W1\n",
    "    lmd = 0.01  # шаг обучения\n",
    "    N = 10000   # число итераций при обучении\n",
    "    count = len(epoch)\n",
    "    for k in range(N):\n",
    "        x = epoch[np.random.randint(0,count)]\n",
    "        y, out = go_forward(x[0:3])\n",
    "        e = y - x[-1] \n",
    "        delta = e * df(y)\n",
    "        W2[0] = W2[0] - lmd * delta * out[0]\n",
    "        W2[1] = W2[1] - lmd * delta * out[1]\n",
    "        \n",
    "        delta2 = W2 * delta * df(out)\n",
    "        \n",
    "        # корректиорвка связей первого слоя\n",
    "        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd\n",
    "        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd\n",
    "        \n",
    "epoch = [(-1, -1, -1, -1,),\n",
    "        (-1, -1, 1, 1),\n",
    "        (-1, 1, -1, -1),\n",
    "        (-1, 1, 1, 1),\n",
    "        (1, -1, -1, -1),\n",
    "        (1, -1, 1, 1),\n",
    "        (1, 1, -1, -1),\n",
    "        (1, 1, 1, -1)]\n",
    "\n",
    "train(epoch)\n",
    "\n",
    "for x in epoch:\n",
    "    y, out = go_forward(x[0:3])\n",
    "    print(f\"Выходные значения НС: {y} => {x[-1]}\")\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a1f8555a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
