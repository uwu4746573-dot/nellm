# Training Strategy Research Report
*Compiled May 23, 2026 by Research Agent*

---

## 1. Gradient Estimator: Decoupled ST-GS (ICLR 2025)

**Ключевая рекомендация:** Использовать **Decoupled Straight-Through Gumbel-Softmax**.

Стандартный ST-GS использует один параметр τ для двух целей сразу. Decoupled ST-GS разделяет их:

```python
# Forward: near-discrete (ближе к инференсу)
hard_sample = argmax(logits + Gumbel_noise * τ_f)
# Backward: smooth gradients
soft_approx = softmax((logits + Gumbel_noise) / τ_b)
```

| Метод | Смещение | Дисперсия | Рекомендация |
|-------|----------|-----------|--------------|
| REINFORCE | Нет | Очень высокая | Гибрид с RLOO baseline |
| STE | Есть | Низкая | Только для квантизации |
| ST-GS (стандарт) | Есть | Низкая | Базовый вариант |
| **Decoupled ST-GS** | Есть | **Ниже** | **Наш случай ✅** |

**Дополнительно (BranchGRPO, NeurIPS 2025):** REINFORCE с RLOO baseline — использовать W-1 проигравших веток как базовую награду. Даёт несмещённый сигнал через дискретный выбор ветки.

---

## 2. Annealing Schedule для Gumbel Temperature

### Decoupled расписание

```python
τ_f(t) = max(0.05, τ_f0 * exp(-λ_f * t))  # быстрее → near-hard
τ_b(t) = max(0.30, τ_b0 * exp(-λ_b * t))  # медленнее → smooth grads
# Правило: τ_b > τ_f всегда, отношение 3-5x
```

### Фазовое расписание (Curriculum)

| Фаза | Шаги | τ_f | τ_b | Глубина дерева |
|------|------|-----|-----|----------------|
| Warm-up | 0–2k | 1.0 | 1.0 | — |
| Curriculum 1 | 2k–10k | 0.8→0.4 | 0.8→0.6 | depth=1, W=2 |
| Curriculum 2 | 10k–30k | 0.4→0.15 | 0.6→0.35 | depth=2-3, W=3 |
| Curriculum 3 | 30k+ | 0.15→0.05 | 0.35→0.1 | full depth, W=5 |

**⚠️ Критично:** Не снижать τ ниже 0.05 во время обучения — градиенты становятся нестабильными.

---

## 3. Curriculum обучения: от простого к сложному

Протокол **Coconut (Meta FAIR, ICLR 2025)** — наиболее близкий аналог нашей архитектуры:

- **Stage 1:** Beam depth=1, W=2. Обучить базовую геометрию латентного пространства (цель: accuracy > 80%)
- **Stage 2:** Depth=2, W=3. Добавить process reward на промежуточных узлах.
- **Stage 3:** Depth=3-4, W=5. Добавить KL-penalty против схлопывания веток.
- **Stage 4:** Full depth, W=5. Добавить REINFORCE сигнал для end-to-end выбора ветки.

**Ключевые правила:**
1. Не увеличивать глубину до конвергенции на текущей глубине.
2. **Process rewards** (пошаговые награды) вдвое эффективнее outcome rewards — ReST-MCTS* (NeurIPS 2024).
3. Сортировать обучающие примеры по минимальной глубине решения (Depth-of-Thought ordering).

---

## 4. Contrastive Learning для выравнивания латентного пространства

### InfoNCE с In-Trajectory Negatives (CRTR, ICLR 2025)

**Ключевая идея:** Использовать W-1 проигравших веток как hard negatives — это заставляет латентное пространство геометрически разделять правильные и неправильные пути внутри одного примера.

```python
loss = InfoNCE(
    anchor=z_t,              # текущее состояние на глубине t
    positive=z_t_winning,    # следующий шаг по победной ветке
    negatives=[
        z_t_losing_1,        # ← IN-TRAJECTORY (ключевой ингредиент!)
        z_t_losing_2,        # все W-1 проигравших ветки
        z_other_examples,    # кросс-пример негативы
    ]
)
```

### Комбинированный лосс

```python
L_total = L_task              # основной (только победная ветка)
        + 0.10 * L_InfoNCE    # contrastive alignment
        + 0.05 * L_branch     # branch separation loss
        + 0.02 * L_coherence  # trajectory smoothness
```

---

## 5. Ключевые статьи для нашей архитектуры

| Статья | Площадка | Применение |
|--------|----------|------------|
| **Coconut** (Chain of Continuous Thought) | ICLR 2025 | Curriculum: токены → непрерывные шаги. Emergent BFS в латентном пространстве. |
| **Decoupled ST-GS** | ICLR 2025 | Раздельные τ_f и τ_b для Top-K градиента. |
| **ReST-MCTS\*** | NeurIPS 2024 | Process reward models + MCTS. Автогенерация пошаговых наград. |
| **CRTR** | ICLR 2025 | In-trajectory negatives. Критично для нашего beam search. |
| **BranchGRPO** | NeurIPS 2025 | REINFORCE + проигравшие ветки как baseline. |
| **Huginn** | 2025 | Рекуррентные "latent thinking" блоки. |
| **Latent-SFT** | OpenReview 2025 | SOTA на GSM8k/AIME24 через латентный подпространство. |

---

## Чеклист изменений в архитектуру

### Без изменений кода (config only)
- [ ] Decoupled ST-GS: добавить τ_f и τ_b как отдельные параметры в LatentRouterLayer
- [ ] Exponential annealing с адаптивной паузой на loss plateau

### Небольшие изменения кода
- [ ] Process rewards на каждой глубине beam search
- [ ] RLOO baseline: reward(winner) - mean(reward(losers))
- [ ] L_coherence loss: гладкость траектории в латентном пространстве

### Ключевые гиперпараметры

| Параметр | Начало | Конец | Примечание |
|----------|--------|-------|------------|
| τ_f | 1.0 | 0.05 | Быстрый annealing |
| τ_b | 1.0 | 0.1–0.3 | Медленный, всегда > τ_f |
| τ_contrastive | — | 0.07 | Отдельно от Gumbel τ! |
| α (contrastive weight) | 0.0 | 0.05–0.15 | Рост вместе с глубиной |
| RLOO / backprop mix | — | 20/80 | Стартовая точка |
