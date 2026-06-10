# Resumen del proceso experimental — TFM Trading RL con PPO sobre ABIDES

Documento de referencia para redactar la memoria. Cubre el camino completo desde
la primera arquitectura hasta MAPPO, con cada error encontrado, su diagnóstico y la solución aplicada.

---

## 0. Contexto del sistema

- **Simulador:** ABIDES (rmsc04 config) — mercado sintético con agentes rule-based de fondo
- **Activo:** SPY simulado, sesión 9:30-16:00, timestep 60s (~390 steps/episodio)
- **Algoritmo base:** PPO (Proximal Policy Optimization)
- **Observación:** 44 features del LOB (Limit Order Book) — 10 niveles bid/ask + estado del portfolio
- **Acciones:** 3 discretas (comprar / vender / mantener), sin venta en corto
- **Capital inicial:** $1,000,000

---

## 1. Diseño inicial de la función de recompensa — iteración por error y corrección

La función de recompensa pasó por **5 iteraciones** antes de estabilizarse:

### v0 — Reward ingenuo (solo PnL final)
- **Problema:** sparse reward, el agente solo recibía señal al final del episodio → no aprendía nada en 390 steps
- **Síntoma:** entropy se mantenía en máximo (~1.10), reward planos en 0

### v1 — Hold penalty
- **Cambio:** penalizar al agente por no operar (`hold_penalty`)
- **Problema:** el agente aprendió a operar de forma errática solo para evitar la penalización, no para ganar dinero
- **Diagnóstico:** estábamos incentivando *actividad*, no *rentabilidad*

### v2 — Opportunity cost
- **Cambio:** sustituir hold_penalty por un coste de oportunidad — penalizar mantener capital ocioso sin invertir
- **Razonamiento:** queríamos que el agente "sintiera" que los días sin operar acumulan pérdida de oportunidad
- **Iteración interna:** primero proporcional al capital ocioso completo (`opportunity_cost_coef=0.001` → demasiado agresivo, explotaba), luego ajustado a 0.0002

### v3 — Inventory penalty
- **Cambio:** se añadió penalización por mantener inventario alto (pensado originalmente para escenarios de market-making)
- **Problema:** el agente estaba haciendo trading direccional, no market-making — la penalización de inventario no tenía sentido conceptual en ese contexto
- **Decisión:** se eliminó completamente (`921f6de — eliminar inv_penalty`)

### v4 — Reward final alineado con el paper ABIDES-Gym
- **Diseño final:**
  ```
  reward = ΔM2M / starting_cash − inv_penalty(0.0001) − opportunity_cost(0.0001)
  ```
  Donde **M2M = Mark-to-Market** = `cash + holdings × mid_price` (valor real del portfolio)
- **Por qué funcionó:** mide directamente la creación de valor real paso a paso (no solo al final), con penalizaciones pequeñas que desalientan comportamientos degenerados sin dominar la señal principal
- **Lección clave:** el reward debe estar dominado por la señal económica real (PnL), las penalizaciones deben ser "ruido de fondo" que dirige sin dominar

---

## 2. Arquitectura de red — de MLP a Transformer con frame stacking

### MLP simple (baseline inicial)
- `obs(44) → 256 → 256 → acción/valor`
- Funcional pero trata el LOB como vector plano, sin estructura

### Transformer Actor-Critic
- **Diseño:** tokeniza la observación de 44 features en 11 tokens de 4 dimensiones
  - Tokens 0-9: niveles del LOB → `(bid_price, bid_vol, ask_price, ask_vol)`
  - Token 10: estado del portfolio → `(holdings, cash, pnl, time_progress)`
- **Razonamiento:** permite que la atención aprenda relaciones entre niveles del libro (imbalance bid/ask, profundidad) que una MLP trataría como features independientes
- **Positional encoding sinusoidal:** necesario porque el orden de los niveles del LOB importa (nivel 0 = mejor precio)

### Frame stacking (N=10)
- **Problema identificado:** el Transformer solo veía el estado actual del mercado, sin contexto temporal — no podía distinguir una tendencia de un movimiento aleatorio
- **Solución:** apilar las últimas 10 observaciones → secuencia de 110 tokens
- **Resultado:** mejoró el PnL de -0.11% a +0.03% (v1 → v2), demostrando que la profundidad temporal aporta señal real

---

## 3. Intento de memoria recurrente — GRU (revertido)

- **Hipótesis:** en vez de frame stacking (memoria fija de N frames), dar al agente memoria recurrente tipo GRU permitiría capturar dependencias de cualquier longitud
- **Implementación:** GRU añadida sobre las features del Transformer (`f05ffac`)
- **Problema fundamental:** en PPO multi-entorno (10 envs en paralelo), el estado oculto `h` se resetea a 0 en cada minibatch durante el update (Opción B de TBPTT) → el crítico veía secuencias falsas/inconsistentes
- **Síntoma:** value loss explotó de 0.005 a 0.41
- **Intento de mitigación:** se subió `entropy_coef` de 0.005 a 0.01 pensando que era colapso de política — no resolvió el problema real
- **Decisión final:** revertir GRU completamente (`3782832`)
- **Lección y línea futura:** GRU con TBPTT correcto (Opción A — actualización secuencial propia por entorno, sin resetear `h`) queda documentado como trabajo futuro

---

## 4. Paralelización — SubprocVecEnv

- **Cambio:** de un solo entorno a 10 entornos ABIDES corriendo en subprocesos paralelos vía `multiprocessing` + `Pipe`
- **Por qué:** ABIDES es lento (simulación de mercado completa por step) — sin paralelización, entrenar 1M steps habría sido inviable en tiempo
- **Verificación de correctitud:** se comprobó explícitamente que (a) los workers recolectan transiciones independientes y (b) el modelo se actualiza correctamente con el rollout combinado (GAE por entorno, flatten a batch único)
- **Resultado:** ~45 FPS estables con 10 workers (vs ~5 FPS en single-env)

---

## 5. Gestión térmica del hardware (miniPC)

- **Problema:** la CPU alcanzaba 104°C con 10 workers a frecuencia máxima (4.9 GHz) → riesgo de throttling agresivo y daño térmico a largo plazo
- **Solución:** cap de frecuencia a 3.2 GHz (`scaling_max_freq`)
- **Resultado (sweet spot encontrado empíricamente):** 10 workers + 3.2 GHz → 96 FPS estables a 68°C
- **Detalle interesante:** limitar la frecuencia *aumentó* la estabilidad de los FPS (menos throttling reactivo) sin pérdida significativa de velocidad — contraintuitivo pero medible

---

## 6. Colapso de política (policy collapse) — el problema central de hiperparámetros

Este fue el problema más recurrente y más instructivo de todo el proceso.

### Manifestación
- La entropía de la política (medida de cuán aleatoria es la distribución de acciones) cae desde el máximo teórico `log(3) ≈ 1.10` hasta valores muy bajos (~0.3)
- El agente deja de explorar y "se aferra" a una única acción, el reward se estanca o cae

### v1/v2 — colapso con `entropy_coef = 0.005`
- **Diagnóstico:** el bonus de entropía era demasiado débil para contrarrestar la presión del gradiente de política hacia la convergencia prematura
- **Momento de colapso:** ~250k steps (update 25-30)

### v3 — corrección con `entropy_coef = 0.02`
- **Resultado:** entropía se estabilizó en ~0.80-0.85 (vs colapso a 0.3), reward convergió suavemente a ~0.30 y se mantuvo estable durante 1M steps completos
- **Este es el modelo que se usó como referencia/mejor resultado**

### MAPPO — colapso inverso (entropy drift hacia el máximo)
- **Manifestación distinta:** con la arquitectura dual, la entropía no caía sino que **subía** de vuelta hacia 1.10 (política → aleatoria uniforme), y el reward se desplomaba en paralelo
- **Diagnóstico:** el gradiente de política del arbitrador era demasiado débil (推 Bull/Bear no aportaban señal útil) para contrarrestar el bonus de entropía con `entropy_coef = 0.02`
- **Corrección:** bajar a `entropy_coef = 0.008` — mejoró pero el problema reapareció cíclicamente a partir del update ~25
- **Conclusión:** el equilibrio entropía/reward es extremadamente sensible a la arquitectura — un valor que funciona en una red no se traslada directamente a otra

**→ Lección general:** el coeficiente de entropía no es un hiperparámetro "universal" — depende de la fuerza relativa del gradiente de política, que a su vez depende de cuán informativo es el reward y cuán compleja es la arquitectura.

---

## 7. Experimento fallido — Inventory Penalty para mejorar el Sharpe Ratio (v4)

### Motivación
- v3 daba mean PnL +0.31% pero con std muy alta (2.19%) → Sharpe bajo (0.14)
- **Hipótesis:** si penalizamos más el inventario (`inv_penalty_coef`: 0.0001 → 0.001, luego 0.0003), el agente operará de forma más conservadora, reduciendo la varianza

### v4_invpen001 (penalización 10x) — fallo por sobre-corrección
- **Síntoma:** la entropía subió progresivamente de 0.84 a 1.06 — el agente no encontraba ninguna estrategia rentable bajo la nueva penalización y "tiraba los dados"
- **Decisión:** parar y reducir la penalización

### v4_invpen0003 (penalización 3x) — fallo por el motivo correcto
- **Resultado en eval:** mean PnL **-0.45%** (peor que v3), std más baja (~1.5%, sí se redujo la varianza) pero Sharpe igualmente negativo
- **Diagnóstico clave (verbalizado por el propio autor del TFM):**
  > "Hacer que no entre tanto simplemente baja la varianza, no va a hacer que mágicamente vaya a ganar más que antes"
- **Conclusión:** reducir la frecuencia de operación es una estrategia *defensiva*, no *generativa* — el agente se acerca a 0% de retorno (ni gana ni pierde) en lugar de aprender a identificar mejores oportunidades. Para mejorar el Sharpe de verdad hace falta mejor *capacidad predictiva*, no menos actividad.

**→ v3 se mantiene como mejor modelo** (+0.31% mean, Sharpe 0.14, validado en eval de 100 episodios con resultado consistente: +0.31% inicial con 20 episodios, +0.31% con 100 episodios)

---

## 8. MAPPO — arquitectura jerárquica Bull/Bear + Arbitrador

### Diseño conceptual
Inspirado en MAPPO (Multi-Agent PPO, Yu et al. 2021) pero adaptado a un contexto de un solo activo:

```
Transformer Encoder (compartido)
        ↓
   ┌────┴────┐
Bull Head   Bear Head      → scores ∈ [0,1] (predicción de dirección a t+k)
   └────┬────┘
   Arbitrador               → decide acción final (buy/sell/hold)
   Critic                   → valor para PPO del arbitrador
```

- **Bull/Bear:** redes especializadas en predecir si el precio subirá/bajará en los próximos `k=5` steps (5 minutos), entrenadas con **reward retardado** (delayed reward) y BCE loss contra el resultado real observado en `t+k`
- **Arbitrador:** red que decide la acción final viendo tanto las features del mercado como los scores de Bull/Bear, entrenada con PPO estándar sobre el reward M2M real

### Por qué NO es MAPPO puro
Se discutió explícitamente esta distinción: MAPPO real requeriría **múltiples agentes actuando simultáneamente en el mismo mercado** (cada uno con su propio portfolio, afectándose mutuamente vía ABIDES). Lo implementado es más cercano a un **ensemble jerárquico con componentes especializados** — una arquitectura *inspirada* en la filosofía de MAPPO (roles especializados + decisión centralizada) pero aplicada a un solo agente con sub-redes internas.

### Decisión de diseño crítica — por qué el arbitrador y no un theta fijo
La primera idea fue un árbitro basado en un umbral fijo (`θ`): si `|score_bull - score_bear| > θ` → actuar. Se descartó por dos razones identificadas explícitamente:
1. **Theta fijo es ciego al dinero:** no sabe si el portfolio actual hace que actuar sea buena idea
2. **Bull/Bear optimizarían precisión direccional, no rentabilidad:** podrían ser muy precisos prediciendo "sube/baja" pero entrar en mal momento relativo al estado del portfolio

La solución fue convertir el árbitro en una **red entrenada con el reward de PnL real (M2M)**, de forma que aprende cuándo *vale la pena* confiar en las señales de Bull/Bear dado el contexto completo del portfolio.

### Errores encontrados y corregidos durante la implementación

#### Error 1 — Value Loss explosivo (49 vs ~0.03 esperado)
- **Causa:** se normalizaban los rewards crudos a media 0 / std 1 con `RunningMeanStd`, pero el GAE acumula esos rewards descontados a lo largo de 1024 steps con `gamma=0.999` → los returns se disparaban a escalas de ±50-100 mientras el crítico arrancaba desde 0
- **Solución:** normalizar los **returns** después de calcular el GAE (no los rewards antes), y eliminar la normalización redundante de rewards crudos

#### Error 2 — Contaminación del encoder compartido por gradiente de ruido direccional
- **Síntoma:** con `alpha=0.5` (peso de la loss Bull/Bear), el reward del sistema completo empezó a *bajar* en lugar de subir, justo cuando en v3 ya estaría subiendo limpiamente
- **Diagnóstico:** Bull/Bear no podían aprender señal real (el mercado es esencialmente aleatorio a 5 steps de horizonte — su BCE loss se quedaba estancada en `ln(2) ≈ 0.693`, el valor exacto de un clasificador binario aleatorio). Su gradiente de "aprendizaje" eraен realidad ruido puro, y al fluir hacia el encoder compartido, corrompía las features que el arbitrador necesitaba para aprender la política de trading
- **Solución:** aplicar `.detach()` a las features antes de pasarlas a las cabezas Bull/Bear — de esta forma Bull/Bear aprenden sus propias cabezas pero **no propagan gradiente al encoder compartido**. El encoder queda protegido y solo lo entrena la señal limpia del arbitrador (PPO sobre PnL real)

#### Error 3 — Entropy drift cíclico (el mismo problema del punto 6, mutado)
- Documentado en la sección 6 — recurrencia del problema de balance entropía/reward, con manifestación inversa (entropía sube en vez de caer) y sensible al nuevo `entropy_coef`

### Verificación de robustez antes de entrenar
Se construyó una suite de **31 tests de integración** (`test_dual.py`) verificando:
- Forward pass: shapes correctas, scores en `[0,1]`, acciones válidas
- Buffer: targets direccionales retardados asignados correctamente con datos sintéticos controlados
- Gradientes: los 7 componentes de la red (encoder, input projection, bull head, bear head, 2 capas del arbitrador, crítico) reciben gradiente
- Update: los pesos de cada componente cambian tras `optimizer.step()`
- Normalización: `RunningMeanStd` converge a media/std correctas y aplica clipping
- Integración end-to-end con el entorno ABIDES real

### Resultado final — comparación cuantitativa (eval de 100 episodios, idéntica metodología)

| Métrica | PPO_Transformer_v3 (mejor modelo) | MAPPO_Dual (checkpoint intermedio, mejor punto) |
|---|---|---|
| Mean PnL diario | **+0.31%** | +0.19% |
| Std PnL | 2.19% | 2.35% |
| Sharpe Ratio | **0.14** | 0.08 |
| % episodios positivos | **53%** | 51% |
| Min / Max PnL | -6.14% / +6.07% | -5.22% / +5.68% |

**v3 se mantiene como modelo final.**

### Por qué falló — el diagnóstico más importante de todo el TFM
Las cabezas Bull/Bear nunca consiguieron bajar su BCE loss de `0.693` (= `ln(2)`, el valor exacto de un clasificador binario que predice 50/50 sin información). Esto significa que **no existe señal direccional explotable a un horizonte de 5 minutos en el mercado simulado rmsc04** — al menos no una que una red neuronal de esta capacidad pueda extraer del LOB.

Esta conclusión se verbalizó explícitamente durante el proceso:
> "Si fuera tan fácil entrar [encontrar señal predictiva en mercados], todo el mundo lo haría"

El experimento MAPPO, aunque no superó al modelo base, **demuestra empíricamente y de forma rigurosa la dificultad fundamental del problema** — confirmando con datos (no solo intuición) la hipótesis de eficiencia de mercado a corto plazo incluso en un entorno sintético simplificado. Una arquitectura más compleja, diseñada específicamente para extraer señal direccional, "delata" mediante su propio fracaso de aprendizaje que esa señal no está disponible.

---

## 9. Benchmarks contra estrategias clásicas — Buy & Hold y MACD, y el descubrimiento de la política degenerada

> **Nota de versión:** una primera iteración de este benchmark (solo B&H, sin seeds fijadas) dio resultados que luego se demostraron no reproducibles y mal interpretados por desajuste de exposición. Esta sección refleja la versión final y verificada. La historia completa del proceso de verificación está en el subapartado "El falso positivo y la cadena de verificación" — es material valioso para la memoria como ejemplo de rigor metodológico.

### Motivación
Un PnL positivo no demuestra por sí solo que el agente haya aprendido algo útil — podría estar simplemente expuesto a una deriva de fondo del mercado. Para que el resultado tenga valor científico hay que compararlo contra baselines clásicos: **Buy & Hold (B&H)** (comprar al inicio y mantener, sin inteligencia) y **MACD crossover** (regla técnica estándar de seguimiento de tendencia: EMA(12)−EMA(26) con línea de señal EMA(9), compra en cruce alcista, vende en cruce bajista, sin venta en corto, warm-up de 35 observaciones, ejecución al mid-price del paso siguiente al cruce).

### Diseño metodológico — comparación emparejada y reproducible
- **Emparejada:** ABIDES genera mercados estocásticos, así que comparar lotes de episodios distintos no es justo. En cada episodio donde corre el agente se registra la **serie completa de mid-prices** (1 obs/minuto), y B&H y MACD se calculan en paper-trading offline sobre **esa misma serie** — las tres estrategias ven exactamente el mismo mercado.
- **Reproducible:** las corridas iniciales no fijaban seed (episodios irrepetibles). La versión final fija `env.seed(42+i)` por episodio (seeds 42–71 documentadas en el JSON) — cualquier revisor puede regenerar los mismos episodios exactos. Se verificó el determinismo entre procesos (mismo episodio, mismo PnL al cuarto decimal).
- **Caveat documentado del MACD:** ejecuta al mid-price sin pagar spread ni mover el mercado; el agente cruza el spread real con cada orden de mercado dentro de la simulación. Esto infla moderadamente la comparativa a favor del MACD y debe mencionarse junto al resultado.

### El descubrimiento central: ambos agentes son un "Buy & Hold apalancado ~10x"

El hallazgo más importante de todo el benchmark no estaba previsto. Al regresar el PnL del agente contra el PnL de B&H episodio a episodio:

```
v3:    agente = 9.8  × B&H − 0.06%    (correlación +0.991)
MAPPO: agente = 10.0 × B&H + 0.01%    (correlación +0.997)
```

Y el dato que lo confirma definitivamente: **`mean_final_hold = 100.0` en los 100 episodios de la evaluación final de AMBOS modelos** — los dos terminan todos los episodios en `max_inventory` (100 acciones × ~100.000 = posición de ~10M sobre 1M de capital). La política aprendida por las dos arquitecturas es la misma política degenerada: **comprar hasta el tope lo antes posible y mantener** — una beta apalancada 10x, no una estrategia de trading.

Esto reinterpreta retroactivamente los resultados previos:
- El **+0.31% de v3 en la eval de 100 episodios** era ≈ 10 × la deriva ligeramente positiva de aquella muestra de mercados — exposición, no habilidad.
- La **std de 2.19%** del agente = ~10 × la std del retorno del mercado (~0.22%) — la varianza que motivó los experimentos del Sharpe (v4) era apalancamiento puro.
- El benchmark justo del agente no es B&H 1x sino **B&H apalancado 10x** (misma exposición).

### Resultados finales (30 episodios emparejados, seeds 42–71, rmsc04)

| | v3 | MAPPO |
|---|---|---|
| Agente (exposición ~10x) | -1.26% | -1.28% |
| B&H 1x | -0.12% | -0.13% |
| B&H apalancado 10x (benchmark justo) | -1.22% | -1.30% |
| **MACD 1x** | **+0.34%** (Sharpe 1.66, 90% pos.) | **+0.34%** (Sharpe 1.72, 93% pos.) |
| **Alpha del agente vs B&H 10x** | **-0.04%** (-0.7σ, no significativo) | **+0.01%** (+0.3σ, no significativo) |

(El bloque de seeds 42–71 resultó tener deriva de mercado levemente negativa — de ahí los PnL negativos del agente y de B&H apalancado; lo relevante es la comparación a exposición igualada, no el signo del lote.)

### Interpretación

1. **A exposición igualada, el alpha de ambos agentes es estadísticamente cero.** Ni v3 ni MAPPO añaden valor de *timing* sobre estar simplemente posicionado al máximo. El intercepto ligeramente negativo de v3 (-0.06%/episodio) refleja la fricción de cruzar el spread al construir la posición.

2. **El MACD es la única estrategia con alpha de timing genuino**: +0.34% por episodio con exposición 1x, Sharpe ~1.7, 90-93% de episodios positivos, consistente en los dos lotes de mercados. Una regla técnica de los años 70 supera a ambas arquitecturas neuronales.

3. **El resultado del MACD no contradice el null result de las cabezas Bull/Bear — opera a otra escala temporal.** Bull/Bear predecían a 5 minutos exactos y no encontraron señal; el MACD con EMAs de 12/26 minutos captura tendencias de decenas de minutos — justamente la escala a la que operan los Momentum Agents del rmsc04. La narrativa empírica completa: **el mercado simulado es eficiente a ~5 minutos pero tiene momentum explotable a ~20-40 minutos**, que el MACD captura y el agente RL no puede ver porque su ventana de contexto (frame stacking) es de solo 10 minutos. Esto da justificación empírica directa a la línea futura de memoria recurrente (sección 11).

4. **Por qué el RL convergió a la política degenerada:** "comprar todo y mantener" es un máximo local muy accesible — da reward positivo en mercados con deriva positiva sin requerir predicción, y una vez alcanzado (entropía colapsada sobre la acción BUY al inicio del episodio), salir de él exigiría descubrir la señal de momentum a 20-40 min que la arquitectura no puede representar bien. Conecta los problemas de colapso de entropía (sección 6) con la limitación de contexto temporal.

### El falso positivo y la cadena de verificación (material metodológico)

La primera corrida del benchmark (sin seeds) dio "alpha +0.08% (v3) / +0.61% (MAPPO) sobre B&H" — un aparente resultado positivo. Al añadir el MACD y fijar seeds para reproducibilidad, el agente pasó a -1.26%, a ~4 errores estándar de lo esperado. En lugar de aceptar ninguno de los dos números, se ejecutó una cadena de verificación:
1. **Paridad de evaluación**: el `evaluate_model` original reproduce exactamente el episodio seed-42 del script nuevo (-0.8880%) → no había bug de evaluación.
2. **Determinismo**: mismas seeds → mismos episodios entre procesos distintos ✓.
3. **Dispersión de seeds**: las 30 seeds de ABIDES extraídas están bien repartidas (sin clustering ni correlación) → no hay sesgo mecánico del seeding.
4. **Regresión agente~B&H**: corr +0.99, pendiente ~10 → descubrimiento de la política degenerada, que explica TODOS los números anteriores a la vez (el "+0.61% de alpha" de MAPPO era 10 × una deriva favorable casual; el "-1.26%" era 10 × una deriva desfavorable casual).

Lección: los dos resultados "contradictorios" eran el mismo fenómeno (beta apalancada) muestreado en lotes con deriva distinta. Sin la comparación a exposición igualada, cualquiera de los dos se habría escrito erróneamente como alpha del agente.

### Archivos de evidencia
- `benchmarking_B&H/PPO_Transformer_v3_vs_baselines.png` / `.json` (incluye series de mid-price por episodio)
- `benchmarking_B&H/MAPPO_Dual_v3_vs_baselines.png` / `.json`
- `benchmarking_B&H/baselines_summary.json`
- Scripts: `benchmark_baselines.py` (corrida con seeds), `verify_parity.py` (verificación de paridad)

---

## 10. Apéndice — listado cronológico de commits relevantes

```
76b1986  expandir observación de 8 → 44 features (LOB completo)
da28872  normalizar reward por starting_cash (estabilidad PPO)
2e519b8  quitar condición de terminación temprana
8d58bf5  añadir penalización de inventario
3863139  clip de penalización de inventario (evitar explosión)
88c1e5f  añadir TransformerActorCritic
2b7ee7e  prevenir venta en corto, exponer mid_price, trackear holdings
9d12535  SubprocVecEnv + PPOTrainer multi-entorno
8250b5b  fix: evaluate_model saltaba el encoder del Transformer
0750327  métricas de entrenamiento + evaluate_model + train.py
9dc9eb9  hold_penalty + entropy_coef 0.005
5514dc3  opportunity cost en lugar de hold penalty
2dc2cfc  opportunity cost proporcional al capital ocioso
d02b07b  PnL real en eval
921f6de  eliminar inv_penalty (trading direccional, no market-making)
deeb5db  reward alineado con paper ABIDES-Gym
d4ec1cc  reward final = M2M − inv_penalty − opp_cost
f05ffac  GRU temporal memory (intento)
061b19f  subir entropy_coef 0.005→0.01 (intento de arreglar GRU)
3782832  revertir GRU (incompatible con PPO multi-env opción B)
0863a8e  resultados v1 (PnL -0.11%)
e00cb1e  frame stacking N=10
89054a9  resultados v2 (PnL +0.03%, mejora por contexto temporal)
28712cc  config v3: entropy_coef 0.02, 1M steps
7fa5a3e  resultados v3 (PnL +0.31%, Sharpe 0.14) + experimentos v4 + eval_final
17f8ab7  arquitectura MAPPO Dual + experimentos + conclusión
```

---

## 11. Posibles líneas futuras (para sección de "Trabajo Futuro")

1. **GRU/LSTM con TBPTT correcto (Opción A):** actualización secuencial real por entorno sin resetear el estado oculto entre minibatches — permitiría memoria de largo plazo sin los problemas de inconsistencia observados
2. **Agente de volatilidad:** red adicional que prediga la volatilidad futura a `t+n` y cuya función de recompensa compare la predicción con la volatilidad real observada — útil para escalar el tamaño de las posiciones dinámicamente
3. **Theta/umbral aprendido como tercer componente:** en vez de que el arbitrador decida directamente, un módulo que aprenda *cuándo* el mercado está suficientemente "claro" para actuar (relacionado con el agente de volatilidad)
4. **MAPPO real multi-agente:** múltiples agentes RL coexistiendo en la misma simulación ABIDES, cada uno con su propio portfolio, explorando dinámicas de mercado emergentes (cooperación/competencia)
5. **Validación en otros mercados sintéticos:** repetir el experimento con configuraciones ABIDES distintas (rmsc05, otros activos) para comprobar si la conclusión de "falta de señal direccional a corto plazo" es generalizable o específica de rmsc04
