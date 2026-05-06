"""
SpyMarketMakerEnv — Gym environment para entrenar un agente de Market Making
sobre el simulador ABIDES (config rmsc04).

Espacio de observación: 44 features
    - 10 niveles del LOB en el lado bid  (precio_bps + log_volumen) → 20
    - 10 niveles del LOB en el lado ask  (precio_bps + log_volumen) → 20
    - 4 features de portfolio: posición normalizada, cash normalizado,
      PnL no realizado, progreso temporal del episodio              →  4

Espacio de acción: 11 acciones discretas
    - 0       : HOLD (cancela órdenes activas y no coloca nuevas)
    - 1..5    : LIMIT BID en niveles 0..4 del libro
    - 6..10   : LIMIT ASK en niveles 0..4 del libro

Reward: compuesto (4 términos del execution_plan del MVP)
    realized_pnl  -  inv_penalty  +  quoting_bonus  -  adverse_selection

Esta extensión amplía el POC de Diego (8 obs / 3 acciones market) al diseño
completo del MVP. Se mantiene como archivo separado de spy_gym_env.py para
que el POC siga sirviendo de sanity check rápido en Colab.
"""
