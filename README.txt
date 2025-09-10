Options Flow Summarizer — PRO (Weighted Bias)
Adds a per-key-level **weighted bullish/bearish precedence** score and label.
Heuristics (tunable in code):
- Base polarity: Call=+1, Put=-1
- Positioning: Vol/OI > 1.0 adds +1.0*sign (+ magnitude), Vol/OI < 0.5 subtracts 0.7*sign (- magnitude)
- Mega-blocks (>=500 contracts share > 20%) add +0.6*sign
- Zone: ATM multiplies *1.10*, support adds +0.10*sign, resistance adds -0.10*sign
- Premium scaling: multiplies by log1p(premium)/15 to weight higher-dollar levels
Outputs columns: `weighted_bias_label`, `weighted_bias_score`