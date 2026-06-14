import numpy as np
import pandas as pd
import os
import json
import google.generativeai as genai
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class LocalFoundationPredictor:
    """
    Local fallback using structural Holt-Winters Triple Exponential Smoothing
    to forecast agricultural price spreads out-of-sample without requiring internet.
    """
    def __init__(self, seasonal_periods=52):
        self.seasonal_periods = seasonal_periods

    def predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        # Prevent fitting errors with very short history
        if len(history) < 2 * self.seasonal_periods:
            period = max(4, len(history) // 4)
        else:
            period = self.seasonal_periods
            
        try:
            # Try Holt-Winters with additive trend and additive seasonality
            model = ExponentialSmoothing(
                history, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=period,
                damped_trend=True
            )
            fit = model.fit()
            fc = fit.forecast(steps)
            if np.any(np.isnan(fc)) or np.any(np.isinf(fc)):
                raise ValueError("NaNs in HW forecast")
            return np.array(fc, dtype=float)
        except Exception:
            try:
                # Fallback to simple Holt Linear Trend (no seasonality)
                model = ExponentialSmoothing(history, trend='add', damped_trend=True)
                fit = model.fit()
                fc = fit.forecast(steps)
                if np.any(np.isnan(fc)) or np.any(np.isinf(fc)):
                    raise ValueError("NaNs in Holt forecast")
                return np.array(fc, dtype=float)
            except Exception:
                # Ultimate fallback: smooth exponential decay back to historical mean
                mean_val = np.mean(history)
                last_val = history[-1]
                t = np.arange(1, steps + 1)
                decay = np.exp(-0.1 * t)
                return np.array(last_val * decay + mean_val * (1.0 - decay), dtype=float)


class LLMFoundationPredictor:
    """
    Zero-shot time-series foundation predictor powered by Gemini API.
    Converts numbers to text prompt and forecasts future steps based on macro global commodities.
    """
    def __init__(self):
        pass

    def predict(self, history: np.ndarray, steps: int, target_name: str) -> np.ndarray:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass
                
        if not api_key or api_key == "AIzaSyBzwUZZklyEAFde6GWoMel8o-WfrZobmLI":
            raise ValueError("No valid API key available for LLM forecasting")
            
        genai.configure(api_key=api_key)
        
        # Select best flash model available
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception:
                model = genai.GenerativeModel('gemini-2.0-flash')

        # Limit history to prevent huge token usage, last 104 weeks (2 years) is ample for zero-shot forecasting
        history_subset = list(history[-104:])
        history_str = ", ".join([f"{v:.2f}" for v in history_subset])
        
        prompt = f"""
        Act as a State-of-the-Art Time-Series Foundation Model (like Amazon Chronos-Large or Google TimesFM) 
        specializing in global grain commodities, macroeconomics, and agricultural spreads.
        
        We have a weekly time series representing the '{target_name}' premium/discount profile in Argentina (USD/Tn).
        Below is the last 104 weeks of historical values in chronological order:
        [{history_str}]
        
        Your task is to perform zero-shot out-of-sample forecasting for exactly the next {steps} weeks.
        
        Consider the following domain knowledge in commodity forecasting:
        1. Mean reversion: Premium spreads typically revert back toward their long-term equilibrium but with inertia.
        2. Seasonality: Annual crop calendars (harvest vs off-season) drive cyclical peaks and troughs.
        3. Momentum and smooth continuation of the recent trend.
        
        Return ONLY a raw JSON object with a single key "forecast" whose value is a list of exactly {steps} floats. 
        Example: {{"forecast": [220.1, 221.3, 222.0]}}
        Do NOT write any markdown blocks (e.g. ```json), explanations, or additional text. Just the raw JSON.
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up any potential markdown formatting from LLM response
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
            
        data = json.loads(text)
        fc = np.array(data["forecast"], dtype=float)
        
        if len(fc) != steps:
            raise ValueError(f"Expected {steps} forecast steps, got {len(fc)}")
            
        return fc


class FoundationTimeSeriesPredictor:
    """
    Hybrid zero-shot forecasting engine that tries Gemini LLM zero-shot prediction
    and falls back to a structural Holt-Winters model in case of any failure.
    """
    def __init__(self, seasonal_periods=52):
        self.local_predictor = LocalFoundationPredictor(seasonal_periods=seasonal_periods)
        self.llm_predictor = LLMFoundationPredictor()

    def predict(self, history: np.ndarray, steps: int, target_name: str = "precio_trigo") -> np.ndarray:
        try:
            # Try zero-shot LLM forecasting
            fc = self.llm_predictor.predict(history, steps, target_name)
            # Sanity checks on the predictions
            if np.any(np.isnan(fc)) or np.any(np.isinf(fc)):
                raise ValueError("NaNs in LLM forecast")
            # Bound check: prevent extreme LLM predictions
            hist_min = np.min(history)
            hist_max = np.max(history)
            # Allow some extrapolation but clip crazy values
            fc = np.clip(fc, hist_min * 0.5, hist_max * 1.5)
            return fc
        except Exception as e:
            # Fallback to local high-fidelity Holt-Winters model
            # print(f"[FOUNDATION] Fallback active due to: {e}")
            return self.local_predictor.predict(history, steps)
