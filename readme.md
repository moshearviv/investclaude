# 📊 Advanced Stock Analyzer

מערכת ניתוח מניות מתקדמת עם ניתוח טכני, יסודי וסנטימנט עבור מניות S&P 500.

## ✨ תכונות

- 📈 **ניתוח טכני**: RSI, Moving Averages, Momentum
- 📊 **ניתוח יסודי**: P/E Ratio, ROE, Debt-to-Equity
- 🎯 **המלצות אוטומטיות**: BUY/SELL/HOLD עם ציוני ביטחון
- 📋 **סריקת S&P 500**: ניתוח אוטומטי של כל המדד
- 🔄 **עיבוד מקבילי**: ניתוח מהיר של מניות רבות

## 🚀 התקנה מהירה

```bash
# שכפול הפרויקט
git clone https://github.com/[your-username]/advanced-stock-analyzer.git
cd advanced-stock-analyzer

# התקנת תלויות
pip install -r requirements.txt

# הרצה בסיסית
python stock_analyzer.py
```

## 💻 שימוש בסיסי

```python
from stock_analyzer import StockAnalyzer

# יצירת analyzer
analyzer = StockAnalyzer()

# ניתוח מניה בודדת
result = analyzer.analyze_stock('AAPL')
print(f"המלצה: {result.recommendation}")
print(f"ציון: {result.overall_score:.1f}/100")

# ניתוח מספר מניות
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
results = analyzer.analyze_multiple_stocks(tickers)

# קבלת המלצות
buy_recs, sell_recs = analyzer.get_recommendations(results)
print(f"המלצות קנייה: {len(buy_recs)}")
print(f"המלצות מכירה: {len(sell_recs)}")
```

## 📋 מבנה הפרויקט

```
advanced-stock-analyzer/
├── stock_analyzer.py      # המנוע הראשי
├── config.py             # הגדרות מערכת
├── requirements.txt      # תלויות Python
├── README.md            # תיעוד זה
└── data_cache/          # תיקיית cache (נוצרת אוטומטית)
```

## ⚙️ הגדרות

ניתן להתאים הגדרות בקובץ `config.py`:

- **מספר מניות לניתוח**: `DEFAULT_NUM_STOCKS = 50`
- **ציון מינימלי**: `DEFAULT_MIN_SCORE = 60`
- **סף RSI**: `RSI_OVERSOLD = 30`, `RSI_OVERBOUGHT = 70`

## 📊 דוגמאות תוצאות

### המלצת קנייה
```
AAPL - Apple Inc.
├─ ציון כללי: 78.5/100
├─ טכני: 75.0 (RSI: 45, מעל SMA)
├─ יסודי: 82.0 (P/E: 18.5, ROE: 25%)
└─ המלצה: BUY (ביטחון: 85%)
```

### המלצת מכירה
```
XYZ - Example Corp
├─ ציון כללי: 22.3/100
├─ טכני: 25.0 (RSI: 75, מתחת SMA)
├─ יסודי: 18.0 (P/E: 45, ROE: -5%)
└─ המלצה: STRONG_SELL (ביטחון: 90%)
```

## 🛠️ פיתוח

המערכת בנויה להיות מודולרית וניתנת להרחבה:

### הוספת אינדיקטורים חדשים
```python
def calculate_new_indicator(self, data):
    # לוגיקת האינדיקטור החדש
    return indicator_value
```

### הוספת קריטריונים יסודיים
```python
# בconfig.py
class FundamentalThresholds:
    NEW_METRIC_MIN = 0.1
    NEW_METRIC_MAX = 0.5
```

## 🧪 בדיקות

```bash
# בדיקה בסיסית
python stock_analyzer.py

# בדיקה עם מניות ספציפיות
python -c "
from stock_analyzer import StockAnalyzer
analyzer = StockAnalyzer()
result = analyzer.analyze_stock('AAPL')
print('✅ Test passed!' if result else '❌ Test failed!')
"
```

## 📈 תוכניות עתידיות

- [ ] **ממשק Streamlit** - ממשק גרפי אינטראקטיבי
- [ ] **ניתוח סנטימנט** - אינטגרציה עם Twitter/News APIs
- [ ] **למידת מכונה** - מודלי חיזוי מתקדמים
- [ ] **התראות real-time** - מעקב שינויים בזמן אמת
- [ ] **אינטגרציה עם ברוקרים** - מסחר אוטומטי

## ⚠️ הגבלות וסיכונים

- **נתונים לא בזמן אמת** - עיכוב של 15-20 דקות
- **לא להסתמך לבד על המערכת** - תמיד עשה מחקר נוסף
- **הגבלות API** - yfinance מוגבל בקצב בקשות
- **לא ייעוץ השקעות** - זה כלי עזר בלבד

## 🤝 תרומה

המערכת פתוחה לשיפורים:

1. Fork את הפרויקט
2. צור branch חדש (`git checkout -b feature/amazing-feature`)
3. Commit השינויים (`git commit -m 'Add amazing feature'`)
4. Push ל-branch (`git push origin feature/amazing-feature`)
5. פתח Pull Request

## 📄 רישיון

MIT License - ראה קובץ LICENSE לפרטים

## 📞 תמיכה

- **Issues**: פתח issue ב-GitHub
- **שאלות**: צור discussion בפרויקט
- **תיעוד נוסף**: Wiki של הפרויקט

---

**💡 זוכר: השקע רק כסף שאתה יכול להרשות לעצמך להפסיד!**

**🚀 נבנה עם ❤️ לקהילת המשקיעים הישראליים**
