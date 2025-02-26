import vectorbt as vbt
import numpy as np
import pandas as pd

start = '2019-01-01 UTC'
end = '2020-01-01 UTC'
price = vbt.YFData.download('BTC-USD', start=start, end=end).get('Close')

ma_20 = vbt.MA.run(price, 20)
ma_50 = vbt.MA.run(price, 30)

entries = ma_20.ma_crossed_below(ma_50)
exits = ma_20.ma_crossed_above(ma_50)

pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)

print(pf.stats())
