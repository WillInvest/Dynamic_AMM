# %%
import numpy as np
import matplotlib.pyplot as plt

def process_real_data(df, gamma):
    df = df.sort_values(by='Date')
    y_initial = y0 = df['PX_LAST'].iloc[0]
    x_initial = x0 = 1
    p0 = y0/x0
    pa = p0/(1-gamma)
    pb = p0*(1-gamma)
    L = np.sqrt(y0*x0)
    fee = 0

    paths = []
    paths.append({
        'date': df['Date'].iloc[0],
        's1': df['PX_LAST'].iloc[0],
        'pa': pa,
        'pb': pb,
        'p0': p0,
        'y': y0,
        'x': x0,
        'fee': fee,
        'hold_value': x_initial * df['PX_LAST'].iloc[0] + y_initial + fee,
        'LP_value': x_initial * p0 + y_initial + fee
    })

    for i in range(len(df)-1):
        s1 = df['PX_LAST'].iloc[i+1]
        
        if s1 > pa:
            y1 = L * np.sqrt((1-gamma)*s1)
            x1 = L**2 / y1
            fee += gamma/(1-gamma) * (y1 - y0)
        elif s1 < pb:
            y1 = L * np.sqrt(s1/(1-gamma))
            x1 = L**2 / y1
            fee += gamma/(1-gamma) * (x1 - x0) * s1
        else:
            y1 = y0
            x1 = x0
            
        y0 = y1
        x0 = x1
        p0 = y0/x0
        pa = p0/(1-gamma)
        pb = p0*(1-gamma)
        hold_value = x_initial * s1 + y_initial
        LP_value = x0 * s1 + y0 + fee
            
        paths.append({
            'date': df['Date'].iloc[i],
            's1': s1,
            'pa': pa,
            'pb': pb,
            'p0': p0,
            'y': y0,
            'x': x0,
            'fee': fee,
            'hold_value': x_initial * s1 + y_initial,
            'LP_value': x0 * s1 + y0 + fee
        })
        
    paths = pd.DataFrame(paths)
    print(pd.concat([paths.head(1), paths.tail(1)]).to_markdown())
    
    return hold_value, LP_value




# %%

import pandas as pd
import numpy as np

df = pd.read_excel('10year_tesla.xlsx', header=6)
df = df.sort_values(by='Date')
print(df.head().to_markdown())

results = []
for gamma in np.arange(0.001, 0.999, 0.001):
    hold_value, LP_value = process_real_data(df, gamma=gamma)
    results.append({
        'gamma': gamma,
        'hold_value': hold_value,
        'LP_value': LP_value
    })
    
results = pd.DataFrame(results)

print(results.head().to_markdown())





# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(results['gamma'], results['hold_value'], label='Hold Value', linestyle='--')
plt.plot(results['gamma'], results['LP_value'], label='LP Value')
plt.title('Tesla Stock Hold Value and LP Value')
plt.xlabel('Gamma')
plt.ylabel('Value')
plt.legend()
plt.show()

# %%


print(results.tail(100).to_markdown())


# %%
df = pd.read_excel('10year_tesla.xlsx', header=6)
hold_value, LP_value = process_real_data(df, gamma=0.003)




# %%
99.8196 * 1.00022
# %%

0.9 * np.sqrt(365)
