import numpy as np
import matplotlib.pyplot as plt

# Time points
t = np.linspace(0, 10, 1000)

# External market price (solid line) - shows price movement
external_price = np.ones_like(t) * 100  # Initial flat price at 100
external_price[300:500] += np.linspace(0, 40, 200)  # Price increases
external_price[500:700] = 140  # Price stays at the higher level
external_price[700:] = 120  # Price partially drops

# AMM bid-ask spread (dotted lines)
spread_width = 10  # Width between bid and ask

# Initial AMM prices (before external price moves)
initial_mid_price = 100
initial_bid = initial_mid_price - spread_width/2
initial_ask = initial_mid_price + spread_width/2

# Time periods for different AMM price regimes
t1, t2, t3, t4 = 300, 500, 700, 1000

# AMM bid price (dotted line)
amm_bid = np.zeros_like(t)
amm_bid[:t1] = initial_bid  # Initial period
amm_bid[t1:t3] = external_price[t1:t3] - spread_width/2  # Follows external price up
amm_bid[t3:] = external_price[t3:] - spread_width/2  # Follows external price down

# AMM ask price (dotted line)
amm_ask = np.zeros_like(t)
amm_ask[:t1] = initial_ask  # Initial period
amm_ask[t1:t3] = external_price[t1:t3] + spread_width/2  # Follows external price up
amm_ask[t3:] = external_price[t3:] + spread_width/2  # Follows external price down

# Create the plot
plt.figure(figsize=(12, 8))
plt.grid(True, linestyle='--', alpha=0.7)

# Plot the external market price (solid line)
plt.plot(t, external_price, 'k-', linewidth=2, label='External Market Price')

# Plot the AMM bid-ask prices (dotted lines)
plt.plot(t, amm_bid, 'k--', linewidth=2, label='AMM Bid Price')
plt.plot(t, amm_ask, 'k--', linewidth=2, label='AMM Ask Price')

# Add labels and title
plt.xlabel('Time', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title('AMM Price Adjustment to External Market Movements', fontsize=14)
plt.legend(loc='best')

# Annotate key points
plt.annotate('AMM adjusts to\nexternal price', xy=(400, 130), xytext=(350, 150),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
plt.annotate('AMM follows\nprice decrease', xy=(750, 120), xytext=(750, 140),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# Set y-axis limits with some padding
plt.ylim(85, 155)

# Save the figure
plt.savefig('amm_price_adjustment.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show() 