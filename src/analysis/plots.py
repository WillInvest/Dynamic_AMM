import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize lists to store the extracted values
a_values = []
b_values = []
fb_values = []

# Regular expression patterns to match A, B, and FB values
pattern_a = r"(?<!F)A: ([\d.]+)"
pattern_b = r"(?<!F)B: ([\d.]+)"
pattern_fb = r"FA: ([\d.]+)"
# Read the file
with open("BA10Percentage.txt", "r") as file:
    content = file.read()

    # Find all occurrences of A, B, and FB and their values
    a_values = [float(value) for value in re.findall(pattern_a, content)]
    b_values = [float(value) for value in re.findall(pattern_b, content)]
    fb_values = [float(value) for value in re.findall(pattern_fb, content)]
    # print(len(a_values))
    # print(len(b_values))


    
    # for values in fb_values:
    #     values = values * -1
    
# Normalize the values

max_a = max(a_values)
max_b = max(b_values)
max_fb = max(fb_values)


# a_values_normalized = [x / max_a for x in a_values]
# b_values_normalized = [x / max_b for x in b_values]
#fb_values_normalized = [x / max_fb for x in fb_values]

# Create subplots
fig = make_subplots(rows=3, cols=1, subplot_titles=("Inventory A", "Inventory B", "Fees charged"))

# Add traces for A, B, and FB in separate subplots
fig.add_trace(go.Scatter(x=list(range(len(a_values))), y=a_values, mode='lines', name='Inventory A'), row=1, col=1)
fig.add_trace(go.Scatter(x=list(range(len(b_values))), y=b_values, mode='lines', name='Inventory B'), row=2, col=1)
fig.add_trace(go.Scatter(x=list(range(len(fb_values))), y=fb_values, mode='lines', name='Fees charged'), row=3, col=1)

# Update layout for a cleaner look
fig.update_layout(height=600, title_text="Experiment results for transaction B A 10 using Percentage Fee", template='plotly_white')

# Show the plot
fig.show()
