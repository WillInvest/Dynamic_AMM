import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

class AccountingProfitAnalyzer:
    """
    A class to analyze factors affecting accounting profit in AMM pools.
    """
    
    def __init__(self, data_path, output_dir='./output/profit_analysis'):
        """
        Initialize the analyzer with data path and output directory.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing the data
        output_dir : str
            Directory to save analysis outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the data."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Add binary classification columns for profit
        self.df['incoming_profit_positive'] = (self.df['accounting_profit_incoming'] > 0).astype(int)
        self.df['outgoing_profit_positive'] = (self.df['accounting_profit_outgoing'] > 0).astype(int)
        
        print(f"Data loaded. Shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        
        # Basic statistics
        print("\nBasic statistics:")
        print(f"Total rows: {len(self.df)}")
        print(f"Rows with positive incoming profit: {self.df['incoming_profit_positive'].sum()} ({self.df['incoming_profit_positive'].mean()*100:.2f}%)")
        print(f"Rows with positive outgoing profit: {self.df['outgoing_profit_positive'].sum()} ({self.df['outgoing_profit_positive'].mean()*100:.2f}%)")
    
    def summarize_factors(self):
        """Summarize the distribution of factors in the dataset."""
        factors = ['x', 'gamma', 'sigma', 'drift', 'relative_p']
        
        print("\nFactor distributions:")
        for factor in factors:
            unique_values = self.df[factor].unique()
            print(f"{factor}: {len(unique_values)} unique values - {sorted(unique_values)}")
    
    def analyze_factor_impact(self, factor, profit_type='incoming'):
        """
        Analyze the impact of a specific factor on profit.
        
        Parameters:
        -----------
        factor : str
            The factor to analyze ('x', 'gamma', 'sigma', 'drift', or 'relative_p')
        profit_type : str
            Type of profit to analyze ('incoming' or 'outgoing')
        """
        profit_col = f'accounting_profit_{profit_type}'
        positive_col = f'{profit_type}_profit_positive'
        
        plt.figure(figsize=(12, 6))
        
        # Group by factor and calculate percentage of positive profit
        grouped = self.df.groupby(factor)[positive_col].mean() * 100
        
        # Plot percentage of positive profit by factor
        ax = grouped.plot(kind='bar', color='skyblue')
        plt.title(f'Percentage of Positive {profit_type.capitalize()} Profit by {factor}')
        plt.xlabel(factor)
        plt.ylabel('Percentage of Positive Profit (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(grouped):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{profit_type}_profit_by_{factor}.png')
        plt.close()
        
        # Calculate average profit by factor
        avg_profit = self.df.groupby(factor)[profit_col].mean()
        
        plt.figure(figsize=(12, 6))
        avg_profit.plot(kind='bar', color='lightgreen')
        plt.title(f'Average {profit_type.capitalize()} Profit by {factor}')
        plt.xlabel(factor)
        plt.ylabel('Average Profit')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{profit_type}_avg_profit_by_{factor}.png')
        plt.close()
        
        return grouped, avg_profit
    
    def analyze_all_factors(self):
        """Analyze the impact of all factors on both profit types."""
        factors = ['x', 'gamma', 'sigma', 'drift', 'relative_p']
        
        results = {}
        for profit_type in ['incoming', 'outgoing']:
            results[profit_type] = {}
            for factor in factors:
                positive_pct, avg_profit = self.analyze_factor_impact(factor, profit_type)
                results[profit_type][factor] = {
                    'positive_percentage': positive_pct,
                    'average_profit': avg_profit
                }
        
        return results
    
    def create_heatmaps(self):
        """Create heatmaps to visualize the interaction between factors."""
        factor_pairs = [
            ('gamma', 'sigma'),
            ('gamma', 'relative_p'),
            ('sigma', 'relative_p'),
            ('drift', 'sigma'),
            ('x', 'gamma')
        ]
        
        for x_factor, y_factor in factor_pairs:
            for profit_type in ['incoming', 'outgoing']:
                positive_col = f'{profit_type}_profit_positive'
                
                # Create pivot table
                pivot = pd.pivot_table(
                    self.df, 
                    values=positive_col,
                    index=y_factor,
                    columns=x_factor,
                    aggfunc=np.mean
                ) * 100
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt='.1f', linewidths=.5)
                plt.title(f'Percentage of Positive {profit_type.capitalize()} Profit\n{y_factor} vs {x_factor}')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/{profit_type}_heatmap_{y_factor}_vs_{x_factor}.png')
                plt.close()
    
    def build_predictive_model(self, profit_type='incoming'):
        """
        Build a predictive model to identify important factors.
        
        Parameters:
        -----------
        profit_type : str
            Type of profit to analyze ('incoming' or 'outgoing')
        """
        positive_col = f'{profit_type}_profit_positive'
        features = ['x', 'gamma', 'sigma', 'drift', 'relative_p', 'p0']
        
        # Prepare data
        X = self.df[features]
        y = self.df[positive_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        print(f"\nRandom Forest Model for {profit_type.capitalize()} Profit:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
        plt.title(f'Feature Importance for {profit_type.capitalize()} Profit')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{profit_type}_feature_importance.png')
        plt.close()
        
        return model, importance
    
    def analyze_threshold_effects(self, factor, profit_type='incoming'):
        """
        Analyze if there are threshold effects for a given factor.
        
        Parameters:
        -----------
        factor : str
            The factor to analyze
        profit_type : str
            Type of profit to analyze ('incoming' or 'outgoing')
        """
        profit_col = f'accounting_profit_{profit_type}'
        
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot
        plt.scatter(self.df[factor], self.df[profit_col], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
        plt.title(f'{profit_type.capitalize()} Profit vs {factor}')
        plt.xlabel(factor)
        plt.ylabel(f'{profit_type.capitalize()} Profit')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{profit_type}_threshold_{factor}.png')
        plt.close()
    
    def run_comprehensive_analysis(self):
        """Run a comprehensive analysis of all factors."""
        print("\nStarting comprehensive analysis...")
        
        # Summarize factors
        self.summarize_factors()
        
        # Analyze individual factors
        self.analyze_all_factors()
        
        # Create heatmaps
        self.create_heatmaps()
        
        # Build predictive models
        for profit_type in ['incoming', 'outgoing']:
            self.build_predictive_model(profit_type)
        
        # Analyze threshold effects
        for factor in ['x', 'gamma', 'sigma', 'drift', 'relative_p']:
            for profit_type in ['incoming', 'outgoing']:
                self.analyze_threshold_effects(factor, profit_type)
        
        print("\nAnalysis complete. Results saved to:", self.output_dir)

if __name__ == "__main__":
    # Initialize analyzer with the complete dataset
    analyzer = AccountingProfitAnalyzer('comprehensive_comparison_results.csv')
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis() 