"""
Roadmap Generation and Visualization Module
Chapter 12: Data-Led Technology Roadmapping

This module provides functions for technology maturity assessment
and integrated roadmap visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def assess_technology_maturity(df: pd.DataFrame,
                               date_column: str = 'Year') -> dict:
    """
    Assess technology maturity using multiple indicators.
    
    Parameters:
        df: DataFrame with publication/patent data
        date_column: Name of date column
        
    Returns:
        Dictionary with maturity assessment results
    """
    # Extract year
    if hasattr(df[date_column].iloc[0], 'year'):
        years = df[date_column].dt.year
    else:
        years = df[date_column]
    
    current_year = years.max()
    recent_years = 3
    
    # Calculate indicators
    total_docs = len(df)
    recent_docs = len(df[years >= current_year - recent_years])
    growth_rate = (recent_docs / total_docs) * 100 if total_docs > 0 else 0
    
    # Calculate year-over-year growth
    yearly_counts = df.groupby(years).size()
    yoy_growth = yearly_counts.pct_change().tail(3).mean() * 100
    
    # Maturity classification
    if yoy_growth > 30 and recent_docs > 50:
        maturity = 'Emerging'
        description = 'Rapid growth phase with accelerating innovation'
    elif yoy_growth > 15:
        maturity = 'Growth'
        description = 'Strong growth with increasing market interest'
    elif yoy_growth > 0:
        maturity = 'Mature'
        description = 'Stable technology with incremental improvements'
    else:
        maturity = 'Declining'
        description = 'Decreasing activity, potential obsolescence'
    
    result = {
        'maturity': maturity,
        'description': description,
        'total_documents': total_docs,
        'recent_documents': recent_docs,
        'recent_share': growth_rate,
        'yoy_growth': yoy_growth,
        'time_span': f"{years.min()}-{years.max()}"
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("Technology Maturity Assessment")
    print(f"{'='*60}")
    print(f"  Maturity Stage: {maturity}")
    print(f"  Description: {description}")
    print(f"  Total Documents: {total_docs:,}")
    print(f"  Recent Documents (last 3 years): {recent_docs:,} ({growth_rate:.1f}%)")
    print(f"  Year-over-Year Growth: {yoy_growth:.1f}%")
    print(f"  Time Span: {result['time_span']}")
    
    return result


def create_maturity_gauge(maturity: str, save_path: str = 'maturity_gauge.png'):
    """
    Create a visual gauge showing technology maturity stage.
    
    Parameters:
        maturity: Maturity stage ('Emerging', 'Growth', 'Mature', 'Declining')
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    stages = ['Emerging', 'Growth', 'Mature', 'Declining']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    # Draw stages
    for i, (stage, color) in enumerate(zip(stages, colors)):
        alpha = 1.0 if stage == maturity else 0.3
        ax.barh(0, 1, left=i, color=color, alpha=alpha, edgecolor='white', linewidth=2)
        ax.text(i + 0.5, 0, stage, ha='center', va='center', 
                fontsize=12, fontweight='bold' if stage == maturity else 'normal')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title('Technology Maturity Stage', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_technology_roadmap(temporal_data: pd.Series,
                              topic_evolution: pd.DataFrame = None,
                              forecast_data: pd.Series = None,
                              maturity: str = None,
                              title: str = 'Data-Led Technology Roadmap',
                              save_path: str = 'technology_roadmap.png'):
    """
    Create comprehensive technology roadmap visualization.
    
    Parameters:
        temporal_data: Series with yearly document counts
        topic_evolution: DataFrame with topic proportions by year
        forecast_data: Series with forecasted values
        maturity: Technology maturity stage
        title: Plot title
        save_path: Path to save figure
    """
    # Determine layout based on available data
    has_topics = topic_evolution is not None
    n_panels = 2 if has_topics else 1
    
    fig = plt.figure(figsize=(16, 5 * n_panels))
    gs = GridSpec(n_panels, 1, hspace=0.3)
    
    # Panel 1: Historical trends + forecast
    ax1 = fig.add_subplot(gs[0])
    
    temporal_data.plot(ax=ax1, color='steelblue', linewidth=2.5, 
                       marker='o', markersize=8, label='Historical')
    
    if forecast_data is not None:
        forecast_years = range(temporal_data.index[-1] + 1,
                              temporal_data.index[-1] + len(forecast_data) + 1)
        ax1.plot(forecast_years, forecast_data.values, color='red',
                linestyle='--', linewidth=2.5, marker='s', 
                markersize=8, label='Forecast')
        
        # Add shaded forecast region
        ax1.axvspan(forecast_years[0], forecast_years[-1], 
                    alpha=0.1, color='red', label='Forecast Period')
    
    ax1.set_title('Technology Activity Timeline', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Documents', fontsize=12)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add maturity annotation
    if maturity:
        ax1.annotate(f'Maturity: {maturity}', 
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=12, fontweight='bold',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Topic evolution (if available)
    if has_topics:
        ax2 = fig.add_subplot(gs[1])
        topic_evolution.plot(kind='area', stacked=True, ax=ax2, 
                            alpha=0.7, colormap='tab10')
        
        ax2.set_title('Research Theme Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Topic Proportion', fontsize=12)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nRoadmap saved to: {save_path}")


def generate_roadmap_report(temporal_data: pd.Series,
                            maturity_result: dict,
                            centrality_df: pd.DataFrame = None,
                            topics: dict = None) -> str:
    """
    Generate text-based roadmap report.
    
    Parameters:
        temporal_data: Series with yearly counts
        maturity_result: Dictionary from assess_technology_maturity()
        centrality_df: DataFrame from network centrality analysis
        topics: Dictionary of topics from topic modeling
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("TECHNOLOGY ROADMAP REPORT")
    report.append("=" * 70)
    
    # Overview
    report.append("\n1. OVERVIEW")
    report.append("-" * 40)
    report.append(f"   Analysis Period: {maturity_result['time_span']}")
    report.append(f"   Total Documents Analyzed: {maturity_result['total_documents']:,}")
    report.append(f"   Technology Maturity: {maturity_result['maturity']}")
    report.append(f"   Assessment: {maturity_result['description']}")
    
    # Growth Analysis
    report.append("\n2. GROWTH ANALYSIS")
    report.append("-" * 40)
    report.append(f"   Recent Activity (last 3 years): {maturity_result['recent_documents']:,} documents")
    report.append(f"   Share of Total: {maturity_result['recent_share']:.1f}%")
    report.append(f"   Year-over-Year Growth: {maturity_result['yoy_growth']:.1f}%")
    
    # Key Technologies (if available)
    if centrality_df is not None and len(centrality_df) > 0:
        report.append("\n3. KEY TECHNOLOGIES (by network centrality)")
        report.append("-" * 40)
        top_keywords = centrality_df.head(10)['Keyword'].tolist()
        for i, kw in enumerate(top_keywords, 1):
            report.append(f"   {i}. {kw}")
    
    # Research Themes (if available)
    if topics:
        report.append("\n4. RESEARCH THEMES")
        report.append("-" * 40)
        for topic_name, words in topics.items():
            report.append(f"   {topic_name}: {', '.join(words[:5])}")
    
    # Recommendations
    report.append("\n5. STRATEGIC RECOMMENDATIONS")
    report.append("-" * 40)
    
    maturity = maturity_result['maturity']
    if maturity == 'Emerging':
        report.append("   • High growth potential - consider early investment")
        report.append("   • Monitor key players and patent filings")
        report.append("   • Build partnerships with research institutions")
    elif maturity == 'Growth':
        report.append("   • Strong momentum - scale up R&D activities")
        report.append("   • Establish competitive position through IP")
        report.append("   • Develop commercialization strategies")
    elif maturity == 'Mature':
        report.append("   • Focus on incremental improvements")
        report.append("   • Explore adjacent technology areas")
        report.append("   • Consider cost optimization strategies")
    else:
        report.append("   • Evaluate exit or pivot strategies")
        report.append("   • Identify replacement technologies")
        report.append("   • Focus resources on emerging alternatives")
    
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    return report_text


def compare_publication_patent_trends(pub_df: pd.DataFrame,
                                      pat_df: pd.DataFrame,
                                      pub_date_col: str = 'Year',
                                      pat_date_col: str = 'Filing_Year',
                                      save_path: str = 'pub_pat_comparison.png'):
    """
    Compare publication and patent trends over time.
    
    Parameters:
        pub_df: Publication DataFrame
        pat_df: Patent DataFrame
        pub_date_col: Publication date column name
        pat_date_col: Patent date column name
        save_path: Path to save figure
    """
    # Extract years
    if hasattr(pub_df[pub_date_col].iloc[0], 'year'):
        pub_years = pub_df[pub_date_col].dt.year
    else:
        pub_years = pub_df[pub_date_col]
    
    pub_counts = pub_df.groupby(pub_years).size()
    pat_counts = pat_df.groupby(pat_date_col).size()
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    color1 = 'steelblue'
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Publications', color=color1, fontsize=12)
    ax1.plot(pub_counts.index, pub_counts.values, 'o-', 
             color=color1, linewidth=2.5, markersize=8, label='Publications')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'coral'
    ax2.set_ylabel('Patents', color=color2, fontsize=12)
    ax2.plot(pat_counts.index, pat_counts.values, 's--', 
             color=color2, linewidth=2.5, markersize=8, label='Patents')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    plt.title('Publication vs Patent Trends', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate and print lag
    common_years = set(pub_counts.index) & set(pat_counts.index)
    if common_years:
        correlation = pub_counts.loc[list(common_years)].corr(
            pat_counts.loc[list(common_years)])
        print(f"\nPublication-Patent Correlation: {correlation:.3f}")


# Example usage
if __name__ == "__main__":
    print("Roadmap Generation Module - Chapter 12")
    print("Functions: assess_technology_maturity(), create_technology_roadmap()")
    print("See README.md for complete documentation")
