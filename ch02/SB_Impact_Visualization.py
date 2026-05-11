"""
Sleeping Beauty Patent Visualization Module

This module provides heliocentric and temporal visualization methods for analyzing
technology evolution patterns in patent citation networks. It implements visual
analytics frameworks for technology foresight, enabling simultaneous interpretation
of temporal, geographic, and technological dimensions of innovation diffusion.

The heliocentric metaphor positions scientific discovery at the center with
technology domains orbiting outward, reflecting increasing temporal and conceptual
distance from the core knowledge. This visualization strategy reveals:
- Technology emergence patterns (new domains entering the ecosystem)
- Innovation clustering (concentration vs. distribution of activity)
- Diffusion rhythms (temporal waves of adoption)

Compatible with Lens.org patent exports containing CPC classifications,
jurisdiction metadata, and citation metrics.

Example:
    Basic usage for heliocentric technology mapping:
        
        >>> analyzer = LensPatentAnalyzer('folkman_patents.csv')
        >>> analyzer.extract_technology_domains()
        >>> fig = analyzer.plot_heliocentric_technology_evolution()
        >>> plt.show()

Reference:
    Visualization methodology draws on temporal flow analysis principles from:
    Börner, K. (2015). Atlas of Knowledge. MIT Press.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pycountry


class LensPatentAnalyzer:
    """
    Dynamic analyzer for Lens.org patent data with heliocentric visualizations.
    
    This class processes patent metadata to create visual representations of
    technology evolution, geographic diffusion, and citation network dynamics.
    All analyses are data-driven without hard-coded parameters, adapting to
    the temporal span and domain composition of the input dataset.
    
    Attributes:
        df (pd.DataFrame): Main patent dataset with temporal periods
        tech_df (pd.DataFrame): Technology domain expansion data
        geo_df (pd.DataFrame): Geographic distribution data
        cpc_mapping (dict): CPC code to technology category mapping
        country_mapping (dict): Jurisdiction code to country name mapping
    """
    
    def __init__(self, csv_file_path):
        """
        Initialize analyzer and load patent data.
        
        Args:
            csv_file_path: Path to Lens.org CSV export file
        
        Side Effects:
            Automatically creates temporal periods (5-year bins) for longitudinal
            analysis during data preprocessing
        """
        self.df = self._load_and_preprocess_data(csv_file_path)
        self.tech_df = None
        self.geo_df = None
        self.cpc_mapping = self._build_cpc_mapping()
        self.country_mapping = self._build_country_mapping()
        
    def _load_and_preprocess_data(self, file_path):
        """
        Load patent data and perform temporal preprocessing.
        
        Preprocessing steps:
        1. Convert date fields to datetime format
        2. Extract publication year if not present
        3. Create 5-year period bins for trend analysis
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            pd.DataFrame: Preprocessed patent data with Period column
        
        Notes:
            Five-year periods align with standard technology foresight horizons
            and enable detection of medium-term innovation cycles.
        """
        df = pd.read_csv(file_path)
        
        # Convert date columns to datetime format
        date_columns = ['Publication Date', 'Application Date', 
                       'Earliest Priority Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create Publication Year column if missing
        if 'Publication Year' not in df.columns and 'Publication Date' in df.columns:
            df['Publication Year'] = df['Publication Date'].dt.year
        
        # Create dynamic 5-year period bins
        if 'Publication Year' in df.columns:
            min_year = int(df['Publication Year'].min())
            max_year = int(df['Publication Year'].max())
            
            # Align to period boundaries (e.g., 1995, 2000, 2005)
            start_year = (min_year // 5) * 5
            end_year = ((max_year // 5) + 1) * 5 + 1
            
            bins = list(range(start_year, end_year, 5))
            labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]
            
            df['Period'] = pd.cut(df['Publication Year'], bins=bins, 
                                labels=labels, include_lowest=True)
        
        return df
    
    def _build_cpc_mapping(self):
        """
        Construct technology domain mapping from CPC classifications.
        
        Extracts 3-character CPC codes (e.g., A61, C12) and maps them to
        human-readable technology categories. Uses domain-specific names
        for common pharmaceutical/biotechnology codes (A6x, C1x) while
        defaulting to CPC section names for others.
        
        Returns:
            dict: Mapping from CPC codes to technology domain names
        
        Notes:
            CPC (Cooperative Patent Classification) is maintained jointly by
            the European Patent Office (EPO) and United States Patent and
            Trademark Office (USPTO). Section codes (A-H, Y) represent
            broad technology areas.
        """
        cpc_mapping = {}
        
        # Collect unique CPC codes from dataset
        all_cpc_codes = set()
        if 'CPC Classifications' in self.df.columns:
            for cpc_str in self.df['CPC Classifications'].dropna():
                codes = str(cpc_str).split(';;')
                for code in codes:
                    main_code = code.strip()[:3]
                    if len(main_code) == 3 and main_code[0].isalpha():
                        all_cpc_codes.add(main_code)
        
        # CPC section names (official classification)
        cpc_categories = {
            'A': 'Human Necessities',
            'B': 'Operations & Transport', 
            'C': 'Chemistry & Metallurgy',
            'D': 'Textiles',
            'E': 'Fixed Constructions',
            'F': 'Mechanical Engineering',
            'G': 'Physics',
            'H': 'Electricity',
            'Y': 'Emerging Technologies'
        }
        
        # Create subcategory mappings
        for code in all_cpc_codes:
            main_section = code[0]
            if main_section in cpc_categories:
                sub_category = code[:2]
                
                # Domain-specific names for biotechnology and pharmaceuticals
                specific_mappings = {
                    'A6': 'Medical/Health',
                    'C0': 'Chemistry',
                    'C1': 'Biochemistry', 
                    'G0': 'Measuring/Testing',
                    'B0': 'Physical Processes',
                    'G1': 'Computing',
                    'H0': 'Communication',
                    'B8': 'Nanotechnology',
                    'Y0': 'Green Technologies',
                    'Y1': 'Technical Subjects'
                }
                
                if sub_category in specific_mappings:
                    cpc_mapping[code] = specific_mappings[sub_category]
                else:
                    cpc_mapping[code] = f"{cpc_categories[main_section]} - {code}"
        
        return cpc_mapping
    
    def _build_country_mapping(self):
        """
        Create jurisdiction code to country name mapping.
        
        Uses pycountry library for ISO 3166 alpha-2 code resolution,
        with special handling for international patent authorities
        (EP, WO) that don't correspond to countries.
        
        Returns:
            dict: Mapping from jurisdiction codes to full names
        
        Notes:
            Jurisdiction field in Lens.org data indicates the patent
            authority where the application was filed (e.g., US for USPTO,
            EP for European Patent Office).
        """
        country_mapping = {}
        
        # Extract unique jurisdictions from dataset
        if 'Jurisdiction' in self.df.columns:
            jurisdictions = self.df['Jurisdiction'].dropna().unique()
            
            for jurisdiction in jurisdictions:
                try:
                    # Attempt ISO country code lookup
                    country = pycountry.countries.get(alpha_2=jurisdiction)
                    if country:
                        country_mapping[jurisdiction] = country.name
                    else:
                        # Manual mapping for international authorities
                        special_cases = {
                            'EP': 'European Patent Office',
                            'WO': 'World Intellectual Property Organization',
                            'EU': 'European Union'
                        }
                        country_mapping[jurisdiction] = special_cases.get(
                            jurisdiction, jurisdiction)
                except Exception:
                    country_mapping[jurisdiction] = jurisdiction
        
        return country_mapping
    
    def extract_technology_domains(self):
        """
        Extract and structure technology domain data from CPC classifications.
        
        Parses CPC codes from patent records and creates a normalized
        dataset linking each patent to its technology domains across
        temporal periods.
        
        Returns:
            pd.DataFrame: Technology domain expansion data with columns:
                - Patent_ID: Record index
                - Period: Temporal period
                - Jurisdiction: Filing authority
                - Technology: Human-readable domain name
                - CPC_Code: Original CPC code
                - Publication_Year: Year of patent publication
                - Title: Patent title
        
        Notes:
            Single patents may map to multiple technology domains if they
            have multiple CPC classifications, reflecting technological
            convergence patterns common in complex innovations.
        """
        tech_data = []
        
        for idx, row in self.df.iterrows():
            if pd.notna(row.get('CPC Classifications')):
                cpc_codes = str(row['CPC Classifications']).split(';;')
                for code in cpc_codes:
                    main_code = code.strip()[:3]
                    if main_code in self.cpc_mapping:
                        tech_data.append({
                            'Patent_ID': idx,
                            'Period': row.get('Period'),
                            'Jurisdiction': row.get('Jurisdiction'),
                            'Technology': self.cpc_mapping[main_code],
                            'CPC_Code': main_code,
                            'Publication_Year': row.get('Publication Year'),
                            'Title': row.get('Title', '')
                        })
        
        self.tech_df = pd.DataFrame(tech_data)
        return self.tech_df
    
    def create_geographic_distribution(self):
        """
        Aggregate patent distribution by period and jurisdiction.
        
        Creates period-jurisdiction matrix showing the temporal evolution
        of geographic innovation spread.
        
        Returns:
            pd.DataFrame: Geographic distribution data with columns:
                - Period: Temporal period
                - Jurisdiction: Filing authority code
                - Patent_Count: Number of patents
                - Country: Full country/authority name
        
        Notes:
            Geographic diversity metrics (count of jurisdictions) indicate
            global diffusion intensity, while concentration patterns reveal
            innovation leadership and technology transfer pathways.
        """
        if 'Period' in self.df.columns and 'Jurisdiction' in self.df.columns:
            geo_data = (self.df.groupby(['Period', 'Jurisdiction'])
                       .size().reset_index(name='Patent_Count'))
            
            geo_data['Country'] = geo_data['Jurisdiction'].map(
                self.country_mapping).fillna(geo_data['Jurisdiction'])
            
            self.geo_df = geo_data
            return self.geo_df
        else:
            return pd.DataFrame()
    
    def plot_heliocentric_technology_evolution(self):
        """
        Create heliocentric visualization of technology domain evolution.
        
        Generates a multi-panel plot where each subplot represents one temporal
        period. Within each panel:
        - Center: Represents the core scientific discovery
        - Orbits: Technology domains at varying distances
        - Circle size: Number of patents in that domain
        - Angular position: Distributed to minimize overlap
        
        Returns:
            matplotlib.figure.Figure: Heliocentric visualization with subplots
                for each temporal period
        
        Visualization Design:
            The heliocentric metaphor reflects temporal flow theory where
            innovation radiates outward from core knowledge. Early periods
            show concentrated activity near the center, while later periods
            exhibit distributed multi-domain ecosystems.
        
        Notes:
            Requires tech_df to be populated via extract_technology_domains().
            Number of subplots adapts dynamically to dataset temporal span.
        """
        if self.tech_df is None:
            self.extract_technology_domains()
        
        periods = self.tech_df['Period'].dropna().unique()
        periods = sorted([p for p in periods if pd.notna(p)])
        
        # Determine subplot grid dimensions
        n_periods = len(periods)
        cols = min(3, n_periods)
        rows = (n_periods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle('Heliocentric Technology Evolution Map\n'
                    'Science-Technology Linkage Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Color mapping for technology domains
        unique_techs = self.tech_df['Technology'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_techs)))
        tech_colors = dict(zip(unique_techs, colors))
        
        # Generate heliocentric plot for each period
        for idx, period in enumerate(periods):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            period_data = self.tech_df[self.tech_df['Period'] == period]
            
            if period_data.empty:
                ax.text(0.5, 0.5, f'{period}\nNo data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Aggregate patents by technology domain
            tech_counts = period_data.groupby('Technology').size()
            
            # Draw central core (scientific discovery)
            center = Circle((0, 0), 0.1, color='gold', alpha=0.8, 
                          label='Science Core')
            ax.add_patch(center)
            
            # Calculate domain positions on orbits
            n_techs = len(tech_counts)
            angles = np.linspace(0, 2*np.pi, n_techs, endpoint=False)
            
            # Distance from center based on logarithmic scaling
            max_count = tech_counts.max()
            
            for i, (tech, count) in enumerate(tech_counts.items()):
                angle = angles[i]
                
                # Logarithmic radial positioning
                radius = 1 + np.log(count + 1) / np.log(max_count + 1)
                
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                # Circle size proportional to patent count
                circle_radius = 0.1 + 0.3 * (count / max_count)
                
                color = tech_colors[tech]
                circle = Circle((x, y), circle_radius, color=color, 
                              alpha=0.6, edgecolor='black', linewidth=1.5)
                ax.add_patch(circle)
                
                # Add labels for larger domains
                if count > tech_counts.median():
                    label_offset = circle_radius + 0.15
                    label_x = x + label_offset * np.cos(angle)
                    label_y = y + label_offset * np.sin(angle)
                    
                    # Shorten long technology names
                    tech_short = tech[:15] + '...' if len(tech) > 15 else tech
                    ax.text(label_x, label_y, f'{tech_short}\n({count})', 
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', 
                                   alpha=0.7, edgecolor='gray'))
            
            # Configure axes
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'{period}\n{len(period_data)} patents', 
                        fontweight='bold', fontsize=11)
        
        # Hide unused subplots
        for idx in range(n_periods, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_heliocentric_geographic_evolution(self):
        """
        Create heliocentric visualization of geographic diffusion.
        
        Similar to technology evolution plot but with jurisdictions as
        orbiting entities. Reveals patterns of:
        - Innovation leadership (dominant jurisdictions)
        - Technology transfer (emergence of new jurisdictions)
        - Global vs. regional innovation patterns
        
        Returns:
            matplotlib.figure.Figure: Heliocentric geographic visualization
        
        Notes:
            Requires geo_df to be populated via create_geographic_distribution().
        """
        if self.geo_df is None or self.geo_df.empty:
            self.create_geographic_distribution()
        
        if self.geo_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No geographic data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        periods = sorted(self.geo_df['Period'].dropna().unique())
        
        # Determine subplot grid
        n_periods = len(periods)
        cols = min(3, n_periods)
        rows = (n_periods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle('Heliocentric Geographic Evolution\n'
                    'Global Innovation Diffusion Patterns', 
                    fontsize=16, fontweight='bold')
        
        # Color mapping for jurisdictions
        unique_countries = self.geo_df['Country'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_countries)))
        country_colors = dict(zip(unique_countries, colors))
        
        # Generate plots for each period
        for idx, period in enumerate(periods):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            period_data = self.geo_df[self.geo_df['Period'] == period]
            
            if period_data.empty:
                ax.text(0.5, 0.5, f'{period}\nNo data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Draw central core
            center = Circle((0, 0), 0.1, color='gold', alpha=0.8)
            ax.add_patch(center)
            
            # Calculate positions
            n_countries = len(period_data)
            angles = np.linspace(0, 2*np.pi, n_countries, endpoint=False)
            
            max_count = period_data['Patent_Count'].max()
            
            for i, (_, row) in enumerate(period_data.iterrows()):
                angle = angles[i]
                count = row['Patent_Count']
                country = row['Country']
                
                # Radial positioning
                radius = 1 + np.log(count + 1) / np.log(max_count + 1)
                
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                # Circle size
                circle_radius = 0.1 + 0.3 * (count / max_count)
                
                color = country_colors.get(country, 'gray')
                circle = Circle((x, y), circle_radius, color=color, 
                              alpha=0.6, edgecolor='black', linewidth=1.5)
                ax.add_patch(circle)
                
                # Labels for significant jurisdictions
                if count > period_data['Patent_Count'].median():
                    label_offset = circle_radius + 0.15
                    label_x = x + label_offset * np.cos(angle)
                    label_y = y + label_offset * np.sin(angle)
                    
                    country_short = (country[:12] + '...' 
                                   if len(country) > 12 else country)
                    ax.text(label_x, label_y, f'{country_short}\n({count})', 
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', 
                                   alpha=0.7, edgecolor='gray'))
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'{period}\n{period_data["Patent_Count"].sum()} patents', 
                        fontweight='bold', fontsize=11)
        
        # Hide unused subplots
        for idx in range(n_periods, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_temporal_trend_analysis(self):
        """
        Generate comprehensive temporal trend visualizations.
        
        Creates 2x2 subplot grid showing:
        1. Annual patent publication volume
        2. Technology domain composition (stacked area)
        3. Geographic distribution evolution (stacked area)
        4. Citation metrics over time
        
        Returns:
            matplotlib.figure.Figure: Temporal trend analysis dashboard
        
        Notes:
            Stacked area charts reveal domain dominance patterns and
            convergence/divergence trends. Citation metrics indicate
            knowledge flow intensity and technological impact evolution.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        fig.suptitle('Temporal Trend Analysis\n'
                    'Patent Volume, Domain Evolution, and Citation Dynamics', 
                    fontsize=14, fontweight='bold')
        
        # 1. Annual patent volume
        if 'Publication Year' in self.df.columns:
            yearly_counts = self.df['Publication Year'].value_counts().sort_index()
            ax1.bar(yearly_counts.index, yearly_counts.values, 
                   color='steelblue', alpha=0.7)
            ax1.set_title('Annual Patent Publications', fontweight='bold')
            ax1.set_xlabel('Publication Year')
            ax1.set_ylabel('Number of Patents')
            ax1.grid(True, alpha=0.3, axis='y')
        else:
            ax1.text(0.5, 0.5, 'No temporal data available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Technology domain evolution (stacked area)
        if self.tech_df is not None and not self.tech_df.empty:
            tech_pivot = (self.tech_df.groupby(['Publication_Year', 'Technology'])
                         .size().unstack(fill_value=0))
            if not tech_pivot.empty:
                tech_pivot.plot(kind='area', ax=ax2, alpha=0.7, stacked=True)
                ax2.set_title('Technology Domain Composition', 
                             fontweight='bold')
                ax2.set_xlabel('Publication Year')
                ax2.set_ylabel('Number of Patents')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No technology data available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Geographic distribution evolution (stacked area)
        if 'Jurisdiction' in self.df.columns:
            geo_pivot = (self.df.groupby(['Publication Year', 'Jurisdiction'])
                        .size().unstack(fill_value=0))
            if not geo_pivot.empty:
                geo_pivot.plot(kind='area', ax=ax3, alpha=0.7, stacked=True)
                ax3.set_title('Geographic Distribution Evolution', 
                             fontweight='bold')
                ax3.set_xlabel('Publication Year')
                ax3.set_ylabel('Number of Patents')
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No geographic data available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Citation metrics over time
        citation_cols = ['Cites Patent Count', 'Cited by Patent Count', 
                        'NPL Citation Count']
        available_cols = [col for col in citation_cols if col in self.df.columns]
        
        if available_cols:
            for col in available_cols:
                yearly_avg = self.df.groupby('Publication Year')[col].mean()
                ax4.plot(yearly_avg.index, yearly_avg.values, marker='o', 
                        label=col, linewidth=2)
            
            ax4.set_title('Citation Metrics Evolution', fontweight='bold')
            ax4.set_xlabel('Publication Year')
            ax4.set_ylabel('Average Citation Count')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No citation data available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self):
        """
        Generate structured textual analysis report.
        
        Produces multi-section report covering:
        - General statistics (total patents, temporal span, diversity metrics)
        - Period analysis (patents per period, jurisdictions, citations)
        - Technology domain analysis (patent distribution, temporal spread)
        - Geographic analysis (jurisdiction distribution, concentration)
        
        Returns:
            str: Formatted report text suitable for file export
        
        Notes:
            Report complements visual analytics by providing quantitative
            evidence for qualitative patterns observed in visualizations.
        """
        report = []
        report.append("=" * 60)
        report.append("SCIENCE-TECHNOLOGY LINKAGE ANALYSIS REPORT")
        report.append("=" * 60)
        
        # General statistics
        report.append("\nGENERAL STATISTICS:")
        report.append(f"Total Patents: {len(self.df)}")
        
        if 'Publication Year' in self.df.columns:
            min_year = self.df['Publication Year'].min()
            max_year = self.df['Publication Year'].max()
            report.append(f"Temporal Coverage: {min_year}-{max_year} "
                         f"({max_year - min_year + 1} years)")
        
        if 'Jurisdiction' in self.df.columns:
            report.append("Number of Jurisdictions: "
                         f"{self.df['Jurisdiction'].nunique()}")
        
        if self.tech_df is not None:
            report.append("Number of Technology Domains: "
                         f"{self.tech_df['Technology'].nunique()}")
        
        # Period analysis
        if 'Period' in self.df.columns:
            report.append("\nPERIOD-WISE ANALYSIS:")
            
            agg_dict = {'Publication Year': 'count'}
            
            if 'Jurisdiction' in self.df.columns:
                agg_dict['Jurisdiction'] = 'nunique'
            
            # Add citation metrics if available
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in ['Cites Patent Count', 'Cited by Patent Count']:
                if col in numeric_cols:
                    agg_dict[col] = 'mean'
            
            period_stats = self.df.groupby('Period').agg(agg_dict).round(2)
            
            for period, stats in period_stats.iterrows():
                if pd.notna(period):
                    report.append(f"\n{period}:")
                    report.append(f"  Patents: {int(stats['Publication Year'])}")
                    
                    if 'Jurisdiction' in stats:
                        report.append(f"  Jurisdictions: {int(stats['Jurisdiction'])}")
                    
                    for col in agg_dict:
                        if col not in ['Publication Year', 'Jurisdiction']:
                            report.append(f"  Average {col}: {stats[col]:.2f}")
        
        # Technology domain analysis
        if self.tech_df is not None and not self.tech_df.empty:
            report.append("\nTECHNOLOGY DOMAIN DISTRIBUTION:")
            
            tech_stats = (self.tech_df.groupby('Technology')
                         .agg({'Patent_ID': 'nunique', 
                              'Period': lambda x: x.dropna().nunique()})
                         .sort_values('Patent_ID', ascending=False))
            
            for tech, stats in tech_stats.iterrows():
                report.append(f"  {tech}: {stats['Patent_ID']} patents "
                             f"across {stats['Period']} periods")
        
        # Geographic analysis
        if self.geo_df is not None and not self.geo_df.empty:
            report.append("\nGEOGRAPHIC DISTRIBUTION:")
            
            geo_stats = (self.geo_df.groupby('Country')['Patent_Count']
                        .sum().sort_values(ascending=False))
            
            total_patents = geo_stats.sum()
            for country, count in geo_stats.items():
                percentage = (count / total_patents) * 100
                report.append(f"  {country}: {int(count)} patents "
                             f"({percentage:.1f}%)")
        
        return "\n".join(report)
    
    def run_complete_analysis(self, save_figures=True):
        """
        Execute complete visual analytics pipeline.
        
        Orchestrates full analysis workflow:
        1. Extract technology domains from CPC codes
        2. Create geographic distribution aggregations
        3. Generate heliocentric visualizations (technology and geography)
        4. Create temporal trend analysis
        5. Produce comprehensive textual report
        
        Args:
            save_figures: Save all figures and report to files (default: True)
        
        Returns:
            dict: Analysis results containing:
                - figures: Dictionary of matplotlib Figure objects
                - report: Formatted text report
                - data: DataFrames for main, technology, and geographic data
        
        Side Effects:
            When save_figures=True, creates:
            - technology_heliocentric.png
            - geographic_heliocentric.png
            - temporal_trends.png
            - analysis_report.txt
        """
        print("Extracting technology domains from CPC classifications...")
        self.extract_technology_domains()
        
        print("Aggregating geographic distribution...")
        self.create_geographic_distribution()
        
        print("Generating heliocentric visualizations...")
        
        figures = {}
        
        # Technology heliocentric map
        fig1 = self.plot_heliocentric_technology_evolution()
        if fig1 and save_figures:
            fig1.savefig('technology_heliocentric.png', dpi=300, 
                        bbox_inches='tight')
        figures['technology_heliocentric'] = fig1
        
        # Geographic heliocentric map
        fig2 = self.plot_heliocentric_geographic_evolution()
        if fig2 and save_figures:
            fig2.savefig('geographic_heliocentric.png', dpi=300, 
                        bbox_inches='tight')
        figures['geographic_heliocentric'] = fig2
        
        # Temporal trend analysis
        fig3 = self.create_temporal_trend_analysis()
        if fig3 and save_figures:
            fig3.savefig('temporal_trends.png', dpi=300, bbox_inches='tight')
        figures['temporal_trends'] = fig3
        
        # Generate comprehensive report
        print("Compiling analysis report...")
        report = self.generate_comprehensive_report()
        
        if save_figures:
            with open('analysis_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
        
        print("\nAnalysis completed successfully!")
        if save_figures:
            print("\nGenerated files:")
            print("  - technology_heliocentric.png")
            print("  - geographic_heliocentric.png")
            print("  - temporal_trends.png")
            print("  - analysis_report.txt")
        
        return {
            'figures': figures,
            'report': report,
            'data': {
                'main_df': self.df,
                'tech_df': self.tech_df,
                'geo_df': self.geo_df
            }
        }


def main():
    """
    Main execution function demonstrating analyzer usage.
    
    Analyzes patent citations to Folkman's (1971) angiogenesis paper,
    demonstrating heliocentric visualization methods for technology
    foresight applications.
    
    The heliocentric metaphor reveals how pharmaceutical and biotechnology
    domains emerged from core angiogenesis science, followed by expansion
    into nanotechnology, green technologies, and medical devices in later
    periods.
    """
    # Example: Folkman patent citations
    file_path = "judahfolkmanpatentciters.csv"
    
    print("Initializing Lens.org patent analyzer...")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = LensPatentAnalyzer(file_path)
    
    # Run complete analysis
    print("\nRunning complete visual analytics pipeline...")
    results = analyzer.run_complete_analysis(save_figures=True)
    
    # Display report
    print("\n" + "=" * 60)
    print("ANALYSIS REPORT")
    print("=" * 60)
    print(results['report'])
    
    # Display figures
    print("\nDisplaying visualizations...")
    for fig_name, fig in results['figures'].items():
        if fig is not None:
            plt.show()


if __name__ == "__main__":
    main()
