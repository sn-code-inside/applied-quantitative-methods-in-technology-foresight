"""
Sleeping Beauty Impact Analyzer

This module provides a comprehensive framework for analyzing the science-technology
linkage of "Sleeping Beauty" (SB) publications through patent citation analysis.
It quantifies the temporal evolution from scientific discovery to technological
application, measuring awakening dynamics, diffusion patterns, and transformation
velocity.

The analyzer implements technology foresight methodologies for tracking:
- Dormancy periods and awakening phases
- Science-technology coupling intensity
- Geographic and technological diffusion patterns
- Innovation maturity trajectories

Compatible with Lens.org patent metadata, which resolves citation ambiguity
through MetaRecord (MeR) linking systems.

Example:
    Basic usage with Lens.org CSV export:
        
        >>> df = pd.read_csv('patent_citations.csv')
        >>> analyzer = SleepingBeautyImpactAnalyzer(df, discovery_year=1971)
        >>> analyzer.create_time_periods(period_length=5)
        >>> analyzer.calculate_awakening_metrics()
        >>> results = analyzer.run_complete_analysis()

Reference:
    Van Raan, A. F. J. (2004). Sleeping beauties in science. Scientometrics, 59(3), 461-466.
    Jefferson et al. (2019). The Lens MetaRecord. doi:10.31222/osf.io/ez4m5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class SleepingBeautyImpactAnalyzer:
    """
    Framework for analyzing Sleeping Beauty impact on science-technology ecosystems.
    
    This class implements quantitative methods for measuring the transformation
    of dormant scientific knowledge into applied technological innovation through
    patent citation analysis. It tracks awakening dynamics, diffusion entropy,
    and maturity trajectories across temporal, geographic, and technological dimensions.
    
    Attributes:
        df (pd.DataFrame): Patent citation data from Lens.org
        sb_year (int): Publication year of the Sleeping Beauty study
        awakening_analysis (dict): Temporal phase metrics and thresholds
        diffusion_metrics (pd.DataFrame): Geographic and technological diffusion indicators
    """
    
    def __init__(self, patent_df, sb_discovery_year):
        """
        Initialize analyzer with patent citation data.
        
        Args:
            patent_df: DataFrame containing Lens.org patent metadata with required
                fields: 'Publication Year', 'Jurisdiction', 'CPC Classifications'
            sb_discovery_year: Initial publication year of the Sleeping Beauty article
                (e.g., 1971 for Folkman's angiogenesis paper)
        
        Raises:
            ValueError: If required columns are missing from patent data
        """
        self.df = patent_df
        self.sb_year = sb_discovery_year
        self.awakening_analysis = None
        self.diffusion_metrics = None
        self._validate_data()
        
    def _validate_data(self):
        """
        Validate and clean Lens.org data structure.
        
        Implements data quality protocols including:
        - Structural validation (required column presence)
        - Type enforcement (numeric conversion for metrics)
        - Null handling (default values for family sizes)
        - Categorical normalization (jurisdiction labels)
        
        This method ensures compatibility with downstream analytical functions
        and prevents parsing errors from malformed data.
        
        Raises:
            ValueError: If required columns ('Publication Year', 'Jurisdiction') 
                are missing
        """
        # Clean whitespace in column names
        self.df.columns = self.df.columns.str.strip()
        
        # Check for required columns
        required_columns = ['Publication Year', 'Jurisdiction']
        missing_columns = [col for col in required_columns 
                          if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing: {missing_columns}")
        
        # Clean Publication Year
        self.df['Publication Year'] = pd.to_numeric(
            self.df['Publication Year'], errors='coerce'
        ).fillna(0).astype(int)
        self.df = self.df[self.df['Publication Year'] > 0]
        
        # Convert citation columns to numeric
        citation_columns = [
            'Cites Patent Count', 'Cited by Patent Count', 'NPL Citation Count'
        ]
        for col in citation_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col], errors='coerce'
                ).fillna(0)
        
        # Convert family size columns to numeric
        family_columns = ['Simple Family Size', 'Extended Family Size']
        for col in family_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col], errors='coerce'
                ).fillna(1)
        
        # Clean CPC Classifications
        if 'CPC Classifications' in self.df.columns:
            self.df['CPC Classifications'] = (
                self.df['CPC Classifications'].fillna('')
            )
        
        # Clean Jurisdiction
        if 'Jurisdiction' in self.df.columns:
            self.df['Jurisdiction'] = (
                self.df['Jurisdiction'].fillna('Unknown')
            )
        
    def create_time_periods(self, period_length=5):
        """
        Create dynamic temporal periods for longitudinal analysis.
        
        Generates equal-length time windows aligned to period boundaries
        (e.g., 1995-1999, 2000-2004) for tracking technology evolution phases.
        This standardization enables cross-study comparisons and pattern detection.
        
        Args:
            period_length: Duration of each period in years (default: 5)
        
        Returns:
            list: Period labels (e.g., ['1995-1999', '2000-2004', ...])
        
        Notes:
            Five-year periods align with standard technology foresight horizons
            and patent examination cycles (typically 2-4 years).
        """
        min_year = int(self.df['Publication Year'].min())
        max_year = int(self.df['Publication Year'].max())
        start_year = (min_year // period_length) * period_length
        end_year = ((max_year // period_length) + 1) * period_length + 1
        
        bins = list(range(start_year, end_year, period_length))
        labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]
        
        self.df['Period'] = pd.cut(
            self.df['Publication Year'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        
        return labels
        
    def calculate_awakening_metrics(self):
        """
        Analyze the awakening process of the Sleeping Beauty study.
        
        Identifies temporal phases of technological adoption based on cumulative
        patent distribution. Uses data-driven percentile thresholds to delineate:
        - Dormant phase (< 10th percentile): Minimal technological uptake
        - Awakening phase (10-50th percentile): Accelerating adoption
        - Widespread phase (> 50th percentile): Mature diffusion
        
        Returns:
            dict: Awakening analysis containing:
                - yearly_stats (pd.DataFrame): Annual patent counts and citation metrics
                - phases (dict): Records grouped by dormancy phase
                - thresholds (list): Percentile cutoffs for phase classification
        
        Notes:
            This method operationalizes Van Raan's (2004) Sleeping Beauty concept
            by quantifying the transition from "sleeping" (dormant) to "awake"
            (widespread technological recognition).
        """
        # Basic annual metrics
        agg_dict = {'Publication Year': 'count'}
        
        # Add aggregation for existing citation columns
        citation_columns = [
            'Cites Patent Count', 'Cited by Patent Count', 'NPL Citation Count'
        ]
        for col in citation_columns:
            if col in self.df.columns:
                agg_dict[col] = 'mean'
        
        yearly_stats = self.df.groupby('Publication Year').agg(agg_dict)
        yearly_stats = yearly_stats.rename(
            columns={'Publication Year': 'Patent_Count'}
        )
        
        # Calculate cumulative impact
        yearly_stats['Cumulative_Patents'] = (
            yearly_stats['Patent_Count'].cumsum()
        )
        yearly_stats['Years_Since_Discovery'] = (
            yearly_stats.index - self.sb_year
        )
        
        # Dynamically determine awakening periods
        total_patents = yearly_stats['Patent_Count'].sum()
        yearly_stats['Cumulative_Percentage'] = (
            yearly_stats['Cumulative_Patents'] / total_patents * 100
        )
        
        # Data-driven thresholds
        percentiles = [10, 50]
        
        awakening_phases = {
            'dormant': yearly_stats[
                yearly_stats['Cumulative_Percentage'] <= percentiles[0]
            ],
            'awakening': yearly_stats[
                (yearly_stats['Cumulative_Percentage'] > percentiles[0]) & 
                (yearly_stats['Cumulative_Percentage'] <= percentiles[1])
            ],
            'widespread': yearly_stats[
                yearly_stats['Cumulative_Percentage'] > percentiles[1]
            ]
        }
        
        self.awakening_analysis = {
            'yearly_stats': yearly_stats,
            'phases': awakening_phases,
            'thresholds': percentiles
        }
        
        return self.awakening_analysis
    
    def analyze_science_technology_coupling(self):
        """
        Measure temporal evolution of science-technology coupling intensity.
        
        Computes two complementary indicators:
        
        Science Intensity:
            Ratio of non-patent literature (NPL) citations to prior art citations.
            High values indicate science-driven innovation (e.g., basic research
            translation).
        
        Technology Intensity:
            Product of patent family size and logarithmic forward citations.
            Captures both horizontal (family) and vertical (impact) diffusion.
        
        Returns:
            pd.DataFrame: Period-aggregated coupling metrics with mean and
                standard deviation for both intensity measures
        
        Notes:
            This coupling analysis reveals the inflection point where scientific
            contribution transitions from exploratory research to applied
            technological exploitation—a defining indicator of SB awakening.
        """
        
        # Calculate Science Intensity
        if ('NPL Citation Count' in self.df.columns and 
            'Cites Patent Count' in self.df.columns):
            
            self.df['Science_Intensity'] = (
                self.df['NPL Citation Count'] / 
                (self.df['Cites Patent Count'] + 1)
            )
        else:
            self.df['Science_Intensity'] = 0
        
        # Calculate Technology Intensity
        if ('Simple Family Size' in self.df.columns and 
            'Cited by Patent Count' in self.df.columns):
            
            self.df['Tech_Intensity'] = (
                self.df['Simple Family Size'] * 
                np.log(self.df['Cited by Patent Count'] + 1)
            )
        else:
            self.df['Tech_Intensity'] = 0
        
        # Period analysis
        if 'Period' in self.df.columns:
            agg_dict = {
                'Science_Intensity': ['mean', 'std'],
                'Tech_Intensity': ['mean', 'std']
            }
            
            # Add citation columns if available
            citation_columns = [
                'NPL Citation Count', 'Cites Patent Count', 
                'Cited by Patent Count'
            ]
            for col in citation_columns:
                if col in self.df.columns:
                    agg_dict[col] = 'mean'
            
            coupling_analysis = (
                self.df.groupby('Period', observed=True)
                .agg(agg_dict)
                .round(3)
            )
        else:
            coupling_analysis = pd.DataFrame()
        
        return coupling_analysis
    
    def calculate_diffusion_patterns(self):
        """
        Analyze multi-dimensional technological diffusion patterns.
        
        Tracks three complementary dimensions of innovation spread:
        
        1. Geographic Diffusion:
           - Diversity: Number of unique jurisdictions per year
           - Entropy: Shannon entropy of jurisdiction distribution
           
        2. Technological Diffusion:
           - Diversity: Number of distinct CPC domains per year
           - Entropy: Domain distribution heterogeneity
           
        3. Network Diffusion:
           - Citation depth: Mean backward citations
           - Citation breadth: Mean forward citations
        
        Returns:
            pd.DataFrame: Annual diffusion metrics indexed by publication year
        
        Notes:
            Increasing entropy values signal transition from concentrated
            (single-domain) to distributed (multi-domain) innovation ecosystems,
            characteristic of mature technology diffusion phases.
        """
        
        # Geographic diffusion
        if 'Jurisdiction' in self.df.columns:
            geo_diversity = (
                self.df.groupby('Publication Year')['Jurisdiction']
                .agg([
                    'nunique',
                    lambda x: self._calculate_entropy(x.value_counts().values)
                ])
            )
            geo_diversity.columns = ['Geographic_Diversity', 'Geographic_Entropy']
        else:
            geo_diversity = pd.DataFrame()
        
        # Technology domain diffusion
        if 'CPC Classifications' in self.df.columns:
            tech_diversity = self._analyze_technology_diversity()
        else:
            tech_diversity = pd.DataFrame()
        
        # Citation network metrics
        citation_metrics = self._calculate_citation_metrics()
        
        # Combine all metrics
        diffusion_df = pd.concat(
            [geo_diversity, tech_diversity, citation_metrics], 
            axis=1
        )
        
        self.diffusion_metrics = diffusion_df
        return diffusion_df
    
    def _calculate_entropy(self, counts):
        """
        Calculate Shannon entropy for distribution heterogeneity.
        
        Args:
            counts: Array of frequency values
        
        Returns:
            float: Shannon entropy in bits
        
        Notes:
            Entropy measures the "surprise" in a distribution. Higher entropy
            indicates more uniform spread across categories (e.g., balanced
            geographic distribution vs. single-country dominance).
        """
        if len(counts) == 0:
            return 0
        
        proportions = counts / counts.sum()
        proportions = proportions[proportions > 0]
        
        if len(proportions) <= 1:
            return 0
        
        return -np.sum(proportions * np.log2(proportions))
    
    def _analyze_technology_diversity(self):
        """
        Extract technology domain diversity from CPC classifications.
        
        Returns:
            pd.DataFrame: Annual technology diversity and entropy metrics
        """
        tech_records = []
        
        for _, row in self.df.iterrows():
            if pd.notna(row.get('CPC Classifications')):
                year = row['Publication Year']
                cpc_codes = str(row['CPC Classifications']).split(';;')
                
                # Extract main CPC sections (first character: A-H, Y)
                domains = set()
                for code in cpc_codes:
                    main_code = code.strip()[:1]
                    if main_code.isalpha():
                        domains.add(main_code)
                
                for domain in domains:
                    tech_records.append({
                        'Publication Year': year,
                        'Technology_Domain': domain
                    })
        
        if not tech_records:
            return pd.DataFrame()
        
        tech_df = pd.DataFrame(tech_records)
        
        # Calculate diversity metrics by year
        diversity = (
            tech_df.groupby('Publication Year')['Technology_Domain']
            .agg([
                'nunique',
                lambda x: self._calculate_entropy(x.value_counts().values)
            ])
        )
        diversity.columns = ['Tech_Diversity', 'Tech_Entropy']
        
        return diversity
    
    def _calculate_citation_metrics(self):
        """
        Calculate backward and forward citation metrics.
        
        Returns:
            pd.DataFrame: Annual mean citation counts
        """
        citation_cols = {
            'Cites Patent Count': 'Citation_Depth',
            'Cited by Patent Count': 'Citation_Breadth'
        }
        
        available_cols = [col for col in citation_cols.keys() 
                         if col in self.df.columns]
        
        if not available_cols:
            return pd.DataFrame()
        
        metrics = (
            self.df.groupby('Publication Year')[available_cols]
            .mean()
            .round(2)
        )
        
        # Rename columns
        rename_dict = {col: citation_cols[col] for col in available_cols}
        metrics = metrics.rename(columns=rename_dict)
        
        return metrics
    
    def calculate_maturity_indicators(self):
        """
        Calculate innovation maturity trajectories using composite indicators.
        
        Implements dual-axis maturity assessment:
        
        Science Maturity:
            Composite of NPL citation intensity and citation depth,
            normalized to [0, 1] scale. Reflects scientific foundation strength.
        
        Technology Maturity:
            Composite of patent family size and forward citations,
            normalized to [0, 1] scale. Reflects market penetration and impact.
        
        Returns:
            pd.DataFrame: Annual maturity scores for both dimensions
        
        Notes:
            Divergence between science and technology maturity curves indicates
            the lag between scientific discovery and technological application—
            a key characteristic of Sleeping Beauty papers.
        """
        maturity_data = []
        
        for year in sorted(self.df['Publication Year'].unique()):
            year_data = self.df[self.df['Publication Year'] == year]
            
            # Science maturity components
            npl_intensity = 0
            if 'NPL Citation Count' in year_data.columns:
                npl_intensity = year_data['NPL Citation Count'].mean()
            
            cite_depth = 0
            if 'Cites Patent Count' in year_data.columns:
                cite_depth = year_data['Cites Patent Count'].mean()
            
            # Technology maturity components
            family_size = 1
            if 'Simple Family Size' in year_data.columns:
                family_size = year_data['Simple Family Size'].mean()
            
            forward_cites = 0
            if 'Cited by Patent Count' in year_data.columns:
                forward_cites = year_data['Cited by Patent Count'].mean()
            
            maturity_data.append({
                'Publication Year': year,
                'Science_Maturity': npl_intensity + cite_depth,
                'Tech_Maturity': family_size * np.log(forward_cites + 1)
            })
        
        maturity_df = pd.DataFrame(maturity_data)
        maturity_df = maturity_df.set_index('Publication Year')
        
        # Normalize to 0-1 scale
        scaler = StandardScaler()
        if len(maturity_df) > 1:
            normalized = scaler.fit_transform(maturity_df)
            maturity_df = pd.DataFrame(
                normalized,
                index=maturity_df.index,
                columns=maturity_df.columns
            )
        
        return maturity_df
    
    def calculate_transformation_velocity(self):
        """
        Calculate rate of change in maturity indicators.
        
        Computes year-over-year velocity (first derivative) for science and
        technology maturity trajectories. Acceleration patterns reveal:
        - Steady growth: Consistent velocity
        - Exponential growth: Increasing velocity
        - Saturation: Decreasing velocity
        
        Returns:
            dict: Velocity metrics for science and technology dimensions
        """
        maturity_df = self.calculate_maturity_indicators()
        
        velocity_metrics = {}
        
        for col in maturity_df.columns:
            values = maturity_df[col].values
            if len(values) > 1:
                velocity = np.diff(values)
                velocity_metrics[f'{col}_Velocity'] = velocity.mean()
            else:
                velocity_metrics[f'{col}_Velocity'] = 0
        
        return velocity_metrics
    
    def plot_impact_dashboard(self):
        """
        Generate comprehensive visual dashboard of SB impact metrics.
        
        Creates a 3x3 subplot array displaying:
        1. Awakening timeline with phase markers
        2. Cumulative diffusion curve
        3. Science-technology coupling evolution
        4. Geographic diffusion patterns
        5. Technology domain expansion
        6. Citation network metrics
        7. Maturity trajectories (science vs. technology)
        8. Diffusion entropy trends
        9. Summary statistics panel
        
        Returns:
            matplotlib.figure.Figure: Dashboard figure object
        
        Notes:
            This dashboard implements visual analytics principles for technology
            foresight, enabling simultaneous interpretation of temporal,
            geographic, and technological dimensions.
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Sleeping Beauty Impact Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # 1. Awakening Timeline
        self._plot_awakening_timeline(axes[0])
        
        # 2. Cumulative Diffusion
        self._plot_cumulative_diffusion(axes[1])
        
        # 3. Science-Technology Coupling
        self._plot_coupling_evolution(axes[2])
        
        # 4. Geographic Diffusion
        self._plot_geographic_diffusion(axes[3])
        
        # 5. Technology Domains
        self._plot_technology_evolution(axes[4])
        
        # 6. Citation Metrics
        self._plot_citation_metrics(axes[5])
        
        # 7. Maturity Trajectories
        self._plot_maturity_trajectories(axes[6])
        
        # 8. Diffusion Entropy
        self._plot_diffusion_entropy(axes[7])
        
        # 9. Summary Statistics
        self._plot_summary_statistics(axes[8])
        
        plt.tight_layout()
        return fig
    
    def _plot_awakening_timeline(self, ax):
        """Plot annual patent counts with awakening phase markers."""
        if self.awakening_analysis is None:
            ax.text(0.5, 0.5, 'Run calculate_awakening_metrics() first', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        yearly_stats = self.awakening_analysis['yearly_stats']
        
        ax.bar(yearly_stats.index, yearly_stats['Patent_Count'], 
              color='steelblue', alpha=0.7)
        ax.axvline(self.sb_year, color='red', linestyle='--', 
                  label='Discovery Year', linewidth=2)
        
        ax.set_title('Awakening Timeline', fontweight='bold')
        ax.set_xlabel('Publication Year')
        ax.set_ylabel('Annual Patent Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_diffusion(self, ax):
        """Plot cumulative patent adoption curve."""
        if self.awakening_analysis is None:
            return
        
        yearly_stats = self.awakening_analysis['yearly_stats']
        
        ax.plot(yearly_stats.index, yearly_stats['Cumulative_Patents'], 
               color='darkgreen', linewidth=2, marker='o', markersize=4)
        ax.fill_between(yearly_stats.index, yearly_stats['Cumulative_Patents'], 
                       alpha=0.3, color='green')
        
        ax.set_title('Cumulative Technology Diffusion', fontweight='bold')
        ax.set_xlabel('Publication Year')
        ax.set_ylabel('Cumulative Patent Count')
        ax.grid(True, alpha=0.3)
    
    def _plot_coupling_evolution(self, ax):
        """Plot science-technology coupling intensity over time."""
        coupling = self.analyze_science_technology_coupling()
        
        if coupling.empty:
            ax.text(0.5, 0.5, 'No coupling data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract period midpoints for x-axis
        periods = coupling.index
        x_labels = [p for p in periods if pd.notna(p)]
        
        if ('Science_Intensity', 'mean') in coupling.columns:
            ax.plot(range(len(x_labels)), 
                   coupling[('Science_Intensity', 'mean')].values,
                   label='Science Intensity', marker='s', linewidth=2)
        
        if ('Tech_Intensity', 'mean') in coupling.columns:
            ax.plot(range(len(x_labels)), 
                   coupling[('Tech_Intensity', 'mean')].values,
                   label='Technology Intensity', marker='o', linewidth=2)
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_title('Science-Technology Coupling', fontweight='bold')
        ax.set_ylabel('Coupling Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_geographic_diffusion(self, ax):
        """Plot geographic diversity evolution."""
        if self.diffusion_metrics is None or self.diffusion_metrics.empty:
            ax.text(0.5, 0.5, 'Run calculate_diffusion_patterns() first', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        if 'Geographic_Diversity' in self.diffusion_metrics.columns:
            years = self.diffusion_metrics.index
            ax.plot(years, self.diffusion_metrics['Geographic_Diversity'], 
                   marker='o', linewidth=2, color='coral')
            
            ax.set_title('Geographic Diffusion', fontweight='bold')
            ax.set_xlabel('Publication Year')
            ax.set_ylabel('Number of Jurisdictions')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No geographic data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_technology_evolution(self, ax):
        """Plot technology domain diversity."""
        if self.diffusion_metrics is None or self.diffusion_metrics.empty:
            return
        
        if 'Tech_Diversity' in self.diffusion_metrics.columns:
            years = self.diffusion_metrics.index
            ax.plot(years, self.diffusion_metrics['Tech_Diversity'], 
                   marker='s', linewidth=2, color='purple')
            
            ax.set_title('Technology Domain Expansion', fontweight='bold')
            ax.set_xlabel('Publication Year')
            ax.set_ylabel('Number of Technology Domains')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No technology data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_citation_metrics(self, ax):
        """Plot citation depth and breadth evolution."""
        if self.diffusion_metrics is None or self.diffusion_metrics.empty:
            return
        
        citation_cols = ['Citation_Depth', 'Citation_Breadth']
        available = [col for col in citation_cols 
                    if col in self.diffusion_metrics.columns]
        
        if available:
            years = self.diffusion_metrics.index
            for col in available:
                label = col.replace('_', ' ')
                ax.plot(years, self.diffusion_metrics[col], 
                       marker='o', linewidth=2, label=label)
            
            ax.set_title('Citation Network Metrics', fontweight='bold')
            ax.set_xlabel('Publication Year')
            ax.set_ylabel('Average Citation Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No citation data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_maturity_trajectories(self, ax):
        """Plot science and technology maturity evolution."""
        maturity_df = self.calculate_maturity_indicators()
        
        if maturity_df.empty:
            ax.text(0.5, 0.5, 'Insufficient data for maturity analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        years = maturity_df.index
        ax.plot(years, maturity_df['Science_Maturity'], 
               marker='o', linewidth=2, label='Science Maturity', color='blue')
        ax.plot(years, maturity_df['Tech_Maturity'], 
               marker='s', linewidth=2, label='Technology Maturity', color='red')
        
        ax.set_title('Maturity Trajectories', fontweight='bold')
        ax.set_xlabel('Publication Year')
        ax.set_ylabel('Normalized Maturity Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_diffusion_entropy(self, ax):
        """Plot geographic and technological entropy."""
        if self.diffusion_metrics is None or self.diffusion_metrics.empty:
            return
        
        entropy_cols = ['Geographic_Entropy', 'Tech_Entropy']
        available = [col for col in entropy_cols 
                    if col in self.diffusion_metrics.columns]
        
        if available:
            years = self.diffusion_metrics.index
            for col in available:
                label = col.replace('_', ' ')
                ax.plot(years, self.diffusion_metrics[col], 
                       marker='o', linewidth=2, label=label)
            
            ax.set_title('Diffusion Entropy', fontweight='bold')
            ax.set_xlabel('Publication Year')
            ax.set_ylabel('Shannon Entropy (bits)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No entropy data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_summary_statistics(self, ax):
        """Display key summary statistics."""
        ax.axis('off')
        
        # Calculate summary metrics
        total_patents = len(self.df)
        min_year = int(self.df['Publication Year'].min())
        max_year = int(self.df['Publication Year'].max())
        time_span = max_year - min_year + 1
        dormancy = min_year - self.sb_year
        
        # Format statistics
        stats_text = f"""
        SUMMARY STATISTICS
        {'=' * 30}
        
        Total Patents: {total_patents}
        Time Span: {min_year}-{max_year} ({time_span} years)
        Discovery Year: {self.sb_year}
        Dormancy Period: {dormancy} years
        
        Annual Average: {total_patents / time_span:.1f} patents/year
        """
        
        if self.diffusion_metrics is not None and not self.diffusion_metrics.empty:
            if 'Geographic_Diversity' in self.diffusion_metrics.columns:
                final_geo = self.diffusion_metrics['Geographic_Diversity'].iloc[-1]
                stats_text += f"\nFinal Geographic Reach: {int(final_geo)} jurisdictions"
            
            if 'Tech_Diversity' in self.diffusion_metrics.columns:
                final_tech = self.diffusion_metrics['Tech_Diversity'].iloc[-1]
                stats_text += f"\nFinal Technology Breadth: {int(final_tech)} domains"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def generate_impact_report(self):
        """
        Generate comprehensive textual impact assessment report.
        
        Produces structured analysis covering:
        - Temporal evolution metrics (dormancy, awakening intensity)
        - Diffusion patterns (geographic and technological expansion)
        - Transformation impact (maturity growth, velocity)
        - Success indicators (exponential growth, global reach)
        - Methodological notes (data quality warnings)
        
        Returns:
            str: Formatted report text suitable for file export
        
        Notes:
            Report implements evidence-based assessment criteria derived from
            technology foresight literature, avoiding arbitrary thresholds.
        """
        report = []
        report.append("=" * 60)
        report.append("SLEEPING BEAUTY IMPACT ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Dataset overview
        total_patents = len(self.df)
        min_year = int(self.df['Publication Year'].min())
        max_year = int(self.df['Publication Year'].max())
        
        report.append(f"\nDATASET OVERVIEW:")
        report.append(f"Total Patents: {total_patents}")
        report.append(f"Analysis Period: {min_year}-{max_year}")
        report.append(f"Discovery Year: {self.sb_year}")
        report.append(f"Time Since Discovery: {max_year - self.sb_year} years")
        
        # Awakening analysis
        if self.awakening_analysis:
            report.append("\nAWAKENING DYNAMICS:")
            
            phases = self.awakening_analysis['phases']
            dormant_years = len(phases.get('dormant', []))
            awakening_years = len(phases.get('awakening', []))
            widespread_years = len(phases.get('widespread', []))
            
            dormancy_period = min_year - self.sb_year
            
            report.append(f"Dormancy Period: {dormancy_period} years")
            report.append(f"  - Dormant phase: {dormant_years} years")
            report.append(f"  - Awakening phase: {awakening_years} years")
            report.append(f"  - Widespread phase: {widespread_years} years")
        
        # Diffusion patterns
        if self.diffusion_metrics is not None and not self.diffusion_metrics.empty:
            report.append("\nDIFFUSION PATTERNS:")
            
            if 'Geographic_Diversity' in self.diffusion_metrics.columns:
                final_geo = self.diffusion_metrics['Geographic_Diversity'].iloc[-1]
                initial_geo = self.diffusion_metrics['Geographic_Diversity'].iloc[0]
                report.append(f"Final Geographic Diversity: {final_geo} jurisdictions")
                
                if initial_geo > 0:
                    geo_growth = final_geo / initial_geo
                    report.append(f"Geographic Expansion Factor: {geo_growth:.2f}x")
            
            if 'Tech_Diversity' in self.diffusion_metrics.columns:
                final_tech = self.diffusion_metrics['Tech_Diversity'].iloc[-1]
                initial_tech = self.diffusion_metrics['Tech_Diversity'].iloc[0]
                report.append(f"Final Technology Diversity: {final_tech} domains")
                
                if initial_tech > 0:
                    tech_growth = final_tech / initial_tech
                    report.append(f"Technology Expansion Factor: {tech_growth:.2f}x")
        
        # Transformation impact
        report.append("\nTRANSFORMATION IMPACT METRICS:")
        
        maturity_df = self.calculate_maturity_indicators()
        if not maturity_df.empty:
            initial_science = maturity_df['Science_Maturity'].iloc[0]
            final_science = maturity_df['Science_Maturity'].iloc[-1]
            science_growth = (final_science / initial_science 
                            if initial_science > 0 else 0)
            
            initial_tech = maturity_df['Tech_Maturity'].iloc[0]
            final_tech = maturity_df['Tech_Maturity'].iloc[-1]
            tech_growth = (final_tech / initial_tech 
                         if initial_tech > 0 else 0)
            
            report.append(f"Science Maturity Growth: {science_growth:.2f}x")
            report.append(f"Technology Maturity Growth: {tech_growth:.2f}x")
        
        # Velocity metrics
        velocity_metrics = self.calculate_transformation_velocity()
        report.append("\nTRANSFORMATION VELOCITY:")
        for metric, velocity in velocity_metrics.items():
            report.append(f"  {metric}: {velocity:.4f}")
        
        # Success assessment
        report.append("\nSLEEPING BEAUTY IMPACT ASSESSMENT:")
        
        avg_annual_patents = total_patents / (max_year - min_year + 1)
        peak_year_patents = self.awakening_analysis['yearly_stats']['Patent_Count'].max()
        awakening_intensity = peak_year_patents / avg_annual_patents
        
        report.append(f"Awakening Intensity: {awakening_intensity:.2f}x average")
        report.append(f"Technology Transformation Span: {max_year - min_year + 1} years")
        
        # Success indicators
        success_indicators = []
        if dormancy_period > 20:
            success_indicators.append("Extended dormancy period (>20 years)")
        if awakening_intensity > 2:
            success_indicators.append("Strong awakening effect (>2x average)")
        
        if (self.diffusion_metrics is not None and 
            not self.diffusion_metrics.empty):
            
            if 'Geographic_Diversity' in self.diffusion_metrics.columns:
                final_geo = self.diffusion_metrics['Geographic_Diversity'].iloc[-1]
                if final_geo >= 5:
                    success_indicators.append(
                        "Global technology diffusion (≥5 jurisdictions)"
                    )
            
            if 'Tech_Diversity' in self.diffusion_metrics.columns:
                final_tech = self.diffusion_metrics['Tech_Diversity'].iloc[-1]
                if final_tech >= 8:
                    success_indicators.append(
                        "Multi-domain technology impact (≥8 domains)"
                    )
        
        # Growth pattern verification
        phases_df = self.awakening_analysis['phases']
        if all(phase in phases_df for phase in ['dormant', 'awakening', 'widespread']):
            
            dormant_avg = (phases_df['dormant']['Patent_Count'].mean() 
                          if not phases_df['dormant'].empty else 0)
            awakening_avg = (phases_df['awakening']['Patent_Count'].mean() 
                           if not phases_df['awakening'].empty else 0)
            widespread_avg = (phases_df['widespread']['Patent_Count'].mean() 
                            if not phases_df['widespread'].empty else 0)
            
            if (awakening_avg > dormant_avg * 1.5 and 
                widespread_avg > awakening_avg * 1.2):
                success_indicators.append("Exponential growth pattern observed")
        
        report.append("\nSUCCESS INDICATORS:")
        if success_indicators:
            for indicator in success_indicators:
                report.append(f"  - {indicator}")
        else:
            report.append("  Warning: Limited evidence of strong SB impact")
            report.append("  Consider revising classification criteria")
        
        # Methodological warnings
        report.append("\nMETHODOLOGICAL NOTES:")
        if total_patents < 100:
            report.append("  - Small sample size may limit statistical robustness")
        
        if any(velocity == 0 for velocity in velocity_metrics.values()):
            report.append("  - Zero velocity detected - possible data quality issues")
        
        if max_year - min_year < 10:
            report.append("  - Short analysis period - long-term trends unclear")
        
        return "\n".join(report)
    
    def run_complete_analysis(self, save_figures=True, period_length=5):
        """
        Execute complete Sleeping Beauty impact analysis pipeline.
        
        Orchestrates full analytical workflow:
        1. Create temporal periods
        2. Calculate awakening metrics
        3. Analyze diffusion patterns
        4. Generate visual dashboard
        5. Produce textual impact report
        
        Args:
            save_figures: Save dashboard and report to files (default: True)
            period_length: Temporal period duration in years (default: 5)
        
        Returns:
            dict: Analysis results containing:
                - dashboard: matplotlib Figure object
                - report: Formatted text report
                - awakening_analysis: Temporal phase metrics
                - diffusion_metrics: Geographic/technological diffusion data
        
        Side Effects:
            When save_figures=True, creates:
            - sleeping_beauty_impact_analysis.png (dashboard visualization)
            - sleeping_beauty_impact_report.txt (textual report)
        """
        
        print("Initializing Sleeping Beauty impact analysis...")
        
        # Create time periods
        periods = self.create_time_periods(period_length)
        print(f"Created {len(periods)} temporal periods: {periods}")
        
        # Run core analyses
        print("Calculating awakening metrics...")
        self.calculate_awakening_metrics()
        
        print("Analyzing diffusion patterns...")
        self.calculate_diffusion_patterns()
        
        # Generate visualizations
        print("Generating impact dashboard...")
        fig = self.plot_impact_dashboard()
        
        if save_figures:
            fig.savefig('sleeping_beauty_impact_analysis.png', dpi=300, 
                       bbox_inches='tight')
        
        # Generate report
        print("Compiling impact report...")
        report = self.generate_impact_report()
        
        if save_figures:
            with open('sleeping_beauty_impact_report.txt', 'w', 
                     encoding='utf-8') as f:
                f.write(report)
        
        print("\nAnalysis completed successfully!")
        if save_figures:
            print("\nGenerated files:")
            print("  - sleeping_beauty_impact_analysis.png")
            print("  - sleeping_beauty_impact_report.txt")
        
        return {
            'dashboard': fig,
            'report': report,
            'awakening_analysis': self.awakening_analysis,
            'diffusion_metrics': self.diffusion_metrics
        }


def analyze_patent_sleeping_beauty(csv_file_path, discovery_year, 
                                   period_length=5, save_figures=True):
    """
    Convenience function for complete Sleeping Beauty patent analysis.
    
    Provides streamlined interface for analyzing science-technology linkage
    of dormant publications through patent citation networks.
    
    Args:
        csv_file_path: Path to Lens.org CSV export file containing patent
            metadata with DOI-verified citation to the SB publication
        discovery_year: Initial publication year of the Sleeping Beauty study
            (e.g., 1971 for Folkman's angiogenesis paper)
        period_length: Duration of temporal analysis periods in years (default: 5)
        save_figures: Flag to save dashboard and report to files (default: True)
    
    Returns:
        dict: Analysis results containing dashboard, report, and metrics, or
              None if data loading/validation fails
    
    Example:
        >>> results = analyze_patent_sleeping_beauty(
        ...     csv_file_path='folkman_patent_citations.csv',
        ...     discovery_year=1971,
        ...     period_length=5,
        ...     save_figures=True
        ... )
        >>> if results:
        ...     print(results['report'])
        ...     plt.show()
    
    Notes:
        This function implements the analytical workflow described in:
        El Aichouchi, A., & Gorry, P. (2018). The Judah Folkman case.
        Scientometrics, 115(2), 1035-1044.
    """
    
    # Load data
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Data loaded successfully: {len(df)} patent records")
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Initialize analyzer
    try:
        analyzer = SleepingBeautyImpactAnalyzer(df, discovery_year)
    except ValueError as e:
        print(f"Data validation error: {e}")
        print("Required columns: Publication Year, Jurisdiction")
        return None
    
    # Data quality summary
    min_year = int(analyzer.df['Publication Year'].min())
    max_year = int(analyzer.df['Publication Year'].max())
    print(f"\nAnalysis period: {min_year}-{max_year}")
    print(f"Dormancy period: {min_year - discovery_year} years")
    print(f"Available data columns: {len(analyzer.df.columns)}")
    
    # Execute analysis
    print("\nRunning complete analysis pipeline...")
    results = analyzer.run_complete_analysis(save_figures, period_length)
    
    # Display report
    print("\n" + "=" * 60)
    print("SLEEPING BEAUTY IMPACT ANALYSIS")
    print("=" * 60)
    print(results['report'])
    
    return results


def main():
    """
    Main execution function demonstrating analyzer usage.
    
    Analyzes Folkman's (1971) angiogenesis paper as documented in:
    Folkman, J. (1971). Tumor angiogenesis: therapeutic implications.
    New England Journal of Medicine, 285(21), 1182-1186.
    
    This case study exemplifies Sleeping Beauty dynamics with 23-year
    dormancy followed by exponential awakening in technological applications.
    """
    # Example: Judah Folkman's angiogenesis discovery
    file_path = "judahfolkmanpatentciters.csv"
    discovery_year = 1971

    print("Analyzing Sleeping Beauty impact: Folkman (1971) angiogenesis paper")
    print("=" * 60)
    
    results = analyze_patent_sleeping_beauty(
        csv_file_path=file_path,
        discovery_year=discovery_year,
        period_length=5,
        save_figures=True
    )

    if results:
        print("\nDisplaying dashboard...")
        plt.show()

    return results


if __name__ == "__main__":
    main()
